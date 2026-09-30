//! # Google Gemma 4 Neural Architecture in Pure Rust (Candle)
//!
//! ## Teaching Notes: Engineering Deep Learning Systems in the Age of AI
//! When AI assistants generate machine learning code, they frequently default to generic Transformer
//! architectures (standard LLaMA/GPT-2). However, cutting-edge foundation models (like Google Gemma 4)
//! introduce non-standard mathematical innovations that cause silent numerical divergence or gibberish output
//! if implemented with cookie-cutter assumptions.
//!
//! ### Key Architectural Innovations in Gemma 4:
//! 1. **Per-Layer Embeddings (PLE)**:
//!    Instead of a single static input embedding at layer 0, Gemma 4 dynamically conditions *every single layer*
//!    with a 256-dimensional layer-specific embedding vector computed from both token identity and context projection:
//!    $$\text{PLE}_l = \text{Norm}(\text{Proj}(x) \cdot \frac{1}{\sqrt{d_{model}}} + \text{Embed}_{\text{PLE}}(w) \cdot \sqrt{d_{\text{ple}}}) \cdot \frac{1}{\sqrt{2}}$$
//!    $$x = x + \text{Norm}(\text{Proj}(\text{GELU}(\text{Gate}(x)) \odot \text{PLE}_l))$$
//!
//! 2. **Upper-Layer KV-Cache Sharing (`num_kv_shared_layers: 20`)**:
//!    To slash VRAM consumption and maximize decode throughput, the top 20 layers (layers 15..34) do *not*
//!    compute separate Key/Value projections. They share the cached KV states from the lower layers.
//!
//! 3. **Heterogeneous Attention with Proportional RoPE**:
//!    - Sliding window attention layers use `head_dim = 256`, `rope_theta = 10,000.0`, and full rotation ($1.0$).
//!    - Full (global) attention layers use `global_head_dim = 512`, `rope_theta = 1,000,000.0`, and **partial rotary factor ($0.25$)**,
//!      meaning only the first 128 dimensions are rotated, while the remaining 384 dimensions pass through unrotated.
//!
//! 4. **Query/Key Normalization & Unit Scaling**:
//!    Because Queries and Keys pass through RMSNorm before attention score calculation, the attention dot-product
//!    scaling factor is $1.0$ (rather than the standard $1/\sqrt{d_k}$).

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

/// Top-level container matching the `config.json` schema of multimodal Gemma 4 models.
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4TopConfig {
    pub text_config: Gemma4TextConfig,
}

/// Structural hyperparameters defining the Gemma 4 language model backbone.
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub num_kv_shared_layers: Option<usize>,
    pub layer_types: Vec<String>,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub final_logit_softcapping: Option<f64>,
    pub sliding_window: Option<usize>,
    pub hidden_activation: String,
    pub hidden_size_per_layer_input: Option<usize>,
    pub vocab_size_per_layer_input: Option<usize>,
    pub use_double_wide_mlp: Option<bool>,
}

// -----------------------------------------------------------------------------
// 1. Root Mean Square Normalization (RMSNorm)
// -----------------------------------------------------------------------------

/// Root Mean Square Layer Normalization (with learned scale parameter).
///
/// Mathematical formulation:
/// $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
///
/// Note on Numerical Precision: We perform variance calculation in FP32 to avoid
/// underflow/overflow before casting back to target dtype (FP16/BF16).
#[derive(Clone, Debug)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let hidden_size = x.dim(candle_core::D::Minus1)?;
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = (x_f32.sqr()?.sum_keepdim(candle_core::D::Minus1)? / (hidden_size as f64))?;
        let rsqrt = (variance + self.eps)?.sqrt()?.recip()?;
        let x_normed = x_f32.broadcast_mul(&rsqrt)?.to_dtype(x_dtype)?;
        x_normed.broadcast_mul(&self.weight)
    }
}

/// Parameter-free Root Mean Square Normalization (Unit Scale).
///
/// Used specifically for Value normalization (`v_norm`) in Gemma 4 attention blocks.
#[derive(Clone, Debug)]
pub struct UnitRmsNorm {
    eps: f64,
}

impl UnitRmsNorm {
    pub fn new(eps: f64) -> Self {
        Self { eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let hidden_size = x.dim(candle_core::D::Minus1)?;
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = (x_f32.sqr()?.sum_keepdim(candle_core::D::Minus1)? / (hidden_size as f64))?;
        let rsqrt = (variance + self.eps)?.sqrt()?.recip()?;
        x_f32.broadcast_mul(&rsqrt)?.to_dtype(x_dtype)
    }
}

// -----------------------------------------------------------------------------
// 2. Rotary Position Embeddings (RoPE)
// -----------------------------------------------------------------------------

/// Precomputed Rotary Positional Embeddings supporting Standard and Proportional RoPE.
#[derive(Clone, Debug)]
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    /// Constructs standard RoPE frequencies for sliding-window layers (theta = 10,000).
    pub fn new_standard(dim: usize, max_pos: usize, theta: f64, dtype: DType, device: &Device) -> Result<Self> {
        let effective_max_pos = max_pos.min(2048);
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0f32 / (theta as f32).powf(i as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let t: Vec<f32> = (0..effective_max_pos).map(|i| i as f32).collect();
        let t = Tensor::new(t.as_slice(), device)?;
        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], candle_core::D::Minus1)?;
        let cos = emb.cos()?.to_dtype(dtype)?;
        let sin = emb.sin()?.to_dtype(dtype)?;
        Ok(Self { cos, sin })
    }

    /// Constructs proportional RoPE for global full-attention layers.
    ///
    /// Proportional RoPE rotates a fraction of the head dimension (`partial_factor = 0.25`),
    /// leaving the rest unrotated. By zero-padding `inv_freq` for non-rotated dimensions:
    /// $\cos(0) = 1.0$ and $\sin(0) = 0.0$, so unrotated channels pass through unmodified with zero branch penalties!
    pub fn new_proportional(dim: usize, max_pos: usize, theta: f64, partial_factor: f64, dtype: DType, device: &Device) -> Result<Self> {
        let effective_max_pos = max_pos.min(2048);
        let rope_angles = ((partial_factor * dim as f64 / 2.0) as usize).max(1);
        let nope_angles = (dim / 2).saturating_sub(rope_angles);

        let mut inv_freq: Vec<f32> = (0..rope_angles)
            .map(|i| 1.0f32 / (theta as f32).powf((2 * i) as f32 / dim as f32))
            .collect();
        inv_freq.extend(std::iter::repeat(0.0f32).take(nope_angles));

        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let t: Vec<f32> = (0..effective_max_pos).map(|i| i as f32).collect();
        let t = Tensor::new(t.as_slice(), device)?;
        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], candle_core::D::Minus1)?;
        let cos = emb.cos()?.to_dtype(dtype)?;
        let sin = emb.sin()?.to_dtype(dtype)?;
        Ok(Self { cos, sin })
    }

    /// Applies 2D complex rotary embedding rotation:
    /// $$R_{\Theta} x = (x \odot \cos) + (\text{rotate\_half}(x) \odot \sin)$$
    pub fn apply(&self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b, _h, seq_len, head_dim) = x.dims4()?;
        let cos = self.cos.narrow(0, pos, seq_len)?;
        let sin = self.sin.narrow(0, pos, seq_len)?;
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        let x1 = x.narrow(candle_core::D::Minus1, 0, head_dim / 2)?;
        let x2 = x.narrow(candle_core::D::Minus1, head_dim / 2, head_dim / 2)?;
        let rotate_x = Tensor::cat(&[&x2.neg()?, &x1], candle_core::D::Minus1)?;
        x.broadcast_mul(&cos)? + rotate_x.broadcast_mul(&sin)?
    }
}

// -----------------------------------------------------------------------------
// 3. Gated Feed-Forward Network (MLP)
// -----------------------------------------------------------------------------

/// Gated Multi-Layer Perceptron using GeGLU activation.
///
/// $$\text{MLP}(x) = \text{DownProj}(\text{GELU}(\text{GateProj}(x)) \odot \text{UpProj}(x))$$
#[derive(Clone, Debug)]
pub struct Gemma4Mlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl Gemma4Mlp {
    pub fn new(cfg: &Gemma4TextConfig, is_double_wide: bool, vb: VarBuilder) -> Result<Self> {
        let intermediate_size = if is_double_wide {
            cfg.intermediate_size * 2
        } else {
            cfg.intermediate_size
        };
        let gate_proj = candle_nn::linear_no_bias(cfg.hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_no_bias(cfg.hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear_no_bias(intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self { gate_proj, up_proj, down_proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.gelu_erf()?;
        let up = self.up_proj.forward(x)?;
        let down = self.down_proj.forward(&(gate * up)?)?;
        Ok(down)
    }
}

// -----------------------------------------------------------------------------
// 4. Multi-Head Attention with KV-Sharing & GQA
// -----------------------------------------------------------------------------

/// Multi-Head Attention block supporting Grouped-Query Attention (GQA) and KV Sharing.
#[derive(Clone, Debug)]
pub struct Gemma4Attention {
    q_proj: candle_nn::Linear,
    k_proj: Option<candle_nn::Linear>,
    v_proj: Option<candle_nn::Linear>,
    o_proj: candle_nn::Linear,
    q_norm: RmsNorm,
    k_norm: Option<RmsNorm>,
    v_norm: UnitRmsNorm,
    rope: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    is_kv_shared: bool,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Gemma4Attention {
    pub fn new(cfg: &Gemma4TextConfig, is_global: bool, is_kv_shared: bool, rope: RotaryEmbedding, vb: VarBuilder) -> Result<Self> {
        let head_dim = if is_global { cfg.global_head_dim } else { cfg.head_dim };

        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;

        let q_proj = candle_nn::linear_no_bias(cfg.hidden_size, q_dim, vb.pp("q_proj"))?;
        let o_proj = candle_nn::linear_no_bias(q_dim, cfg.hidden_size, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;

        // Layers that share KV do not instantiate their own K or V projection weight matrices.
        let (k_proj, v_proj, k_norm) = if !is_kv_shared {
            let k = candle_nn::linear_no_bias(cfg.hidden_size, kv_dim, vb.pp("k_proj"))?;
            let v = candle_nn::linear_no_bias(cfg.hidden_size, kv_dim, vb.pp("v_proj"))?;
            let kn = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
            (Some(k), Some(v), Some(kn))
        } else {
            (None, None, None)
        };

        let v_norm = UnitRmsNorm::new(cfg.rms_norm_eps);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            rope,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            is_kv_shared,
            kv_cache: None,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        pos: usize,
        shared_kv: Option<&(Tensor, Tensor)>,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let (b, seq_len, _h) = x.dims3()?;

        // Project and normalize Query states
        let q = self.q_proj.forward(x)?;
        let q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let q = self.q_norm.forward(&q)?;
        let q = self.rope.apply(&q, pos)?.contiguous()?;

        // Retrieve or compute Key and Value states
        let (k, v, produced_shared_kv) = if self.is_kv_shared {
            let (sk, sv) = shared_kv.expect("Shared KV cache missing for upper layer");
            (sk.clone(), sv.clone(), None)
        } else {
            let k = self.k_proj.as_ref().unwrap().forward(x)?;
            let v = self.v_proj.as_ref().unwrap().forward(x)?;

            let k = k.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
            let v = v.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

            let k = self.k_norm.as_ref().unwrap().forward(&k)?;
            let k = self.rope.apply(&k, pos)?.contiguous()?;
            let v = self.v_norm.forward(&v)?.contiguous()?;

            // Update persistent KV Cache across autoregressive decode steps
            let (full_k, full_v) = match &self.kv_cache {
                Some((prev_k, prev_v)) => {
                    let k_cat = Tensor::cat(&[prev_k, &k], 2)?;
                    let v_cat = Tensor::cat(&[prev_v, &v], 2)?;
                    (k_cat, v_cat)
                }
                None => (k, v),
            };
            self.kv_cache = Some((full_k.clone(), full_v.clone()));

            (full_k.clone(), full_v.clone(), Some((full_k, full_v)))
        };

        let kv_len = k.dim(2)?;

        // Broadcast KV heads across Query heads for Grouped-Query Attention (GQA)
        let k_exp = if self.num_kv_heads != self.num_heads {
            k.broadcast_as((b, self.num_heads, kv_len, self.head_dim))?.contiguous()?
        } else {
            k.contiguous()?
        };

        let v_exp = if self.num_kv_heads != self.num_heads {
            v.broadcast_as((b, self.num_heads, kv_len, self.head_dim))?.contiguous()?
        } else {
            v.contiguous()?
        };

        // Attention score computation (Scale = 1.0 due to QK normalization)
        let k_t = k_exp.transpose(2, 3)?.contiguous()?;
        let mut att = q.matmul(&k_t)?;

        // Causal masking for prompt prefill phase
        if seq_len > 1 {
            let mask: Vec<f32> = (0..seq_len)
                .flat_map(|i| (0..kv_len).map(move |j| if j > i { -1e4f32 } else { 0.0 }))
                .collect();
            let mask = Tensor::from_vec(mask, (1, 1, seq_len, kv_len), x.device())?.to_dtype(x.dtype())?;
            att = att.broadcast_add(&mask)?;
        }

        let att = candle_nn::ops::softmax_last_dim(&att)?.contiguous()?;
        let out = att.matmul(&v_exp)?;

        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, seq_len, self.num_heads * self.head_dim))?;
        let final_out = self.o_proj.forward(&out)?;
        Ok((final_out, produced_shared_kv))
    }

    #[allow(dead_code)]
    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

// -----------------------------------------------------------------------------
// 5. Decoder Layer with Per-Layer Embeddings (PLE)
// -----------------------------------------------------------------------------

/// Single Gemma 4 Transformer Decoder Layer with 4 RMSNorms and PLE residual injection.
#[derive(Clone, Debug)]
pub struct Gemma4DecoderLayer {
    self_attn: Gemma4Attention,
    mlp: Gemma4Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: Option<RmsNorm>,
    pre_feedforward_layernorm: Option<RmsNorm>,
    post_feedforward_layernorm: Option<RmsNorm>,
    layer_scalar: Option<Tensor>,
    per_layer_input_gate: Option<candle_nn::Linear>,
    per_layer_projection: Option<candle_nn::Linear>,
    post_per_layer_input_norm: Option<RmsNorm>,
    pub is_global: bool,
}

impl Gemma4DecoderLayer {
    pub fn new(cfg: &Gemma4TextConfig, layer_idx: usize, rope: RotaryEmbedding, vb: VarBuilder) -> Result<Self> {
        let is_global = cfg.layer_types.get(layer_idx).map(|s| s == "full_attention").unwrap_or(false);
        let num_shared = cfg.num_kv_shared_layers.unwrap_or(0);
        let first_shared_layer = cfg.num_hidden_layers.saturating_sub(num_shared);
        let is_kv_shared = layer_idx >= first_shared_layer;

        let self_attn = Gemma4Attention::new(cfg, is_global, is_kv_shared, rope, vb.pp("self_attn"))?;

        let is_double_wide = cfg.use_double_wide_mlp.unwrap_or(false) && layer_idx >= first_shared_layer;
        let mlp = Gemma4Mlp::new(cfg, is_double_wide, vb.pp("mlp"))?;

        let input_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm")).ok();
        let pre_feedforward_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("pre_feedforward_layernorm")).ok();
        let post_feedforward_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_feedforward_layernorm")).ok();
        let layer_scalar = vb.get(1, "layer_scalar").ok();

        let ple_dim = cfg.hidden_size_per_layer_input.unwrap_or(256);
        let (per_layer_input_gate, per_layer_projection, post_per_layer_input_norm) = if cfg.hidden_size_per_layer_input.is_some() {
            let gate = candle_nn::linear_no_bias(cfg.hidden_size, ple_dim, vb.pp("per_layer_input_gate")).ok();
            let proj = candle_nn::linear_no_bias(ple_dim, cfg.hidden_size, vb.pp("per_layer_projection")).ok();
            let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_per_layer_input_norm")).ok();
            (gate, proj, norm)
        } else {
            (None, None, None)
        };

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            layer_scalar,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            is_global,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        per_layer_input: Option<&Tensor>,
        pos: usize,
        shared_kv: Option<&(Tensor, Tensor)>,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        // Block 1: Self-Attention with Pre & Post Layernorms
        let residual = x;
        let normed = self.input_layernorm.forward(x)?;
        let (mut attn_out, produced_shared_kv) = self.self_attn.forward(&normed, pos, shared_kv)?;
        if let Some(norm) = &self.post_attention_layernorm {
            attn_out = norm.forward(&attn_out)?;
        }
        let mut x = (residual + attn_out)?;

        // Block 2: Feed-Forward MLP with Pre & Post Layernorms
        let residual = &x;
        let normed = if let Some(norm) = &self.pre_feedforward_layernorm {
            norm.forward(&x)?
        } else {
            x.clone()
        };
        let mut mlp_out = self.mlp.forward(&normed)?;
        if let Some(norm) = &self.post_feedforward_layernorm {
            mlp_out = norm.forward(&mlp_out)?;
        }
        x = (residual + mlp_out)?;

        // Block 3: Per-Layer Embedding (PLE) Injection
        if let (Some(gate), Some(proj), Some(norm), Some(ple_in)) = (
            &self.per_layer_input_gate,
            &self.per_layer_projection,
            &self.post_per_layer_input_norm,
            per_layer_input,
        ) {
            let residual = &x;
            let gated = gate.forward(&x)?.gelu_erf()?;
            let hidden = gated.broadcast_mul(ple_in)?;
            let projected = proj.forward(&hidden)?;
            let normed = norm.forward(&projected)?;
            x = (residual + normed)?;
        }

        // Layer-specific output scalar scaling
        if let Some(scalar) = &self.layer_scalar {
            x = x.broadcast_mul(scalar)?;
        }

        Ok((x, produced_shared_kv))
    }

    #[allow(dead_code)]
    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// -----------------------------------------------------------------------------
// 6. Top-Level Causal Language Model
// -----------------------------------------------------------------------------

/// Complete Gemma 4 Causal Language Model.
#[derive(Clone, Debug)]
pub struct Gemma4ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    embed_tokens_per_layer: Option<candle_nn::Embedding>,
    per_layer_model_projection: Option<candle_nn::Linear>,
    per_layer_projection_norm: Option<RmsNorm>,
    lm_head_weight: Tensor,
    layers: Vec<Gemma4DecoderLayer>,
    norm: RmsNorm,
    final_logit_softcapping: Option<f64>,
    hidden_size: usize,
    ple_dim: usize,
    shared_sliding_kv: Option<(Tensor, Tensor)>,
    shared_global_kv: Option<(Tensor, Tensor)>,
}

impl Gemma4ForCausalLM {
    pub fn new(cfg: &Gemma4TextConfig, vb: VarBuilder) -> Result<Self> {
        let vb_lm = if vb.contains_tensor("model.language_model.embed_tokens.weight") {
            vb.pp("model.language_model")
        } else if vb.contains_tensor("language_model.embed_tokens.weight") {
            vb.pp("language_model")
        } else if vb.contains_tensor("model.embed_tokens.weight") {
            vb.pp("model")
        } else {
            vb.clone()
        };

        // Primary word embedding
        println!("   -> Loading embed_tokens...");
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_lm.pp("embed_tokens"))?;

        // Tied word embedding: output projection reuses transpose of token embedding weights
        println!("   -> Computing tied lm_head_weight...");
        let lm_head_weight = embed_tokens.embeddings().t()?.contiguous()?;

        let ple_dim = cfg.hidden_size_per_layer_input.unwrap_or(256);
        let total_ple_size = cfg.num_hidden_layers * ple_dim;

        // Per-Layer Embedding parameter tensors
        println!("   -> Loading PLE parameters...");
        let embed_tokens_per_layer = candle_nn::embedding(cfg.vocab_size, total_ple_size, vb_lm.pp("embed_tokens_per_layer")).ok();
        let per_layer_model_projection = candle_nn::linear_no_bias(cfg.hidden_size, total_ple_size, vb_lm.pp("per_layer_model_projection")).ok();
        let per_layer_projection_norm = RmsNorm::new(ple_dim, cfg.rms_norm_eps, vb_lm.pp("per_layer_projection_norm")).ok();

        println!("   -> Initializing RoPE tables...");
        let dtype = vb.dtype();
        let rope_sliding = RotaryEmbedding::new_standard(cfg.head_dim, cfg.max_position_embeddings, 10_000.0, dtype, vb.device())?;
        let rope_global = RotaryEmbedding::new_proportional(cfg.global_head_dim, cfg.max_position_embeddings, 1_000_000.0, 0.25, dtype, vb.device())?;

        println!("   -> Instantiating {} decoder layers...", cfg.num_hidden_layers);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb_lm.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            let is_global = cfg.layer_types.get(i).map(|s| s == "full_attention").unwrap_or(false);
            let rope = if is_global { rope_global.clone() } else { rope_sliding.clone() };
            layers.push(Gemma4DecoderLayer::new(cfg, i, rope, vb_layers.pp(i))?);
        }

        println!("   -> Loading final norm...");
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_lm.pp("norm"))?;

        Ok(Self {
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            lm_head_weight,
            layers,
            norm,
            final_logit_softcapping: cfg.final_logit_softcapping,
            hidden_size: cfg.hidden_size,
            ple_dim,
            shared_sliding_kv: None,
            shared_global_kv: None,
        })
    }

    /// Encodes token IDs into base text embeddings scaled by sqrt(hidden_size).
    pub fn embed_text(&self, input_ids: &Tensor) -> Result<Tensor> {
        let raw_embeds = self.embed_tokens.forward(input_ids)?;
        raw_embeds * (self.hidden_size as f64).sqrt()
    }

    /// Primary forward pass receiving discrete token IDs.
    pub fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let inputs_embeds = self.embed_text(input_ids)?;
        self.forward_with_embeds(&inputs_embeds, input_ids, pos)
    }

    /// Multimodal forward pass receiving pre-fused embeddings (e.g. text + vision tokens).
    pub fn forward_with_embeds(
        &mut self,
        inputs_embeds: &Tensor,
        input_ids: &Tensor,
        pos: usize,
    ) -> Result<Tensor> {
        let (b, seq_len, _h) = inputs_embeds.dims3()?;

        // 1. Dual-Component Per-Layer Embedding (PLE) Calculation
        let ple_tensor = if let (Some(embed_ple), Some(proj_ple), Some(norm_ple)) = (
            &self.embed_tokens_per_layer,
            &self.per_layer_model_projection,
            &self.per_layer_projection_norm,
        ) {
            let num_layers = self.layers.len();

            // Token-identity component scaled by sqrt(ple_dim)
            let token_ple = (embed_ple.forward(input_ids)? * (self.ple_dim as f64).sqrt())?;
            let token_ple = token_ple.reshape((b, seq_len, num_layers, self.ple_dim))?;

            // Context-projection component scaled by 1/sqrt(hidden_size)
            let context_proj = (proj_ple.forward(inputs_embeds)? * (1.0 / (self.hidden_size as f64).sqrt()))?;
            let context_proj = context_proj.reshape((b, seq_len, num_layers, self.ple_dim))?;
            let context_ple = norm_ple.forward(&context_proj)?;

            // Combine both components with 1/sqrt(2) normalization
            let combined = ((token_ple + context_ple)? * (1.0 / 2.0f64.sqrt()))?;
            Some(combined)
        } else {
            None
        };

        // 2. Sequential Layer Execution with KV-Sharing
        let mut x = inputs_embeds.clone();

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let layer_ple = if let Some(ple) = &ple_tensor {
                Some(ple.narrow(2, i, 1)?.squeeze(2)?)
            } else {
                None
            };

            let shared_kv = if layer.is_global {
                self.shared_global_kv.as_ref()
            } else {
                self.shared_sliding_kv.as_ref()
            };

            let (next_x, produced_kv) = layer.forward(&x, layer_ple.as_ref(), pos, shared_kv)?;
            x = next_x;

            if let Some(kv) = produced_kv {
                if layer.is_global {
                    self.shared_global_kv = Some(kv);
                } else {
                    self.shared_sliding_kv = Some(kv);
                }
            }
        }

        // 3. Final RMS Normalization
        let x = self.norm.forward(&x)?;
        let last_hidden = x.narrow(1, seq_len - 1, 1)?.squeeze(1)?;

        // 4. Output Projection onto Vocabulary
        let logits = last_hidden.matmul(&self.lm_head_weight)?;
        let mut logits = logits.to_dtype(DType::F32)?;

        // Optional logit softcapping: cap * tanh(logits / cap)
        if let Some(cap) = self.final_logit_softcapping {
            logits = ((&logits / cap)?.tanh()? * cap)?;
        }

        Ok(logits)
    }

    #[allow(dead_code)]
    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
        self.shared_sliding_kv = None;
        self.shared_global_kv = None;
    }
}
