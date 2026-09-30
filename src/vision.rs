//! # Google Gemma 4 Vision Tower & Multimodal Encoder in Pure Rust (Candle)
//!
//! Implements the 16-layer Vision Transformer (ViT) patch embedder, heterogeneous 2D positional
//! embedding table, RMSNorm-gated vision attention blocks, and multimodal projection head
//! that bridges image pixels to the language model embedding space.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

/// Configuration hyperparameters for the Gemma 4 Vision Tower.
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub position_embedding_size: usize,
    pub rms_norm_eps: f64,
    pub default_output_length: usize,
}

impl Default for Gemma4VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 16,
            num_attention_heads: 12,
            num_key_value_heads: 12,
            head_dim: 64,
            patch_size: 16,
            pooling_kernel_size: 3,
            position_embedding_size: 10240,
            rms_norm_eps: 1e-6,
            default_output_length: 280,
        }
    }
}

/// Vision RMS Normalization block.
#[derive(Clone, Debug)]
pub struct VisionRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl VisionRmsNorm {
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

/// Vision Rotary Position Embeddings (RoPE with base frequency theta = 100.0).
#[derive(Clone, Debug)]
pub struct VisionRotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl VisionRotaryEmbedding {
    pub fn new(dim: usize, max_pos: usize, theta: f64, dtype: DType, device: &Device) -> Result<Self> {
        let effective_max_pos = max_pos.min(4096);
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

/// Vision Patch Embedder converting image pixel grids into 768-dimensional token sequences.
#[derive(Clone, Debug)]
pub struct Gemma4PatchEmbedder {
    input_proj: candle_nn::Linear,
    position_embedding_table: Tensor,
    patch_size: usize,
    _hidden_size: usize,
}

impl Gemma4PatchEmbedder {
    pub fn new(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_dim = 3 * cfg.patch_size * cfg.patch_size; // 3 * 16 * 16 = 768
        let input_proj = candle_nn::linear_no_bias(patch_dim, cfg.hidden_size, vb.pp("input_proj"))?;
        let position_embedding_table = vb.get(
            (2, cfg.position_embedding_size, cfg.hidden_size),
            "position_embedding_table",
        )?;
        Ok(Self {
            input_proj,
            position_embedding_table,
            patch_size: cfg.patch_size,
            _hidden_size: cfg.hidden_size,
        })
    }

    /// Forward pass taking a raw image tensor `(Batch, Channels, Height, Width)`
    /// and producing a sequence of embedded patch tokens with 2D positional embeddings.
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = pixel_values.dims4()?;
        let p = self.patch_size;
        let num_patches_h = h / p;
        let num_patches_w = w / p;
        let num_patches = num_patches_h * num_patches_w;

        // Reshape image into patches: (B, C, num_patches_h, p, num_patches_w, p)
        // -> Permute to (B, num_patches_h, num_patches_w, C, p, p) -> Flatten to (B, num_patches, C*p*p)
        let patches = pixel_values
            .reshape((b, c, num_patches_h, p, num_patches_w, p))?
            .permute((0, 2, 4, 1, 3, 5))?
            .contiguous()?
            .reshape((b, num_patches, c * p * p))?;

        // Linear patch projection to hidden_size (768)
        let patch_embeddings = self.input_proj.forward(&patches)?;

        // Extract 2D Positional Embeddings: (Row Pos + Col Pos)
        let row_pos = self.position_embedding_table.narrow(0, 0, 1)?.squeeze(0)?;
        let col_pos = self.position_embedding_table.narrow(0, 1, 1)?.squeeze(0)?;

        let mut pos_embeds = Vec::with_capacity(num_patches);
        for row in 0..num_patches_h {
            for col in 0..num_patches_w {
                let r_emb = row_pos.narrow(0, row, 1)?;
                let c_emb = col_pos.narrow(0, col, 1)?;
                pos_embeds.push((r_emb + c_emb)?);
            }
        }
        let pos_tensor = Tensor::cat(&pos_embeds.iter().collect::<Vec<_>>(), 0)?;
        let pos_tensor = pos_tensor.unsqueeze(0)?.to_dtype(patch_embeddings.dtype())?;

        patch_embeddings.broadcast_add(&pos_tensor)
    }
}

/// Vision Gated MLP block (GeGLU).
#[derive(Clone, Debug)]
pub struct Gemma4VisionMlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl Gemma4VisionMlp {
    pub fn new(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("gate_proj").pp("linear"),
        ).or_else(|_| candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj")))?;

        let up_proj = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("up_proj").pp("linear"),
        ).or_else(|_| candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj")))?;

        let down_proj = candle_nn::linear_no_bias(
            cfg.intermediate_size,
            cfg.hidden_size,
            vb.pp("down_proj").pp("linear"),
        ).or_else(|_| candle_nn::linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj")))?;

        Ok(Self { gate_proj, up_proj, down_proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.gelu_erf()?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Vision Multi-Head Self-Attention with QK RMS Normalization.
#[derive(Clone, Debug)]
pub struct Gemma4VisionAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    q_norm: VisionRmsNorm,
    k_norm: VisionRmsNorm,
    rope: VisionRotaryEmbedding,
    num_heads: usize,
    head_dim: usize,
}

impl Gemma4VisionAttention {
    pub fn new(cfg: &Gemma4VisionConfig, rope: VisionRotaryEmbedding, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.hidden_size;
        let q_proj = candle_nn::linear_no_bias(dim, dim, vb.pp("q_proj").pp("linear"))
            .or_else(|_| candle_nn::linear_no_bias(dim, dim, vb.pp("q_proj")))?;
        let k_proj = candle_nn::linear_no_bias(dim, dim, vb.pp("k_proj").pp("linear"))
            .or_else(|_| candle_nn::linear_no_bias(dim, dim, vb.pp("k_proj")))?;
        let v_proj = candle_nn::linear_no_bias(dim, dim, vb.pp("v_proj").pp("linear"))
            .or_else(|_| candle_nn::linear_no_bias(dim, dim, vb.pp("v_proj")))?;
        let o_proj = candle_nn::linear_no_bias(dim, dim, vb.pp("o_proj").pp("linear"))
            .or_else(|_| candle_nn::linear_no_bias(dim, dim, vb.pp("o_proj")))?;

        let q_norm = VisionRmsNorm::new(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = VisionRmsNorm::new(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rope,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq_len, _h) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let q = self.rope.apply(&q, 0)?.contiguous()?;
        let k = self.rope.apply(&k, 0)?.contiguous()?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let att = q.matmul(&k_t)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?.contiguous()?;
        let out = att.matmul(&v)?;

        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, seq_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }
}

/// Single Gemma 4 Vision Transformer Layer.
#[derive(Clone, Debug)]
pub struct Gemma4VisionLayer {
    self_attn: Gemma4VisionAttention,
    mlp: Gemma4VisionMlp,
    input_layernorm: VisionRmsNorm,
    post_attention_layernorm: Option<VisionRmsNorm>,
    pre_feedforward_layernorm: Option<VisionRmsNorm>,
    post_feedforward_layernorm: Option<VisionRmsNorm>,
}

impl Gemma4VisionLayer {
    pub fn new(cfg: &Gemma4VisionConfig, rope: VisionRotaryEmbedding, vb: VarBuilder) -> Result<Self> {
        let self_attn = Gemma4VisionAttention::new(cfg, rope, vb.pp("self_attn"))?;
        let mlp = Gemma4VisionMlp::new(cfg, vb.pp("mlp"))?;

        let input_layernorm = VisionRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = VisionRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm")).ok();
        let pre_feedforward_layernorm = VisionRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("pre_feedforward_layernorm")).ok();
        let post_feedforward_layernorm = VisionRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_feedforward_layernorm")).ok();

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let normed = self.input_layernorm.forward(x)?;
        let mut attn_out = self.self_attn.forward(&normed)?;
        if let Some(norm) = &self.post_attention_layernorm {
            attn_out = norm.forward(&attn_out)?;
        }
        let mut x = (residual + attn_out)?;

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
        Ok(x)
    }
}

/// Complete Gemma 4 Vision Tower (Patch Embedder + 16 Vision Encoder Layers).
#[derive(Clone, Debug)]
pub struct Gemma4VisionTower {
    patch_embedder: Gemma4PatchEmbedder,
    layers: Vec<Gemma4VisionLayer>,
}

impl Gemma4VisionTower {
    pub fn new(cfg: &Gemma4VisionConfig, vb: VarBuilder) -> Result<Self> {
        let vb_tower = if vb.contains_tensor("model.vision_tower.patch_embedder.input_proj.weight") {
            vb.pp("model.vision_tower")
        } else if vb.contains_tensor("vision_tower.patch_embedder.input_proj.weight") {
            vb.pp("vision_tower")
        } else {
            vb.clone()
        };

        let patch_embedder = Gemma4PatchEmbedder::new(cfg, vb_tower.pp("patch_embedder"))?;
        let rope = VisionRotaryEmbedding::new(cfg.head_dim, 4096, 100.0, vb.dtype(), vb.device())?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb_tower.pp("encoder").pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Gemma4VisionLayer::new(cfg, rope.clone(), vb_layers.pp(i))?);
        }

        Ok(Self {
            patch_embedder,
            layers,
        })
    }

    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let mut x = self.patch_embedder.forward(pixel_values)?;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
}

/// Multimodal Vision Projector connecting 768-dim vision outputs to 1536-dim text embedding space.
#[derive(Clone, Debug)]
pub struct Gemma4EmbedVision {
    embedding_projection: candle_nn::Linear,
}

impl Gemma4EmbedVision {
    pub fn new(vision_dim: usize, text_dim: usize, vb: VarBuilder) -> Result<Self> {
        let vb_proj = if vb.contains_tensor("model.embed_vision.embedding_projection.weight") {
            vb.pp("model.embed_vision")
        } else if vb.contains_tensor("embed_vision.embedding_projection.weight") {
            vb.pp("embed_vision")
        } else {
            vb.clone()
        };

        let embedding_projection = candle_nn::linear_no_bias(
            vision_dim,
            text_dim,
            vb_proj.pp("embedding_projection"),
        )?;

        Ok(Self { embedding_projection })
    }

    pub fn forward(&self, vision_tokens: &Tensor) -> Result<Tensor> {
        self.embedding_projection.forward(vision_tokens)
    }
}
