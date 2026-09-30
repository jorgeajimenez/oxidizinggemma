//! # Google Gemma 4 Multimodal Architecture (Conditional Generation)
//!
//! Unifies the 16-layer Vision Transformer (`vision_tower`), Multimodal Projector (`embed_vision`),
//! and 35-layer Causal Language Model backbone into an end-to-end vision-language inference engine.

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

use crate::gemma4::{Gemma4ForCausalLM, Gemma4TextConfig};
use crate::vision::{Gemma4EmbedVision, Gemma4VisionConfig, Gemma4VisionTower};

/// Full Gemma 4 Multimodal Configuration schema matching `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4MultimodalConfig {
    pub text_config: Gemma4TextConfig,
    pub vision_config: Option<Gemma4VisionConfig>,
    pub image_token_id: Option<u32>,
    pub boi_token_id: Option<u32>,
    pub eoi_token_id: Option<u32>,
    pub vision_soft_tokens_per_image: Option<usize>,
}

/// End-to-end multimodal model for image + text understanding and generation.
pub struct Gemma4ForConditionalGeneration {
    pub vision_tower: Option<Gemma4VisionTower>,
    pub embed_vision: Option<Gemma4EmbedVision>,
    pub language_model: Gemma4ForCausalLM,
    pub image_token_id: u32,
    pub boi_token_id: u32,
    pub eoi_token_id: u32,
    pub vision_soft_tokens_per_image: usize,
}

impl Gemma4ForConditionalGeneration {
    pub fn new(cfg: &Gemma4MultimodalConfig, vb: VarBuilder) -> Result<Self> {
        println!("1. Initializing Language Model Backbone (35 layers)...");
        let language_model = Gemma4ForCausalLM::new(&cfg.text_config, vb.clone())
            .map_err(|e| anyhow::anyhow!("LM initialization failed: {}", e))?;

        let (vision_tower, embed_vision) = if let Some(v_cfg) = &cfg.vision_config {
            println!("2. Initializing Vision Transformer Tower (16 layers, patch_size={})...", v_cfg.patch_size);
            let tower = match Gemma4VisionTower::new(v_cfg, vb.clone()) {
                Ok(t) => {
                    println!("   Vision tower loaded successfully.");
                    Some(t)
                }
                Err(e) => {
                    eprintln!("   Vision tower weights not found or partial: {}", e);
                    None
                }
            };

            let proj = match Gemma4EmbedVision::new(v_cfg.hidden_size, cfg.text_config.hidden_size, vb.clone()) {
                Ok(p) => {
                    println!("   Multimodal projection head loaded ({} -> {}).", v_cfg.hidden_size, cfg.text_config.hidden_size);
                    Some(p)
                }
                Err(e) => {
                    eprintln!("   Multimodal projector weights not found or partial: {}", e);
                    None
                }
            };

            (tower, proj)
        } else {
            (None, None)
        };

        let image_token_id = cfg.image_token_id.unwrap_or(258880);
        let boi_token_id = cfg.boi_token_id.unwrap_or(255999);
        let eoi_token_id = cfg.eoi_token_id.unwrap_or(258882);
        let vision_soft_tokens_per_image = cfg.vision_soft_tokens_per_image.unwrap_or(280);

        Ok(Self {
            vision_tower,
            embed_vision,
            language_model,
            image_token_id,
            boi_token_id,
            eoi_token_id,
            vision_soft_tokens_per_image,
        })
    }

    /// Encodes an image or video tensor `(B, 3, H, W)` into multimodal soft tokens.
    /// For multi-frame video inputs (B > 1), frames are concatenated across the sequence dimension `(1, B * NumPatches, 1536)`.
    pub fn encode_image(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let tower = self.vision_tower.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vision tower is not initialized in this model instance"))?;
        let projector = self.embed_vision.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Multimodal projector is not initialized in this model instance"))?;

        // 1. Vision Transformer Forward Pass -> (B, NumPatches, 768)
        let vision_features = tower.forward(pixel_values)
            .map_err(|e| anyhow::anyhow!("Vision tower forward pass failed: {}", e))?;

        // 2. Multimodal Linear Projection -> (B, NumPatches, 1536)
        let projected_embeddings = projector.forward(&vision_features)
            .map_err(|e| anyhow::anyhow!("Multimodal projector forward pass failed: {}", e))?;

        let (b, num_patches, hidden_dim) = projected_embeddings.dims3()?;
        if b > 1 {
            let flattened = projected_embeddings.reshape((1, b * num_patches, hidden_dim))?;
            Ok(flattened)
        } else {
            Ok(projected_embeddings)
        }
    }

    /// Fuses prompt text token IDs and optional image pixels into a single multimodal sequence.
    pub fn fuse_multimodal_embeddings(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
    ) -> Result<Tensor> {
        let base_text_embeds = self.language_model.embed_text(input_ids)
            .map_err(|e| anyhow::anyhow!("Text embedding failed: {}", e))?;

        if let Some(pixels) = pixel_values {
            let vision_embeds = self.encode_image(pixels)?;
            let (_b, seq_len, _h) = base_text_embeds.dims3()?;
            let (_vb, _v_len, _vh) = vision_embeds.dims3()?;

            let token_vec = input_ids.to_vec2::<u32>()
                .map_err(|e| anyhow::anyhow!("Failed reading token IDs: {}", e))?;
            let tokens = &token_vec[0];

            // Locate image placeholder tokens or prepend vision tokens
            let mut image_pos = None;
            for (idx, &tok) in tokens.iter().enumerate() {
                if tok == self.image_token_id || tok == self.boi_token_id {
                    image_pos = Some(idx);
                    break;
                }
            }

            match image_pos {
                Some(pos) => {
                    // Splice vision tokens into the text embedding stream
                    let prefix = base_text_embeds.narrow(1, 0, pos)?;
                    let suffix = if pos + 1 < seq_len {
                        base_text_embeds.narrow(1, pos + 1, seq_len - (pos + 1))?
                    } else {
                        Tensor::zeros((1, 0, _h), base_text_embeds.dtype(), base_text_embeds.device())?
                    };

                    let fused = Tensor::cat(&[&prefix, &vision_embeds, &suffix], 1)
                        .map_err(|e| anyhow::anyhow!("Tensor concatenation failed: {}", e))?;
                    Ok(fused)
                }
                None => {
                    // Prepend vision features before user prompt
                    let fused = Tensor::cat(&[&vision_embeds, &base_text_embeds], 1)
                        .map_err(|e| anyhow::anyhow!("Tensor prepending failed: {}", e))?;
                    Ok(fused)
                }
            }
        } else {
            Ok(base_text_embeds)
        }
    }

    /// Multimodal forward pass for conditional generation.
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        pos: usize,
    ) -> Result<Tensor> {
        let fused_embeds = self.fuse_multimodal_embeddings(input_ids, pixel_values)?;
        self.language_model.forward_with_embeds(&fused_embeds, input_ids, pos)
            .map_err(|e| anyhow::anyhow!("LM forward pass failed: {}", e))
    }

    /// Autoregressive single-token decode pass (uses text cache directly).
    pub fn decode_step(&mut self, next_token: u32, pos: usize, device: &Device) -> Result<Tensor> {
        let token_tensor = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        self.language_model.forward(&token_tensor, pos)
            .map_err(|e| anyhow::anyhow!("Decode forward pass failed: {}", e))
    }

    /// Clears the KV cache across all layers.
    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache();
    }
}
