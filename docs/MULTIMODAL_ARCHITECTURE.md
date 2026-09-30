# Gemma 4 Multimodal Architecture: Vision, Video, and Audio Internals

This document details the **multimodal vision, video, and audio pipeline** implemented in pure Rust inside [`src/vision.rs`](../src/vision.rs), [`src/multimodal.rs`](../src/multimodal.rs), and [`src/bin/multimodal_cli.rs`](../src/bin/multimodal_cli.rs).

---

## End-to-End Multimodal Vision and Video Pipeline

```mermaid
flowchart TD
    subgraph InputVisual ["Visual Input Processing"]
        InputType{"Input Type?"}
        ImageFile["Single Image (JPEG / PNG / WebP)"]
        VideoFile["Video File (MP4 / MOV / AVI / WebM)"]
        
        InputType -->|"Image"| ImageFile
        InputType -->|"Video"| VideoFile
        
        SingleTensor["load_single_image()<br/>Tensor: (1, 3, 256, 256)"]
        VideoTensor["load_video_frames() via ffmpeg<br/>Tensor: (N_frames, 3, 256, 256)"]
        
        ImageFile --> SingleTensor
        VideoFile --> VideoTensor
    end

    subgraph VisionPipeline ["Vision Transformer (ViT)"]
        PatchEmbed["Gemma4PatchEmbedder<br/>- 16x16 Pixel Slicing<br/>- Linear Projection (768-dim)<br/>- 2D Row & Col Positional Table"]
        ViT_Layers["16-Layer Vision Transformer (ViT)<br/>- 12 Attention Heads (head_dim = 64)<br/>- QK RMSNorm + RoPE (theta = 100)<br/>- GeGLU Gated MLP"]
        VisionProj["Multimodal Projector (embed_vision)<br/>Linear: 768 -> 1536"]
        
        SingleTensor & VideoTensor --> PatchEmbed --> ViT_Layers --> VisionProj
    end

    subgraph TemporalFlattening ["Temporal Video Token Flattening"]
        Flatten["Multi-frame Reshape:<br/>(N_frames, 256_patches, 1536) -> (1, N_frames * 256, 1536)"]
        VisionProj --> Flatten
    end

    subgraph TextPipeline ["Text Processing Pipeline"]
        UserPrompt["User Prompt: '<|turn>user\n<|image|>\nDescribe this scene...<turn|>'"]
        Tokenizer["Gemma Tokenizer<br/>(Token IDs)"]
        TextEmbed["Language Model Embedding<br/>(embed_tokens * sqrt(1536))"]

        UserPrompt --> Tokenizer --> TextEmbed
    end

    subgraph Fusion ["Multimodal Fusion Stage"]
        Splice["fuse_multimodal_embeddings()<br/>- Replaces '<|image|>' with Projected Vision Patches<br/>- Preserves Conversation Formatting"]
        Flatten --> Splice
        TextEmbed --> Splice
    end

    subgraph LLM_Backbone ["Gemma 4 Language Model Backbone"]
        Backbone["35 Decoder Layers<br/>- Per-Layer Embeddings (PLE)<br/>- KV Cache Sharing (Top 20 Layers)<br/>- Heterogeneous Attention"]
        OutputHead["Tied LM Head + Logit Softcapping"]
        
        Splice --> Backbone --> OutputHead
    end

    OutputHead --> StreamOut["Token-by-Token Generated Text Stream"]
```

---

## 1. Vision Patch Embedder & 2D Spatial Positional Table
* **Code Reference:** [`src/vision.rs:108-169`](../src/vision.rs#L108-L169)
* **Mathematical Operation:**
  1. An image or video frame tensor of shape $(B, 3, H, W)$ is sliced into non-overlapping patches of size $p = 16$.
  2. Each patch of raw pixels ($3 \times 16 \times 16 = 768$ values) is linearly projected onto the 768-dimensional vision embedding space.
  3. A 2D positional embedding vector is derived by indexing into `position_embedding_table` for both row index $y$ and column index $x$:
     $$\text{PosEmbed}(y, x) = \text{Table}[0, y] + \text{Table}[1, x]$$
  4. The 2D positional vector is broadcast-added to the projected patch tokens across all batch/frame dimensions.

```mermaid
flowchart LR
    subgraph Patching ["Image / Frame Slicing"]
        Img["256x256 Image / Frame"] --> Grid["16x16 Grid of Patches<br/>(256 Patches Total)"]
        Grid --> Flatten["Flatten Patches:<br/>(256, 768)"]
    end

    subgraph Embedding ["Projection and 2D Coordinates"]
        Flatten --> Proj["input_proj<br/>(768 -> 768)"]
        RowCol["2D Table Lookup<br/>(Row_Pos + Col_Pos)"]
        Proj & RowCol --> Add["(+) Positional Addition"]
    end

    Add --> PatchesOut["Embedded Patch Sequence (256, 768)"]
```

---

## 2. 16-Layer Vision Transformer (ViT) Encoder
* **Code Reference:** [`src/vision.rs:241-356`](../src/vision.rs#L241-L356)
* **Architectural Features:**
  - **16 Sequential Transformer Layers:** Each layer features pre-layernorms, post-attention norms, and post-feedforward norms using RMSNorm.
  - **12 Attention Heads:** Head dimension $d_{\text{head}} = 64$ ($12 \times 64 = 768$).
  - **Vision RoPE:** Rotary position embeddings initialized with a base frequency $\theta = 100.0$.
  - **Gated GeGLU MLP:** Intermediate size $3072$ ($4 \times 768$).

---

## 3. Video Frame Temporal Processing
* **Code Reference:** [`src/bin/multimodal_cli.rs:52-88`](../src/bin/multimodal_cli.rs#L52-L88), [`src/multimodal.rs:88-103`](../src/multimodal.rs#L88-L103)
* **Pipeline:**
  1. The CLI accepts video containers (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`).
  2. Extracts $N = 8$ uniformly sampled keyframes at $256 \times 256$ resolution.
  3. Encodes all $N$ frames through the ViT encoder $(8, 256, 768)$ and multimodal projection head $(8, 256, 1536)$.
  4. Flattens temporal frames into a continuous visual token stream $(1, 2048, 1536)$ spliced into the `<|image|>` placeholder.

---

## 4. Multimodal Projector (`embed_vision`)
* **Code Reference:** [`src/vision.rs:374-403`](../src/vision.rs#L374-L403)
* **Function:** Linear projection layer mapping the 768-dimensional vision representation into the language model's 1536-dimensional hidden dimension.
  $$h_{\text{multimodal}} = W_{\text{proj}} \cdot h_{\text{vision}}$$

---

## 5. Audio Tower Subsampling Convolutions (`model.audio_tower`)
* **Model Configuration:** [`models/gemma-4-e2b/config.json:5-44`](../models/gemma-4-e2b/config.json#L5-L44)
* **Pipeline:**
  1. Audio spectrogram inputs pass through 2 stages of subsampling convolutions (`128` channels $\to$ `32` channels).
  2. 12-layer audio transformer encoder ($d_{\text{model}} = 1024$).
  3. `embed_audio` linear projection head maps audio tokens into the 1536-dimensional language model embedding space.

```mermaid
flowchart LR
    AudioSpec["Raw Audio Spectrogram<br/>(128 Mel Bins)"] --> Conv0["Subsample Conv Layer 0<br/>(1 -> 128 Channels)"]
    Conv0 --> Conv1["Subsample Conv Layer 1<br/>(128 -> 32 Channels)"]
    Conv1 --> AudioLin["Linear Projection (1024)"]
    AudioLin --> AudioEnc["12-Layer Audio Encoder"]
    AudioEnc --> AudioProj["embed_audio Projector<br/>(1536 -> 1536)"]
    AudioProj --> FusedPrompt["Fused Multimodal Embedding Stream"]
```

---

## 6. How to Run Image & Video Inference

### Running with an Image:
```bash
# On Apple Silicon (Metal FP16):
cargo run --bin multimodal_cli --release -- photo.jpg "Describe the objects in this photo."

# On NVIDIA CUDA:
cargo run --bin multimodal_cli --release --no-default-features --features cuda -- photo.png "Explain the visual scene."
```

### Running with a Video:
```bash
# On Apple Silicon (Metal FP16):
cargo run --bin multimodal_cli --release -- clip.mp4 "Summarize the key events in this video."

# On NVIDIA CUDA:
cargo run --bin multimodal_cli --release --no-default-features --features cuda -- action.mov "What happens throughout this clip?"
```
