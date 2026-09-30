# Builder's Log: Custom Gemma 4 Inference Engine in Rust & Candle

> **Project Name:** Oxidizing Gemma — Rust, Candle, and Native Inference  
> **Target Model:** `google/gemma-4-E2B-it` (Google Gemma 4 Multimodal / Edge Architecture)  
> **Framework:** Rust + [Candle](https://github.com/huggingface/candle) (Zero-C / Minimal Runtime) with Apple Metal & NVIDIA CUDA Acceleration  
> **Goal:** Build an offline, native, zero-dependency inference engine from scratch in Rust to demonstrate modern systems programming and architecture verification in the age of AI code generation.

---

## 1. Why Do We Even Have a `gemma4.rs` File?

A common question when building with modern machine learning frameworks is:  
**"Why didn't we just call `candle_transformers::models::gemma::Model` or `from_pretrained()`?"**

### The Answer: Gemma 4 is a Fundamentally New Architecture
Upstream libraries (including Candle `0.8.x`, Ollama, and standard PyTorch Transformers distributions prior to late 2026) only implement **Gemma 1**, **Gemma 2**, or **Gemma 3**. 

Google's **Gemma 4** (`gemma-4-E2B`) is not an incremental parameter bump; it introduces completely novel architectural mechanisms that do not exist in any prior model:

1. **Per-Layer Embeddings (PLE):** A dual-stream embedding injection system (`embed_tokens_per_layer` + `per_layer_model_projection`) that re-conditions every single decoder layer dynamically.
2. **Upper-Layer KV-Cache Sharing (`num_kv_shared_layers: 20`):** The top 20 layers (15 through 34) do *not* have Key or Value projection weights; they share KV caches across sliding/global attention boundaries.
3. **Heterogeneous Attention with Proportional RoPE:** Interleaving 256-dimension sliding attention heads with 512-dimension global attention heads that rotate only 25% of their channels ($128$ rotated, $384$ pass-through).
4. **Quad-RMSNorm Blocks & Unit Value Normalization:** 4 RMSNorms per decoder layer, an unscaled unit RMSNorm on Value vectors, and layer-specific scalar multipliers.

Because **zero standard libraries supported this topology**, we had to author [`src/gemma4.rs`](src/gemma4.rs) from first principles in pure Rust.

---

## 2. The Battle Log: Every Issue Encountered & How We Solved It

Building a custom foundation model inference engine from raw Safetensors tensors was a journey of debugging numerical precision, hardware memory bandwidth, and architectural reverse-engineering. Here is the chronicle of every issue we hit:

---

### Issue 1: Upstream Hugging Face HTTP 401 & Gated Model Access
* **Symptom:** Trying to load `google/gemma-4-E2B-it` dynamically via the Hugging Face API threw `401 Unauthorized` and failed URL resolution.
* **Root Cause:** Gemma 4 is a gated model requiring a signed user license agreement on Hugging Face. Furthermore, relying on remote HTTP downloads during a live talk creates a single point of failure on conference Wi-Fi.
* **Fix:** Downloaded the full `config.json`, `tokenizer.json`, and `model.safetensors` offline directly into `models/gemma-4-e2b/`. The engine now runs 100% locally with zero internet access required.

---

### Issue 2: Rotary Embedding Startup Latency (30s+ initialization)
* **Symptom:** Initializing the model took over 30 seconds before printing a single prompt.
* **Root Cause:** In our initial draft, every one of the 35 layers instantiated its own independent `RotaryEmbedding` struct, creating and dispatching over 70 sequential GPU buffer allocations on the command queue during startup.
* **Fix:** Lifted RoPE frequency precomputation out of the loop. We precompute two shared frequency tables (`rope_sliding` for 256-dim and `rope_global` for 512-dim) and pass lightweight clones into each layer. Startup time dropped to **0.78s on H100** and **1.5s on Metal**.

---

### Issue 3: Metal Batched Matmul Dimension Mismatch
* **Symptom:** Runtime panic during the output projection: `Dimension mismatch in matmul: [1, 1, 1536] x [1536, 262144]`.
* **Root Cause:** The final hidden state had shape `[batch=1, seq_len=1, hidden=1536]`. Candle's matrix multiplication against the 2D tied word embedding matrix `[1536, 262144]` expected a 2D matrix.
* **Fix:** Squeezed the sequence dimension before the vocabulary projection: `last_hidden.squeeze(1)?.matmul(&lm_head_weight)?`.

---

### Issue 4: Strided View Panics in Apple Metal (`[1, 0, 24, 1]`)
* **Symptom:** Candle crashed with `Metal error: buffer is not contiguous with stride 0`.
* **Root Cause:** Gemma 4 uses Multi-Query Attention (`num_key_value_heads = 1`). Broadcasting this 1 head to 8 Query heads via `.broadcast_as()` creates zero-stride memory views. Apple Metal's hardware matrix-multiplication kernels require physically contiguous buffers.
* **Fix:** Inserted explicit `.contiguous()?` calls immediately following `.transpose()` and `.broadcast_as()` across all Query, Key, Value, and Attention tensor operations.

---

### Issue 5: FP16 Causal Mask NaN / Panic (`A weight is negative, too large...`)
* **Symptom:** Generation failed with: `Response: Error: A weight is negative, too large or not a valid number`.
* **Root Cause:** During the causal attention masking step in FP16, we used `f32::NEG_INFINITY`. In 16-bit half precision, $-\infty$ subtracted or added to FP16 logits causes numerical underflow resulting in `NaN` (Not a Number). When `LogitsProcessor::sample()` encountered `NaN` probability distributions, it panicked.
* **Fix:** Replaced `f32::NEG_INFINITY` with `-1e4f32` (`-10000.0`). Additionally, logit softcapping and final softmax are cast to `F32` before sampling to preserve numerical stability.

---

### Issue 6: The RMSNorm Offset Trap (`+ 1.0` vs Direct Multiplier)
* **Symptom:** The model generated repetitive numerical gibberish.
* **Root Cause:** Many older Gemma implementations added `1.0` to the scale weights: $x \times (1.0 + w)$. In Gemma 4's official checkpoints, the weights in `model.safetensors` are already absolute scale factors (mean $\approx 10.6$). Adding `1.0` shifted every layer's activation scale drastically out of distribution.
* **Fix:** Updated `RmsNorm` to apply direct weight multiplication: `x_normed.broadcast_mul(&self.weight)`.

---

### Issue 7: Prompt Format & Control Delimiters
* **Symptom:** Model responded with endless question echoes or hallucinated prompt continuations.
* **Root Cause:** Gemma 4 instruction models do not use standard `<bos>` or `<|im_start|>`. They expect exact Google Gemma 4 turn tokens:
  ```
  <|turn>user\n{prompt}<turn|>\n<|turn>model\n
  ```
* **Fix:** Updated the prompt wrapper in `main.rs` and configured the End-Of-Sequence (EOS) detector to stop on `<turn|>`.

---

### Issue 8: Multilingual Gibberish (The Gemma 4 PLE Discovery)
* **Symptom:** The model generated tokens at 43 tokens/sec on H100, but outputted scrambled multilingual phrases:
  ```
  Response: 一错误 的จ Apakah預空 gleichenяр préal सावधान Sử준 Whateverimat매...
  ```
* **Root Cause:** When the AI assistant initially generated the decoder layer, it assumed a vanilla Transformer without **Per-Layer Embeddings (PLE)** and without **Upper-Layer KV Sharing**. Because layers 15..34 received unconditioned activations and un-shared KV states, the higher-level representations degraded into random noise.
* **Fix:** Reverse-engineered the complete PLE pipeline from Hugging Face's modeling implementation:
  1. Implemented dual-component PLE: `embed_tokens_per_layer` + `per_layer_model_projection`.
  2. Added the per-layer gated residual injection: `x = x + Norm(Proj(GELU(Gate(x)) * PLE_l))`.
  3. Linked KV cache sharing across layers 15..34.
  *Result:* Immediate, crystal-clear, grammatically perfect English generation!

---

### Issue 9: Local Memory Bandwidth Limits vs. Datacenter HBM3
* **Symptom:** On the Apple Silicon M3 Max, generation throughput was ~4 tokens/sec due to memory bus competition while running local development workloads.
* **Root Cause:** Decoding autoregressively requires streaming the full 5GB model weights through memory for *every single token*. On unified DDR5 memory (300-400 GB/s), this hits a physical bandwidth ceiling.
* **Fix:** Provisioned a bare-metal **NVIDIA H100 80GB SXM5** instance on **Nebius Cloud**. Enabled Candle's CUDA backend with `cargo run --release --no-default-features --features cuda`. Generation throughput rocketed to **41.06 - 43.50+ tokens/sec** thanks to **3.35 TB/s HBM3 memory bandwidth**.

---

## 3. Architecture Comparison: Gemma 4 vs. Prior Generations

| Architectural Component | Gemma 1 / 2 | Gemma 3 | **Gemma 4 (This Implementation)** |
| :--- | :--- | :--- | :--- |
| **Token Embeddings** | Static input at Layer 0 only | Static input at Layer 0 | **Dynamic Dual-Component PLE at every layer** |
| **KV-Cache Sharing** | None (Every layer has own KV) | None | **Shared across upper 20 layers (15..34)** |
| **Attention Head Dims** | Uniform (e.g. 256 for all) | Uniform | **Heterogeneous: 256 (Sliding) vs 512 (Global)** |
| **RoPE Encoding** | Uniform across all dimensions | Uniform | **Proportional RoPE: 0.25 on 512-dim heads** |
| **RMSNorm per Block** | 2 norms (Pre-Attn, Pre-FFW) | 2 norms | **4 norms + Unit RMSNorm on Values + Scalars** |
| **Feed-Forward MLP** | Fixed intermediate dimension | Fixed | **Double-Wide MLP on upper layers (12,288)** |

---

## 4. Final Benchmark Summary

```
=== Custom Gemma 4 E2B Rust + Candle Inference Engine ===
Active hardware device: Cuda(CudaDevice(DeviceId(0))) (Precision: F16)
Memory-mapping 1 safetensors file(s)
Model initialized in 0.78s

Prompt:
<|turn>user
Explain why Rust is great for AI and systems in one sentence.<turn|>
<|turn>model

Response: Rust's combination of high performance, memory safety, and concurrency makes it an ideal language for building efficient and reliable AI systems and complex software infrastructure.

--- Benchmark ---
Generated 30 tokens in 811.48ms (36.97 tokens/sec)
```
