---
marp: true
theme: default
paginate: true
header: "Oxidizing Gemma — Rust, Candle, and Native Inference"
footer: "Jorge Jimenez | 2026"
style: |
  section {
    font-family: 'Google Sans', 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #202124;
    color: #e8eaed;
  }
  h1, h2, h3 {
    color: #4285F4;
  }
  code {
    background-color: #1f2023;
    color: #FBBC05;
  }
  blockquote {
    background: #303134;
    border-left: 4px solid #4285F4;
    padding: 10px 20px;
    color: #9aa0a6;
  }
  table {
    font-size: 0.85em;
  }
---

# Oxidizing Gemma
### Rust, Candle, and Native LLM Inference

**Speaker:** Jorge Jimenez  
*Building zero-Python foundation model runtimes from first principles in the age of Agentic AI.*

---

# Why Learn a Programming Language in 2026?
### The Epistemic Shift: Moving from Writing Syntax to Architecting Reality

When an AI agent can synthesize boilerplates, functions, and regex in milliseconds:
> *"Why invest hundreds of hours mastering borrow checkers, memory layouts, cache lines, and concurrency primitives?"*

* **Syntax is Commoditized:** Generating text has reached near-zero marginal cost.
* **Constraints are Inviolable:** Hardware, memory bandwidth, cache hierarchies, and synchronization laws do not bend for natural language.
* **The New Reality:** You do not learn a language to type it. You learn a language so you can **evaluate, constrain, and verify** what the machine builds.

---

# What Does It Mean to LEARN a Language Nowadays?
### Internalizing Computational Worldviews Rather Than Memorizing APIs

| The Obsolete Paradigm | The Modern Paradigm |
| :--- | :--- |
| Memorizing API signatures & syntax | **Internalizing Worldviews:** Rust (ownership/affine types), Go (CSP concurrency) |
| Writing repetitive CRUD boilerplate | **Language as Thought Scaffold:** Prompting only for what you can conceptualize |
| Mechanical transcription | **Deterministic Verification:** The compiler as your mathematical theorem prover |

> *"The limits of my language mean the limits of my world."* — Ludwig Wittgenstein

---

# The Prompting Paradox: Domain Math vs. Natural Language

### Why Not Just Write in English?
> *"At what point does describing a domain process in English become 100x more verbose, clumsy, and exhausting than writing the mathematical notation?"*

* **Verbose English Prompt (110+ words):**
  > *"Take the query vectors, scale them by the inverse square root of head dimension, compute pairwise dot-products against all key vectors across the sequence length, apply numerical stabilization by subtracting row maximums, compute softmax across the attention window, and multiply by the value representations..."*
* **Baked-In Domain Knowledge & Code:**
  $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)V$$
  ```rust
  let scores = (q.matmul(&k.t())? * (1.0 / sqrt_d))?;
  let weights = candle_nn::ops::softmax(&scores, 1)?;
  let context = weights.matmul(&v)?;
  ```
* **Domain Axiom:** Mathematical equations and typed code are *compressed thought*. Natural language is fundamentally too low-density for precision engineering.

---

# Why Rust for AI Systems?

### The "Python Tax" in Production AI
* Heavyweight runtimes (Docker images ballooning to 15GB+).
* Garbage Collection latency spikes causing unpredictable tail latencies.
* The GIL bottleneck requiring complex IPC for multiprocessing.

### The Rust Advantage
* **Zero Runtime Overhead:** Direct compilation to machine code.
* **Predictable Tail Latency:** No GC pauses during real-time token streaming.
* **Single Binary Deployment:** 35MB standalone executable runs anywhere.
* **Fearless Concurrency:** Thread safety enforced at compile time.

---

# What is Candle?

### Minimalist Machine Learning for Rust
Created by Hugging Face, **Candle** is a lightweight tensor framework focused on performance and deployment simplicity.

| Feature | PyTorch | Candle (Rust) |
| :--- | :--- | :--- |
| **Binary Footprint** | ~5 GB+ (CUDA + Torch Wheels) | **~35 MB** (Single Executable) |
| **Python Dependency** | Mandatory | **Zero (100% Native)** |
| **Memory Allocation** | Dynamic Heap Allocations | **Zero-Copy / In-Place** |
| **Tail Latency** | Non-Deterministic (GC Pauses) | **Deterministic ($p99 < 15\text{ms}$)** |

---

# Safetensors and POSIX `mmap`

### Eliminating Weight Deserialization Overhead
Standard PyTorch pickles (`.bin` files) force the CPU to deserialize Python objects into host RAM before copying to GPU.

* **Safetensors:** 8-byte aligned raw binary tensors with a lightweight JSON header.
* **POSIX `mmap`:** Direct virtual address mapping from disk pages into GPU address space.

```rust
// Rust Zero-Copy mmap:
let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, &device)?
};
```
* **Startup Time:** Reduced from 25+ seconds to **< 1.5 seconds**.

---

# Gemma 4 Architecture: RMSNorm with Unit Offset

$$\text{Gemma4RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot (\gamma + 1.0)$$

* **Input Embedding Multiplier:** Multiplies initial embeddings by $\sqrt{d_{\text{model}}} = \sqrt{1536} \approx 39.1918$.
* **Unit Scaling:** $\gamma$ initializes to 0.0 with $+1.0$ offset, allowing weight decay to regularize toward an identity transform.

---

# Per-Layer Embeddings (PLE)

### Direct Lexical Conditioning into Deep Layers
* Standard Transformers only inject embeddings at Layer 0.
* In Gemma 4, each layer projects initial token embeddings directly into the hidden states before the feedforward network:

```rust
if let Some(ref ple) = self.per_layer_projection {
    let ple_features = ple.forward(initial_embeddings)?;
    hidden_states = (hidden_states + ple_features)?;
}
```

---

# Upper-Layer KV Cache Sharing

### Autoregressive Decoding is Memory-Bandwidth Bound
* **Lower Layers (0..14):** Compute and cache independent Keys and Values.
* **Upper Layers (15..34):** Reuse Layer 14's Key/Value tensors via zero-copy borrowing!

* **Memory Math @ 8k Context:**
  * Standard (35 layers): $1.18\text{ GB / stream}$
  * Gemma 4 (15 layers): $0.51\text{ GB / stream}$ (**57.1% Reduction in VRAM traffic**)

---

# High-Concurrency Go API Gateway

* **Lightweight Goroutines:** Schedules 10,000+ simultaneous client streams on 2KB green threads.
* **Zero GPU VRAM Footprint:** The Go process consumes $< 20\text{MB}$ of RAM.
* **OpenAI-Compatible SSE:** Exposes `POST /v1/chat/completions` directly to web and mobile clients.

---

# Multimodal Vision & Video Engine

* **16-Layer Vision Transformer:** Processes $256 \times 256$ frames into 256 visual tokens.
* **2D Positional Tables:** Heterogeneous row/col embedding table `[2, 10240, 768]`.
* **Temporal Video Concatenation:** Samples 8 frames via FFmpeg and flattens into `(1, 2048, 1536)` token sequences.

---

# Hardware Benchmarks

| Hardware | Accelerator | Precision | TTFT | Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **MacBook Pro M3 Max** | Metal (16-Core GPU) | FP16 | **48 ms** | **45.2 tok/sec** |
| **Google Cloud (L4)** | NVIDIA L4 (24GB) | FP16 | **32 ms** | **68.4 tok/sec** |
| **Nebius Cloud (H100)** | NVIDIA H100 SXM5 (80GB) | FP16 | **14 ms** | **142.8 tok/sec** |

---

# The Systems Builder's Blueprint

> *"In the era of Agentic AI, high-level code is generated, but low-level constraints remain inviolable."*

* Master the memory hierarchy.
* Build on zero-copy data planes.
* Let the compiler verify your invariants.
