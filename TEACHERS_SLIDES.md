# Oxidizing Gemma: Teacher's Edition & Lecture Notes

**Subtitle:** *Building Zero-Python Foundation Model Systems in Rust and Go*  
**Audience:** Systems Engineers, ML Infrastructure Builders, and Advanced Developers  
**Author / Instructor:** Jorge Jimenez  

---

## Slide 1: Title & The Zero-Python Thesis

### Slide Content:
* **Oxidizing Gemma:** *Rust, Candle, and Native Foundation Model Inference*
* **Core Philosophy:** Moving foundational AI runtimes from dynamic interpreted Python to compiled, memory-safe, zero-GC systems languages.

### Teacher's Guide & Speaker Notes:
* **The Hook:** Open by asking: *"How many gigabytes is your production PyTorch Docker image?"* When people answer 15GB to 25GB, explain that today we will build a production Gemma 4 runtime in a **35MB standalone binary** with zero Python dependencies.
* **Socratic Concept Check:**
  * *Question:* Why does ML research love Python while ML deployment struggles with it?
  * *Answer:* Python is optimized for human developer iteration speed in notebooks. Serving is optimized for machine memory bandwidth, thread scheduling, and predictable latency.

---

## Slide 2: Why Learn a Programming Language in 2026?

### Slide Content:
* **The Epistemic Shift:** Moving from Writing Syntax to Architecting Reality.
* When AI generates syntax on demand, why invest hundreds of hours mastering borrow checkers, memory layouts, cache lines, and concurrency primitives in 2026?
* **Syntax is Commoditized:** Generating text tokens has reached near-zero marginal cost.
* **Constraints are Inviolable:** Hardware, memory bandwidth, cache hierarchies, and synchronization laws do not bend for natural language.
* **The New Reality:** You do not learn a language to type characters. You learn it so you have the *conceptual vocabulary* to evaluate, constrain, and verify what the machine builds.

### Teacher's Guide & Speaker Notes:
* **Pedagogical Narrative:** Address the existential elephant in the room: Why bother learning low-level programming in 2026 when LLMs can write code? We are the first generation of engineers where the physical act of typing syntax is no longer the bottleneck. But syntax was never the true substance of engineering—understanding constraints, memory boundaries, and physical reality was.
* **Concept Check for Students:**
  * *Question:* What is the danger of relying purely on natural language prompting without systems foundations?
  * *Answer:* You cannot diagnose why a system is slow, why tail latencies spike, or why memory leaks occur—because conversational English lacks the vocabulary of physical hardware constraints.

---

## Slide 3: What Does It Mean to LEARN a Language Nowadays?

### Slide Content:
* **Internalizing Computational Worldviews Rather Than Memorizing APIs.**
* **The Obsolete Paradigm:** Memorizing function signatures, syntax rules, and writing repetitive CRUD boilerplate by hand.
* **The Modern Paradigm:**
  * *Internalizing Worldviews:* Rust teaches affine ownership and memory safety; Go teaches CSP concurrency and goroutines.
  * *Language as Thought Scaffold:* You can only prompt for architectures you have mental models for.
  * *Deterministic Verification:* The compiler is your mathematical theorem prover.

### Teacher's Guide & Speaker Notes:
* **Pedagogical Narrative:** Emphasize Wittgenstein's insight: *"The limits of my language mean the limits of my world."* If your only mental model is Python, you think in terms of objects, dynamic dictionaries, and garbage collectors. When you learn Rust, your brain starts thinking in terms of cache lines, stack vs heap, pointer aliasing, and thread-safety invariants.
* **Concept Check for Students:**
  * *Question:* How does learning Rust change how you prompt an AI coding assistant?
  * *Answer:* Instead of vague prompts like 'make this faster and safe', you prompt with exact mechanical constraints: 'Borrow the KV cache immutably across upper layers without heap allocation, ensuring Send + Sync across Tokio threads.'

---

## Slide 4: The Prompting Paradox: Domain Math vs. Natural Language

### Slide Content:
* *"At what point does describing a domain process in English become 100x more verbose, clumsy, and exhausting than writing the mathematical notation?"*
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

### Teacher's Guide & Speaker Notes:
* **Pedagogical Narrative:** Remind the room: Why did physicists and mathematicians stop writing paragraphs in Latin and invent algebra and calculus? Because a single equation captures relationships that would take 5 pages of prose. Code is applied domain math. Trying to replace domain code with conversational prompts is literally trying to replace algebra with word problems.
* **Concept Check for Students:**
  * *Question:* Why is natural language inefficient for specifying tensor operations?
  * *Answer:* Natural language lacks native syntax for dimensionality, transpositions, broadcast rules, and matrix multiplication. A 10-character math expression contains more structured information than a paragraph of text.

---

## Slide 5: Safetensors & Zero-Copy POSIX `mmap`

### Slide Content:
* **Legacy PyTorch (.bin):** Python `pickle` executes arbitrary code, deserializing into RAM before copying to GPU.
* **Safetensors (.safetensors):** 8-byte aligned raw binary tensors with a pure JSON header.
* **POSIX `mmap`:** Direct virtual address mapping from disk pages into GPU address space.

### Teacher's Guide & Speaker Notes:
* **How to Teach This:** Draw memory on a whiteboard. Show how `mmap` assigns virtual memory addresses to disk offsets. The OS page cache pages in weights on-demand with zero serialization or host memory duplication.
* **Code Highlight:** Show `unsafe { VarBuilder::from_mmaped_safetensors(...) }` and explain why Rust requires the `unsafe` keyword (external process file modification invariant).

---

## Slide 6: Gemma 4 Neural Decoder & RMSNorm (+1 Offset)

### Slide Content:
$$\text{Gemma4RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot (\gamma + 1.0)$$
* Scaling by $\sqrt{d_{\text{model}}} = \sqrt{1536} \approx 39.1918$.
* Pre-attention, post-attention, pre-feedforward, and post-feedforward normalization boundaries.

### Teacher's Guide & Speaker Notes:
* **Deep-Dive Question:** *"Why does Gemma initialize RMSNorm weights to 0.0 with a +1.0 unit offset instead of initializing them to 1.0?"*
* **Answer:** With residual unit scaling $\gamma + 1.0$, weight decay during training regularizes $\gamma \to 0$, which acts as an identity function ($x \cdot 1.0$) rather than destroying activation scale!

---

## Slide 7: Per-Layer Embeddings (PLE)

### Slide Content:
* Direct linear projection from initial token embeddings into every decoder layer.
* Injects lexical features directly into deep layers, providing a direct gradient highway.

### Teacher's Guide & Speaker Notes:
* **Analogy:** Compare standard Transformers to a game of "telephone" across 35 layers. By Layer 30, the subtle original prompt tokens might be blurred. PLE acts as a direct intercom from Layer 0 to every intermediate layer.

---

## Slide 8: Upper-Layer KV Cache Sharing & Memory Math

### Slide Content:
* Layers 0–14 compute independent KV caches.
* Layers 15–34 share Layer 14's KV cache.
* **Memory Reduction:** $57.1\%$ savings in KV Cache VRAM at 8k context lengths.

### Teacher's Guide & Speaker Notes:
* **Interactive Math Exercise:** Have students calculate the KV Cache size for 35 layers vs 15 layers at 32k context on a whiteboard.
* **Systems Insight:** Autoregressive LLMs are memory-bandwidth bound. Cutting memory reads in half nearly doubles decoding throughput on bandwidth-constrained GPUs.

---

## Slide 9: The 3-Tier Network Architecture (Rust + Go + gRPC)

### Slide Content:
* **Rust:** Raw compute, GEMM tensor operations, Metal/CUDA kernels.
* **gRPC / Protobuf:** Strongly-typed streaming contract on port 50051.
* **Go Gateway:** High-concurrency reverse proxy, SSE streaming on port 8080.

### Teacher's Guide & Speaker Notes:
* **Why not write the web server in Rust?** While Rust web frameworks (Axum, Actix) are fast, separating the network routing layer in Go allows zero-downtime gateway restarts, independent autoscaling, and lightweight goroutine connection pooling.

---

## Slide 10: High-Concurrency Go API Gateway

### Slide Content:
* Exposes OpenAI-compatible `POST /v1/chat/completions` (Server-Sent Events).
* Goroutine multiplexing: Handles 10,000+ open HTTP SSE client streams.
* Zero GPU footprint: Consumes $< 20\text{MB}$ of RAM.

---

## Slide 11: Multimodal Vision & Video Engine

### Slide Content:
* 16-layer Vision Transformer (ViT) with 2D spatial positional table `[2, 10240, 768]`.
* Linear projector `embed_vision` ($768 \to 1536$).
* Temporal multi-frame video extraction using FFmpeg and batch flattening.

---

## Slide 12: Multi-Cloud Deployment & SkyPilot

### Slide Content:
* Automated deployment across Google Cloud (L4/A100) and Nebius Cloud (H100).
* Single-command provisioning: `sky launch -c gemma4-gcp sky-gcp.yaml --yes`.
