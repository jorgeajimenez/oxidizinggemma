# Deep Learning & Neural Architecture Internals

This document breaks down the top 10 neural architectural mechanics implemented in pure Rust (`src/gemma4.rs`) for Google's Gemma 4 foundation model.

---

## 1. End-to-End Gemma 4 Model Architecture

```mermaid
flowchart TD
    InputIDs["Token IDs (Batch, SeqLen)"] --> TextEmbed["embed_tokens * sqrt(1536)"]
    InputIDs --> PLE_Engine["Per-Layer Embedding (PLE) Engine"]
    
    TextEmbed --> Layer0["Layer 0 (Sliding Attention)"]
    PLE_Engine -.->|"PLE Injection Layer 0"| Layer0
    
    Layer0 --> Layer1["Layer 1..34 (Heterogeneous Attention)"]
    PLE_Engine -.->|"PLE Injection Layers 1..34"| Layer1
    
    Layer1 --> FinalNorm["Final RmsNorm (hidden_size: 1536)"]
    FinalNorm --> TiedHead["Tied LM Head Projection (embed_tokens.t())"]
    TiedHead --> SoftCap["Logit Softcapping (tanh * 30.0)"]
    SoftCap --> OutputLogits["Output Logits (Batch, VocabSize: 262,144)"]
```

---

## 2. Per-Layer Embeddings (PLE) Residual Injection
* **File Reference:** [`src/gemma4.rs:583-604`](../src/gemma4.rs#L583-L604), [`src/gemma4.rs:377-384`](../src/gemma4.rs#L377-L384)
* **Mathematical Formulation:**
  Gemma 4 computes a layer-specific identity and context signal:
  $$\text{PLE}_l = \text{Norm}\left(\text{Proj}(x) \cdot \frac{1}{\sqrt{d_{\text{model}}}} + \text{Embed}_{\text{PLE}}(w) \cdot \sqrt{d_{\text{ple}}}\right) \cdot \frac{1}{\sqrt{2}}$$

```mermaid
flowchart LR
    TokenIDs["Token IDs (w)"] --> PLE_Embed["embed_tokens_per_layer * sqrt(256)"]
    HiddenState["Input Embeddings (x)"] --> PLE_Proj["per_layer_model_projection * (1 / sqrt(1536))"]
    
    PLE_Proj --> PLE_Norm["per_layer_projection_norm (RmsNorm)"]
    PLE_Embed & PLE_Norm --> AddComponents["(+) Combine Components"]
    AddComponents --> ScaleComponents["Scale by 1/sqrt(2)"]
    ScaleComponents --> PLE_Out["Layer PLE Signal (Batch, SeqLen, 35, 256)"]
```

---

## 3. Upper-Layer KV-Cache Sharing (Layers 15..34)
* **File Reference:** [`src/gemma4.rs:430-438`](../src/gemma4.rs#L430-L438), [`src/gemma4.rs:616-631`](../src/gemma4.rs#L616-L631)
* **Mechanics:**
  In standard transformers, every layer maintains its own Key and Value tensors. In Gemma 4:
  - **Layers 0..14:** Allocate and compute independent Key and Value projections.
  - **Layers 15..34:** Do NOT have `k_proj` or `v_proj` weights. They reuse Key and Value states produced by lower layers.

```mermaid
flowchart TD
    subgraph LowerLayers ["Lower Layers (0..14) - Full Key/Value Projections"]
        L0["Layer 0: Computes Q, K, V -> Updates Sliding KV Cache"]
        L4["Layer 4: Computes Q, K, V -> Updates Global KV Cache"]
        L14["Layer 14: Final Layer with Dedicated KV Projections"]
    end

    subgraph UpperLayers ["Upper Layers (15..34) - KV Cache Sharing"]
        L15["Layer 15: Q Projection Only -> Reuses Shared KV"]
        L20["Layer 20: Q Projection Only -> Reuses Shared KV"]
        L34["Layer 34: Q Projection Only -> Reuses Shared KV"]
    end

    L0 -.->|"Passes Sliding KV"| L15
    L4 -.->|"Passes Global KV"| L20
    L14 -.->|"Passes KV Cache"| L34
```

---

## 4. Heterogeneous Attention (Sliding Window vs Global Full Attention)
* **File Reference:** [`src/gemma4.rs:271-295`](../src/gemma4.rs#L271-L295)

```mermaid
flowchart LR
    subgraph SlidingWindowLayer ["Sliding Attention (30 of 35 Layers)"]
        SW_Q["Q Dim: 256"]
        SW_KV["KV Dim: 256"]
        SW_RoPE["Standard RoPE (theta: 10,000)"]
        SW_Mask["Sliding Window Mask (512 tokens)"]
        SW_Q & SW_KV --> SW_RoPE --> SW_Mask
    end

    subgraph GlobalFullLayer ["Global Full Attention (5 of 35 Layers: 4, 9, 14, 19, 34)"]
        GL_Q["Global Q Dim: 512"]
        GL_KV["Global KV Dim: 512"]
        GL_RoPE["Proportional RoPE (theta: 1,000,000)"]
        GL_Mask["Causal Full Attention Mask"]
        GL_Q & GL_KV --> GL_RoPE --> GL_Mask
    end
```

---

## 5. Proportional RoPE (Rotary Position Embeddings)
* **File Reference:** [`src/gemma4.rs:134-192`](../src/gemma4.rs#L134-L192)

```mermaid
flowchart TD
    InputQ["Query / Key Vector (head_dim: 512)"]
    
    InputQ --> SplitQ{"Split Dimensions"}
    SplitQ -->|"First 128 Dims (25%)"| RotatedPart["Apply 2D Rotary Matrix<br/>(theta: 1,000,000)"]
    SplitQ -->|"Remaining 384 Dims (75%)"| PassThrough["Pass-through Unmodified"]
    
    RotatedPart & PassThrough --> ConcatQ["Concatenate along Head Dim"]
    ConcatQ --> OutputRoPE["RoPE Augmented Vector (512)"]
```

---

## 6. Grouped-Query Attention (GQA) & KV Broadcasting
* **File Reference:** [`src/gemma4.rs:326-328`](../src/gemma4.rs#L326-L328)

```mermaid
flowchart TD
    KV_Head["1 Key/Value Head (num_kv_heads = 1)"]
    
    KV_Head -->|"repeat(num_heads / num_kv_heads)"| Broadcast["Broadcast 16x Across Query Heads"]
    
    Broadcast --> Q0["Query Head 0"]
    Broadcast --> Q1["Query Head 1"]
    Broadcast --> Q15["Query Head 15"]
```

---

## 7. QK RMS Normalization with Unit Scaling
* **File Reference:** [`src/gemma4.rs:260-264`](../src/gemma4.rs#L260-L264), [`src/gemma4.rs:306-308`](../src/gemma4.rs#L306-L308)

---

## 8. Gated GeGLU Feed-Forward Network (Double-Wide MLP)
* **File Reference:** [`src/gemma4.rs:223-239`](../src/gemma4.rs#L223-L239)

```mermaid
flowchart TD
    MLP_In["MLP Input (Batch, SeqLen, hidden_size: 1536)"]
    
    MLP_In --> GateProj["gate_proj Linear (1536 -> 6144)"]
    MLP_In --> UpProj["up_proj Linear (1536 -> 6144)"]
    
    GateProj --> GELU["GELU Activation (gelu_erf)"]
    GELU & UpProj --> ElementwiseMul["Elementwise Multiply ⊙"]
    
    ElementwiseMul --> DownProj["down_proj Linear (6144 -> 1536)"]
    DownProj --> MLP_Out["MLP Output (1536)"]
```

---

## 9. Full Decoder Layer Residual & Scalar Integration
* **File Reference:** [`src/gemma4.rs:356-428`](../src/gemma4.rs#L356-L428)

```mermaid
flowchart TD
    LayerIn["Layer Input (x)"]
    
    subgraph SelfAttentionBlock ["Self-Attention Block"]
        Norm1["input_layernorm"]
        Attn["Gemma4Attention (RoPE + GQA + QK Norm)"]
        PostAttnNorm["post_attention_layernorm"]
        PostAttnScalar["post_attention_scalar"]
        
        Norm1 --> Attn --> PostAttnNorm --> PostAttnScalar
    end
    
    LayerIn --> Norm1
    LayerIn --> Add1["(+) Residual Add"]
    PostAttnScalar --> Add1
    
    subgraph PLE_Block ["Per-Layer Embedding Residual Stage"]
        PLE_In["Layer PLE Signal (256)"]
        PLE_Gate["per_layer_input_gate (1536)"]
        PLE_Proj["per_layer_model_projection"]
        
        PLE_In & PLE_Gate --> PLE_Proj
    end
    
    Add1 --> Add2["(+) Residual Add PLE"]
    PLE_Proj --> Add2
    
    subgraph FFN_Block ["Feed-Forward Network (FFN) Block"]
        Norm2["pre_feedforward_layernorm"]
        MLP["Gemma4Mlp (GeGLU)"]
        PostFFNNorm["post_feedforward_layernorm"]
        PostFFNScalar["post_feedforward_scalar"]
        
        Norm2 --> MLP --> PostFFNNorm --> PostFFNScalar
    end
    
    Add2 --> Norm2
    Add2 --> Add3["(+) Residual Add FFN"]
    PostFFNScalar --> Add3
    
    Add3 --> LayerOut["Layer Output to Next Stage"]
```

---

## 10. Prefill vs. Autoregressive Decode Lifecycle
* **File Reference:** [`src/main.rs:163-198`](../src/main.rs#L163-L198)

```mermaid
sequenceDiagram
    autonumber
    actor Client as HTTP Client
    participant GW as Go API Gateway
    participant Worker as Rust Inference Worker
    participant Model as Gemma4ForCausalLM (GPU VRAM)

    Client->>GW: POST /v1/chat/completions (Prompt: N tokens)
    GW->>Worker: gRPC StreamGenerate(Prompt)
    
    Note over Worker,Model: Phase 1: Prefill (Prompt Processing)
    Worker->>Model: forward(prompt_tokens[0..N], pos=0)
    Model->>Model: Populate Initial KV Cache for N tokens
    Model-->>Worker: Output Logits for Token N+1
    Worker->>GW: Stream Token N+1
    GW-->>Client: SSE data: {"token": "..."}

    Note over Worker,Model: Phase 2: Autoregressive Decode
    loop Each Generated Token
        Worker->>Model: forward(next_token[1], pos=N+step)
        Model->>Model: Append 1 Entry to KV Cache
        Model-->>Worker: Output Logits for Token N+step+1
        Worker->>GW: Stream Token
        GW-->>Client: SSE data: {"token": "..."}
    end

    Worker->>Model: clear_kv_cache()
    Worker-->>GW: Stream Finished
    GW-->>Client: SSE data: [DONE]
```
