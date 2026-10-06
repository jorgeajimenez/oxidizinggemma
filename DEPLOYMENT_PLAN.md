# Multi-Cloud Gemma 4 Deployment Plan (Nebius & GCP)

> **Objective:** Deliver a resilient, zero-friction automated deployment of the custom Gemma 4 Rust + Go inference stack across **Nebius AI Cloud** and **Google Cloud Platform (GCP)** using **SkyPilot**, eliminating slow local file transfers and resolving CUDA toolkit version mismatches.

---

## 1. Problem Statement & Root Cause Fixes

```mermaid
flowchart LR
    subgraph Issues["Previous Bottlenecks"]
        I1["Slow Local Upload<br/>(9.8 GB rsync over Wi-Fi)"]
        I2["SSH Broken Pipe<br/>(Exit Code 255)"]
        I3["cudarc Build Panic<br/>(CUDA 13.0 unsupported)"]
    end

    subgraph Solutions["Engineered Resolutions"]
        S1["Cloud-Native HF Download<br/>(10 Gbps Datacenter ~15s)"]
        S2[".skyignore Local Weights<br/>(Sync code only < 2MB)"]
        S3["Automated nvcc Shim<br/>(Aliased 12.6 Compatibility)"]
    end

    I1 --> S1
    I2 --> S2
    I3 --> S3
```

| Issue Encountered | Root Cause | Permanent Resolution |
| :--- | :--- | :--- |
| **SkyPilot rsync 255 Error** | `sky-nebius.yaml` had `file_mounts: models/gemma-4-e2b`, forcing 9.8 GB of weights to upload over residential Wi-Fi (~3.9 MB/s = ~45 mins). | **Remove local file mount.** Download weights directly on the VM from Hugging Face via datacenter connection (10 Gbps = ~15 seconds). |
| **`cudarc 0.13.9` Build Panic** | Fresh cloud images (Ubuntu 24.04) come with CUDA 13.0, but `cudarc`'s `build.rs` only recognizes CUDA versions up to 12.8. | **Add an automated `nvcc` wrapper** in the `setup` phase that aliases the version check to 12.6 while using the real CUDA 13 compilers/drivers. |
| **GCP / Nebius Drift** | Different setup scripts and disparate environment configs between `sky-nebius.yaml` and `sky-gcp.yaml`. | **Standardize both YAML specifications** to share identical setup, environment variables, build steps, and endpoint contracts. |

---

## 2. Architecture & Service Topology

```mermaid
flowchart TD
    Client["Client / stream.sh / Web UI"]
    
    subgraph GatewayLayer["Ingress & Routing Layer"]
        Gateway["Go API Gateway (Port 8080)
        - HTTP/2 & Server-Sent Events (SSE)
        - CORS & Graceful Shutdown
        - Health Check Endpoint (/health)"]
    end
    
    subgraph InferenceLayer["High-Performance Compute Layer"]
        RustWorker["Rust Candle Engine (Port 50051)
        - gRPC Microservice (Tonic + Tokio)
        - Per-Layer Embeddings (PLE)
        - Upper 20-Layer KV Cache Sharing
        - Proportional RoPE (0.25 on 512-dim)
        - Quad-RMSNorm & Unit Value Scaling"]
    end

    subgraph StorageHardware["Hardware & Storage Subsystem"]
        Weights[("Local Weights: SafeTensors
        /root/models/gemma-4-e2b")]
        GPU["NVIDIA GPU Acceleration
        (H100 SXM5 / L4 / A100)"]
    end

    Client -->|"POST /v1/chat/completions (SSE)"| Gateway
    Gateway -->|"gRPC StreamGenerate()"| RustWorker
    RustWorker -->|"Memory-Mapped VarBuilder"| Weights
    RustWorker -->|"CUDA Kernels"| GPU
    RustWorker -.->|"Stream Generated Tokens"| Gateway
    Gateway -.->|"data: {'token': '...'} \n\n"| Client
```

---

## 3. Real-Time Token Generation & Streaming Lifecycle

```mermaid
sequenceDiagram
    autonumber
    actor User as Client (stream.sh / curl)
    participant Go as Go API Gateway (:8080)
    participant Rust as Rust gRPC Worker (:50051)
    participant Engine as Gemma4 LM Backbone (CUDA)

    User->>Go: POST /v1/chat/completions {prompt, max_tokens}
    Go->>Rust: gRPC StreamGenerate(GenerateRequest)
    Rust->>Engine: Tokenize & Prefill Prompt Tensor
    Engine->>Engine: Run 35 Layers + PLE + KV-Cache Sharing
    
    loop Autoregressive Decoding Loop
        Engine->>Engine: Forward Step (Token N)
        Engine-->>Rust: Logit Sampling -> Decoded Token
        Rust-->>Go: gRPC GenerateResponse {token, is_final: false}
        Go-->>User: SSE Chunk: data: {"token": "..."}\n\n
    end
    
    Rust-->>Go: gRPC GenerateResponse {token: "", is_final: true}
    Go-->>User: SSE End: data: [DONE]\n\n
```

---

## 4. Multi-Cloud Parity Matrix (Nebius vs. GCP)

```mermaid
flowchart LR
    subgraph UniformClient["Uniform Client API Interface"]
        C1["POST http://&lt;IP&gt;:8080/v1/chat/completions"]
        C2["POST /v1/chat/completions { prompt, max_tokens, temp }"]
        C3["SSE Token Stream Response"]
    end

    subgraph NebiusStack["Nebius AI Cloud (sky-nebius.yaml)"]
        N_GW["Go API Gateway (:8080)"]
        N_RUST["Rust Candle Engine (:50051)"]
        N_GPU["NVIDIA H100 80GB SXM5 (~41-43 tok/s)"]
        N_GW --> N_RUST --> N_GPU
    end

    subgraph GCPStack["Google Cloud Platform (sky-gcp.yaml)"]
        G_GW["Go API Gateway (:8080)"]
        G_RUST["Rust Candle Engine (:50051)"]
        G_GPU["NVIDIA L4 24GB / A100 (~15-25 tok/s)"]
        G_GW --> G_RUST --> G_GPU
    end

    UniformClient ===>|"100% Contract Parity"| NebiusStack
    UniformClient ===>|"100% Contract Parity"| GCPStack
```

| Dimension | Nebius Cloud (`sky-nebius.yaml`) | Google Cloud (`sky-gcp.yaml`) | Parity Status |
| :--- | :--- | :--- | :--- |
| **API Contract** | `POST http://<IP>:8080/v1/chat/completions` | `POST http://<IP>:8080/v1/chat/completions` | **Exact (100%)** |
| **Payload Schema** | `{"prompt": "...", "max_tokens": 64, "temperature": 0.7}` | `{"prompt": "...", "max_tokens": 64, "temperature": 0.7}` | **Exact (100%)** |
| **Response Protocol**| Server-Sent Events (`data: {"token": "..."}`) | Server-Sent Events (`data: {"token": "..."}`) | **Exact (100%)** |
| **Health Check** | `GET http://<IP>:8080/health` | `GET http://<IP>:8080/health` | **Exact (100%)** |
| **Engine Runtime** | Pure Rust (`candle-core` + CUDA) | Pure Rust (`candle-core` + CUDA) | **Exact (100%)** |
| **Model Weights** | `google/gemma-4-E2B-it` (SafeTensors) | `google/gemma-4-E2B-it` (SafeTensors) | **Exact (100%)** |
| **Inference Hardware**| NVIDIA H100 80GB SXM5 (~41–43 tok/s) | NVIDIA L4 24GB (~15–20 tok/s) / A100 | **Hardware tier difference only** |

---

## 5. Step-by-Step Execution Plan

```mermaid
flowchart TD
    Step1["Step 1: Configure .skyignore & YAMLs
    - Exclude local models/ directory from upload
    - Embed HF download & nvcc shim in setup script"]
    
    Step2["Step 2: Launch Nebius Cluster with SkyPilot
    - Provision H100 instance via sky-nebius.yaml"]
    
    Step3["Step 3: Automated Fast Provisioning & Build
    - Direct HF snapshot download (10 Gbps ~15s)
    - Compile Rust CUDA worker & Go Gateway"]
    
    Step4["Step 4: Verify Nebius Live Endpoint
    - Query stream.sh against Nebius Public IP"]
    
    Step5["Step 5: Apply Mirror Blueprint to GCP
    - Sync sky-gcp.yaml with identical scripts"]
    
    Step6["Step 6: Launch & Verify GCP Parity
    - Test identical streaming completions on GCP IP"]

    Step1 --> Step2 --> Step3 --> Step4 --> Step5 --> Step6
```

### Execution Commands:

1. **Add `.skyignore`**: Ensure `models/` and `target/` are never synced from local disk over Wi-Fi.
2. **Update `sky-nebius.yaml`**: Embed the fast Hugging Face snapshot download and CUDA version shim directly in the setup script.
3. **Launch Nebius**:
   ```bash
   sky launch -c gemma4-nebius sky-nebius.yaml --yes
   ```
4. **Verify Nebius Endpoint**:
   ```bash
   NEBIUS_IP=$(sky status --ip gemma4-nebius)
   curl -N -X POST http://$NEBIUS_IP:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Explain why Rust and Go make a great pair.", "max_tokens": 50}'
   ```
5. **Update `sky-gcp.yaml`**: Mirror all changes to the GCP configuration.
6. **Launch & Verify GCP**:
   ```bash
   sky launch -c gemma4-gcp sky-gcp.yaml --yes
   GCP_IP=$(sky status --ip gemma4-gcp)
   curl -N -X POST http://$GCP_IP:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Explain why Rust and Go make a great pair.", "max_tokens": 50}'
   ```
