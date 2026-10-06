# Gemma 4 Multi-Cloud Step-by-Step Task Roadmap

> **Overview:** 10 discrete, sequential tasks to deploy Gemma 4 with zero local uploads and achieve 100% parity across Nebius and GCP. Every task produces a concrete deliverable and an immediate test command.

---

## 1. Live Task Progress & Status Table

| # | Task | Deliverable | Status | Verification Result |
|---|---|---|---|---|
| **1** | Create `.skyignore` to block 9.8GB upload | [`.skyignore`](file:///Users/jorgeajimenez/repos/candle/.skyignore) | **DONE** | Upload payload reduced to **433 KB** (99.99% reduction). |
| **2** | Create CUDA Compatibility Shim | [`scripts/setup_cuda_compat.sh`](file:///Users/jorgeajimenez/repos/candle/scripts/setup_cuda_compat.sh) | **DONE** | Intercepts version to return `12.6`, passes kernel compilation. |
| **3** | Create Model Downloader Script | [`scripts/download_model.py`](file:///Users/jorgeajimenez/repos/candle/scripts/download_model.py) | **DONE** | CLI flags and `huggingface_hub` snapshot download ready. |
| **4** | Standardize `sky-nebius.yaml` | [`sky-nebius.yaml`](file:///Users/jorgeajimenez/repos/candle/sky-nebius.yaml) | **DONE** | Integrated `envs: HF_TOKEN: null`, CUDA shim, HF fetch, & `:50051` readiness probe. |
| **5** | Launch Nebius H100 Cluster | `gemma4-nebius` VM | **DONE** | Provisioned `gpu-h100-sxm_1gpu-16vcpu-200gb` in `eu-north1`. |
| **6** | Verify Remote Build & Weights | Remote Binaries & Weights | **DONE** | 9.5GB SafeTensors downloaded in ~15s, Rust CUDA + Go Gateway compiled. |
| **7** | Test Live Nebius Streaming | Live SSE API on Nebius port `8080` | **DONE** | Verified real-time SSE token stream from `http://89.169.97.17:8080/v1/chat/completions`. |
| **8** | Upgrade `stream.sh` Auto-Discovery | [`stream.sh`](file:///Users/jorgeajimenez/repos/candle/stream.sh) | **DONE** | Auto-detects `gemma4-nebius` and `gemma4-gcp` IPs, renders clean typewriter tokens. |
| **9** | Update `sky-gcp.yaml` for Parity | [`sky-gcp.yaml`](file:///Users/jorgeajimenez/repos/candle/sky-gcp.yaml) | **DONE** | Standardized for 100% cloud parity (HF snapshot download, CUDA shim, `:50051` readiness loop). |
| **10** | Launch & Verify GCP Parity | Live SSE API on GCP port `8080` | **NEXT UP** | Launch with `sky launch -c gemma4-gcp sky-gcp.yaml --yes` once GCP credentials/project are authenticated. |
