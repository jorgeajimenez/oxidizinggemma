//! # Gemma 4 High-Performance gRPC Inference Worker
//!
//! A persistent, long-running microservice built with Tonic and Tokio.
//! Loads Gemma 4 weights into GPU memory (Apple Metal or NVIDIA CUDA) once at boot,
//! keeping the neural network warm and streaming generated tokens in real time.

pub mod inference {
    tonic::include_proto!("inference");
}

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use gemma_hello::gemma4::{Gemma4ForCausalLM, Gemma4TopConfig};
use inference::inference_service_server::{InferenceService, InferenceServiceServer};
use inference::{GenerateRequest, GenerateResponse};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{transport::Server, Request, Response, Status};

/// Helper to scan for `.safetensors` files in the model directory.
fn get_safetensors_files<P: AsRef<Path>>(dir: P) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            files.push(path);
        }
    }
    files.sort();
    if files.is_empty() {
        anyhow::bail!("No .safetensors files found in {:?}", dir.as_ref());
    }
    Ok(files)
}

/// Shared worker state holding the warm neural network model, tokenizer, and hardware device.
pub struct WorkerState {
    pub model: Gemma4ForCausalLM,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub eos_token_id: u32,
}

/// Implementation of the generated gRPC InferenceService.
pub struct InferenceServiceImpl {
    state: Arc<Mutex<WorkerState>>,
}

impl InferenceServiceImpl {
    pub fn new(state: Arc<Mutex<WorkerState>>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl InferenceService for InferenceServiceImpl {
    type StreamGenerateStream = ReceiverStream<Result<GenerateResponse, Status>>;

    async fn stream_generate(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<Self::StreamGenerateStream>, Status> {
        let req = request.into_inner();
        let prompt_raw = req.prompt;
        let max_tokens = if req.max_tokens <= 0 { 128 } else { req.max_tokens as usize };
        let temperature = if req.temperature > 0.0 { Some(req.temperature as f64) } else { None };

        println!("[gRPC] Incoming request: {:?} (max_tokens={}, temp={:?})", prompt_raw, max_tokens, temperature);

        let state_clone = self.state.clone();
        let (tx, rx) = mpsc::channel(128);

        tokio::spawn(async move {
            let mut state = state_clone.lock().await;

            // Canonical Gemma 4 turn delimiter formatting
            let formatted_prompt = format!("<|turn>user\n{}<turn|>\n<|turn>model\n", prompt_raw.trim());

            let prompt_tokens = match state.tokenizer.encode(formatted_prompt.as_str(), true) {
                Ok(encoding) => encoding.get_ids().to_vec(),
                Err(err) => {
                    let _ = tx.send(Err(Status::internal(format!("Tokenization error: {}", err)))).await;
                    return;
                }
            };

            let prompt_len = prompt_tokens.len();
            let mut logits_processor = LogitsProcessor::new(42, temperature, None);
            let eos_token_id = state.eos_token_id;
            let device = state.device.clone();

            let gen_start = Instant::now();
            let mut generated_count = 0;

            // 1. Prefill Phase (Forward entire prompt at position 0)
            let prompt_tensor = match Tensor::new(&prompt_tokens[..], &device).and_then(|t| t.unsqueeze(0)) {
                Ok(t) => t,
                Err(err) => {
                    let _ = tx.send(Err(Status::internal(format!("Prompt tensor creation failed: {}", err)))).await;
                    state.model.clear_kv_cache();
                    return;
                }
            };

            let logits = match state.model.forward(&prompt_tensor, 0).and_then(|l| l.squeeze(0)) {
                Ok(l) => l,
                Err(err) => {
                    let _ = tx.send(Err(Status::internal(format!("Prefill forward pass failed: {}", err)))).await;
                    state.model.clear_kv_cache();
                    return;
                }
            };

            let mut next_token = match logits_processor.sample(&logits) {
                Ok(t) => t,
                Err(err) => {
                    let _ = tx.send(Err(Status::internal(format!("Initial sampling failed: {}", err)))).await;
                    state.model.clear_kv_cache();
                    return;
                }
            };

            // 2. Autoregressive Streaming Decode Phase
            for step in 0..max_tokens {
                if next_token == eos_token_id || next_token == 1 {
                    break;
                }

                let token_str = state.tokenizer.decode(&[next_token], false).unwrap_or_default();
                generated_count += 1;

                if tx.send(Ok(GenerateResponse {
                    token: token_str,
                    is_final: false,
                })).await.is_err() {
                    // HTTP client or gateway hung up
                    break;
                }

                let input_tensor = match Tensor::new(&[next_token], &device).and_then(|t| t.unsqueeze(0)) {
                    Ok(t) => t,
                    Err(err) => {
                        let _ = tx.send(Err(Status::internal(format!("Step tensor creation failed: {}", err)))).await;
                        break;
                    }
                };

                let logits = match state.model.forward(&input_tensor, prompt_len + step) {
                    Ok(l) => match l.squeeze(0) {
                        Ok(sq) => sq,
                        Err(err) => {
                            let _ = tx.send(Err(Status::internal(format!("Logits squeeze failed: {}", err)))).await;
                            break;
                        }
                    },
                    Err(err) => {
                        let _ = tx.send(Err(Status::internal(format!("Decode forward pass failed: {}", err)))).await;
                        break;
                    }
                };

                next_token = match logits_processor.sample(&logits) {
                    Ok(t) => t,
                    Err(err) => {
                        let _ = tx.send(Err(Status::internal(format!("Sampling failed: {}", err)))).await;
                        break;
                    }
                };
            }

            let elapsed = gen_start.elapsed();
            let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
                generated_count as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            };
            println!(
                "Streamed {} tokens in {:.2?} ({:.2} tok/sec)",
                generated_count, elapsed, tok_per_sec
            );

            // Clean up KV cache for subsequent requests to avoid VRAM accumulation
            state.model.clear_kv_cache();

            // Send final termination packet
            let _ = tx.send(Ok(GenerateResponse {
                token: String::new(),
                is_final: true,
            })).await;
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Gemma 4 High-Performance gRPC Inference Worker ===");

    let model_dir = Path::new("models/gemma-4-e2b");
    if !model_dir.exists() {
        eprintln!("Error: models/gemma-4-e2b directory does not exist.");
        return Ok(());
    }

    let config_path = model_dir.join("config.json");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let safetensor_files = get_safetensors_files(model_dir)?;

    println!("1. Loading configuration from {:?}", config_path);
    let top_config: Gemma4TopConfig = serde_json::from_reader(std::fs::File::open(&config_path)?)?;
    let text_config = top_config.text_config;

    println!("2. Initializing compute device");
    let device = Device::new_metal(0)
        .or_else(|_| Device::cuda_if_available(0))
        .unwrap_or(Device::Cpu);

    let dtype = if device.is_metal() || device.is_cuda() {
        DType::F16
    } else {
        DType::F32
    };
    println!("   Active hardware device: {:?} (Precision: {:?})", device, dtype);

    println!("3. Loading tokenizer from {:?}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(anyhow::Error::msg)?;
    let eos_token_id = tokenizer.token_to_id("<turn|>").unwrap_or(1);

    println!("4. Memory-mapping {} safetensors file(s)", safetensor_files.len());
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, &device)? };

    println!("5. Instantiating Gemma 4 neural network into GPU memory");
    let load_start = Instant::now();
    let model = Gemma4ForCausalLM::new(&text_config, vb)?;
    println!("   Model initialized and warm in {:.2?}", load_start.elapsed());

    let state = Arc::new(Mutex::new(WorkerState {
        model,
        tokenizer,
        device,
        eos_token_id,
    }));

    let service = InferenceServiceImpl::new(state);
    let addr = "0.0.0.0:50051".parse()?;

    println!("\nGemma 4 gRPC Worker listening on http://{}", addr);
    Server::builder()
        .add_service(InferenceServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}