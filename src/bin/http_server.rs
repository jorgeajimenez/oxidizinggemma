//! =============================================================================
//! Gemma 4 Native HTTP Inference Server
//! =============================================================================

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use gemma_hello::gemma4::{Gemma4ForCausalLM, Gemma4TopConfig};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

#[derive(Deserialize)]
struct PromptRequest {
    prompt: Option<String>,
}

#[derive(Serialize)]
#[allow(dead_code)]
struct GenerateResponse {
    model: String,
    response: String,
    tokens: usize,
    duration_ms: u128,
}

struct Engine {
    model: Gemma4ForCausalLM,
    tokenizer: Tokenizer,
    device: Device,
    eos_token_id: u32,
}

impl Engine {
    fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<(String, usize, u128)> {
        let start = Instant::now();
        let formatted_prompt = format!("<|turn>user\n{}<turn|>\n<|turn>model\n", prompt);
        let encoding = self.tokenizer.encode(formatted_prompt, true).map_err(anyhow::Error::msg)?;
        let prompt_tokens = encoding.get_ids();
        let prompt_len = prompt_tokens.len();

        let mut logits_processor = LogitsProcessor::new(1337, Some(0.7), Some(0.9));
        let mut generated_tokens = 0;
        let mut output_text = String::new();

        let prompt_tensor = Tensor::new(prompt_tokens, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&prompt_tensor, 0)?.squeeze(0)?;
        let mut next_token = logits_processor.sample(&logits)?;

        for step in 0..max_tokens {
            if next_token == self.eos_token_id || next_token == 1 {
                break;
            }
            let token_str = self.tokenizer.decode(&[next_token], false).unwrap_or_default();
            output_text.push_str(&token_str);
            generated_tokens += 1;

            let input_tensor = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input_tensor, prompt_len + step)?.squeeze(0)?;
            next_token = logits_processor.sample(&logits)?;
        }

        self.model.clear_kv_cache();
        let duration = start.elapsed().as_millis();
        Ok((output_text, generated_tokens, duration))
    }
}

fn get_safetensors_files<P: AsRef<Path>>(dir: P) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            files.push(path);
        }
    }
    files.sort();
    if files.is_empty() {
        anyhow::bail!("No .safetensors files found in the specified model directory");
    }
    Ok(files)
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Gemma 4 Native Candle HTTP Server ===");

    let model_dir = Path::new("models/gemma-4-e2b");
    if !model_dir.exists() {
        anyhow::bail!("models/gemma-4-e2b directory does not exist.");
    }

    let config_path = model_dir.join("config.json");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let safetensor_files = get_safetensors_files(model_dir)?;

    println!("1. Loading configuration from {:?}", config_path);
    let top_config: Gemma4TopConfig = serde_json::from_reader(std::fs::File::open(&config_path)?)?;
    let text_config = top_config.text_config;

    println!("2. Initializing compute device");
    let device = Device::cuda_if_available(0)
        .or_else(|_| Device::new_metal(0))
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

    println!("5. Instantiating Gemma 4 neural network into memory");
    let load_start = Instant::now();
    let model = Gemma4ForCausalLM::new(&text_config, vb)?;
    println!("   Model initialized and warm in {:.2?}", load_start.elapsed());

    let engine = Arc::new(Mutex::new(Engine {
        model,
        tokenizer,
        device,
        eos_token_id,
    }));

    let listener = TcpListener::bind("0.0.0.0:8080").await?;
    println!("\n🚀 Gemma 4 Candle HTTP Server listening on http://0.0.0.0:8080\n");

    loop {
        let (mut socket, _) = listener.accept().await?;
        let engine_clone = Arc::clone(&engine);

        tokio::spawn(async move {
            let mut buf = vec![0u8; 16384];
            let n = match socket.read(&mut buf).await {
                Ok(n) if n > 0 => n,
                _ => return,
            };

            let req_str = String::from_utf8_lossy(&buf[..n]);
            
            // Extract body after \r\n\r\n or \n\n
            let body = if let Some(idx) = req_str.find("\r\n\r\n") {
                &req_str[idx + 4..]
            } else if let Some(idx) = req_str.find("\n\n") {
                &req_str[idx + 2..]
            } else {
                ""
            };

            let prompt = if let Ok(parsed) = serde_json::from_str::<PromptRequest>(body) {
                parsed.prompt.unwrap_or_else(|| body.trim().to_string())
            } else if !body.trim().is_empty() {
                body.trim().to_string()
            } else {
                "Hello from Gemma 4!".to_string()
            };

            let result = {
                let mut guard = engine_clone.lock().await;
                guard.generate(&prompt, 128)
            };

            let (status, resp_body) = match result {
                Ok((text, tokens, dur)) => {
                    let json = serde_json::json!({
                        "model": "gemma-4-e2b",
                        "response": text,
                        "tokens": tokens,
                        "duration_ms": dur
                    });
                    ("200 OK", json.to_string())
                }
                Err(e) => {
                    let json = serde_json::json!({
                        "error": e.to_string()
                    });
                    ("500 Internal Server Error", json.to_string())
                }
            };

            let response = format!(
                "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                status,
                resp_body.len(),
                resp_body
            );

            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.flush().await;
        });
    }
}
