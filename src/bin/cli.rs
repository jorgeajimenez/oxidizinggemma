//! =============================================================================
//! Gemma 4 Native Inference Engine — Standalone Interactive / Single-Shot CLI
//! =============================================================================
//!
//! Run with:
//!     cargo run --bin cli --release
//! Or with CUDA:
//!     cargo run --bin cli --release --no-default-features --features cuda

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use gemma_hello::gemma4::{Gemma4ForCausalLM, Gemma4TopConfig};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;

/// Scans the specified directory and collects all `.safetensors` weight shards.
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

fn main() -> Result<()> {
    println!("=== Oxidizing Gemma 4 — Standalone CLI Engine ===");

    // 1. Locate model directory
    let model_dir = Path::new("models/gemma-4-e2b");
    if !model_dir.exists() {
        eprintln!("Error: models/gemma-4-e2b directory does not exist.");
        eprintln!("Please download the weights first or ensure you are at the repo root.");
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
    let mut model = Gemma4ForCausalLM::new(&text_config, vb)?;
    println!("   Model initialized and warm in {:.2?}", load_start.elapsed());

    // Prompt configuration
    let args: Vec<String> = std::env::args().collect();
    let user_prompt = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        "Explain why Rust and Candle provide an ideal native foundation for LLM inference.".to_string()
    };

    let formatted_prompt = format!("<|turn>user\n{}<turn|>\n<|turn>model\n", user_prompt);
    println!("\nPrompt: \"{}\"", user_prompt);
    println!("--- Generating Response ---");

    let encoding = tokenizer.encode(formatted_prompt, true).map_err(anyhow::Error::msg)?;
    let prompt_tokens = encoding.get_ids();
    let prompt_len = prompt_tokens.len();

    let mut logits_processor = LogitsProcessor::new(1337, Some(0.7), Some(0.9));
    let mut generated_count = 0;
    let max_tokens = 120;

    let gen_start = Instant::now();

    // 1. Prefill Phase: Process the entire prompt prompt_len tokens at once
    let prompt_tensor = Tensor::new(prompt_tokens, &device)?.unsqueeze(0)?;
    let logits = model.forward(&prompt_tensor, 0)?.squeeze(0)?;
    let mut next_token = logits_processor.sample(&logits)?;

    // 2. Decode Phase: Autoregressive single-token streaming loop
    for step in 0..max_tokens {
        if next_token == eos_token_id || next_token == 1 {
            break;
        }

        let token_str = tokenizer.decode(&[next_token], false).unwrap_or_default();
        print!("{}", token_str);
        std::io::stdout().flush()?;
        generated_count += 1;

        let input_tensor = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input_tensor, prompt_len + step)?.squeeze(0)?;
        next_token = logits_processor.sample(&logits)?;
    }

    let elapsed = gen_start.elapsed();
    let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
        generated_count as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    println!("\n---------------------------");
    println!(
        "Generated {} tokens in {:.2?} ({:.2} tok/sec)",
        generated_count, elapsed, tok_per_sec
    );

    // Reset KV cache
    model.clear_kv_cache();

    Ok(())
}
