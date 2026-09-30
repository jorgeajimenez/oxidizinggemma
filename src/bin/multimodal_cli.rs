//! =============================================================================
//! Gemma 4 Multimodal Native Engine — Image and Video Inference CLI
//! =============================================================================
//!
//! Run with an image:
//!     cargo run --bin multimodal_cli --release -- photo.jpg "Describe what is in this image."
//!
//! Run with a video:
//!     cargo run --bin multimodal_cli --release -- clip.mp4 "Summarize the events in this video."
//!
//! Run on NVIDIA GPU with CUDA:
//!     cargo run --bin multimodal_cli --release --no-default-features --features cuda -- photo.jpg "Analyze the visual details."

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use gemma_hello::multimodal::{Gemma4ForConditionalGeneration, Gemma4MultimodalConfig};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use tokenizers::Tokenizer;

/// Scans the model directory and collects all `.safetensors` weight shards.
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
        bail!("No .safetensors files found in the specified model directory");
    }
    Ok(files)
}

/// Loads a single image file, resizes to 256x256, normalizes, and returns a (1, 3, 256, 256) Tensor.
fn load_single_image<P: AsRef<Path>>(image_path: P, dtype: DType, device: &Device) -> Result<Tensor> {
    let img = image::open(&image_path)
        .with_context(|| format!("Failed to open image at {:?}", image_path.as_ref()))?;
    
    let resized = img.resize_exact(256, 256, image::imageops::FilterType::Triangle).to_rgb8();
    let raw_pixels = resized.into_raw();
    
    let float_pixels: Vec<f32> = raw_pixels.into_iter().map(|v| v as f32 / 255.0).collect();
    
    let tensor = Tensor::from_vec(float_pixels, (256, 256, 3), device)?
        .permute((2, 0, 1))?
        .unsqueeze(0)?
        .to_dtype(dtype)?;
        
    Ok(tensor)
}

/// Extracts N evenly spaced frames from a video file using ffmpeg, resizing to 256x256,
/// and returns a batched Tensor of shape (NumFrames, 3, 256, 256).
fn load_video_frames<P: AsRef<Path>>(video_path: P, num_frames: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let temp_dir = std::env::temp_dir().join(format!("gemma4_video_{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir)?;

    let output_pattern = temp_dir.join("frame_%03d.png");
    
    println!("Extracting {} frames from video using ffmpeg...", num_frames);
    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-i")
        .arg(video_path.as_ref())
        .arg("-vf")
        .arg(format!("fps=1,scale=256:256"))
        .arg("-frames:v")
        .arg(num_frames.to_string())
        .arg(&output_pattern)
        .output()
        .with_context(|| "Failed to execute ffmpeg. Please verify ffmpeg is installed.")?;

    if !status.status.success() {
        let err = String::from_utf8_lossy(&status.stderr);
        let _ = std::fs::remove_dir_all(&temp_dir);
        bail!("ffmpeg frame extraction failed: {}", err);
    }

    let mut frame_tensors = Vec::new();
    for i in 1..=num_frames {
        let frame_path = temp_dir.join(format!("frame_{:03}.png", i));
        if frame_path.exists() {
            let frame_tensor = load_single_image(&frame_path, dtype, device)?;
            frame_tensors.push(frame_tensor);
        }
    }

    let _ = std::fs::remove_dir_all(&temp_dir);

    if frame_tensors.is_empty() {
        bail!("No frames could be extracted from video at {:?}", video_path.as_ref());
    }

    println!("Successfully loaded {} video frames into temporal sequence.", frame_tensors.len());
    let batched = Tensor::cat(&frame_tensors.iter().collect::<Vec<_>>(), 0)?;
    Ok(batched)
}

/// Automatically detects whether the input file is an image or video and loads the appropriate Tensor.
fn load_visual_input<P: AsRef<Path>>(file_path: P, dtype: DType, device: &Device) -> Result<Tensor> {
    let path = file_path.as_ref();
    if !path.exists() {
        bail!("Specified visual input file does not exist: {:?}", path);
    }

    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
    match ext.as_str() {
        "mp4" | "mov" | "avi" | "mkv" | "webm" | "m4v" => {
            println!("Detected video input format ({:?}). Extracting temporal frames...", ext);
            load_video_frames(path, 8, dtype, device)
        }
        _ => {
            println!("Detected image input format ({:?}). Loading 256x256 image tensor...", ext);
            load_single_image(path, dtype, device)
        }
    }
}

fn parse_cli_args() -> Result<(PathBuf, String)> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("\nUsage: cargo run --bin multimodal_cli --release -- <image_or_video_path> [prompt]");
        eprintln!("Examples:");
        eprintln!("  cargo run --bin multimodal_cli --release -- photo.jpg \"Describe this image.\"");
        eprintln!("  cargo run --bin multimodal_cli --release -- clip.mp4 \"Summarize the action in this video.\"\n");
        bail!("Missing required visual input file argument.");
    }

    let input_path = PathBuf::from(&args[1]);
    let prompt = if args.len() > 2 {
        args[2..].join(" ")
    } else {
        "Describe what is happening in this visual scene in detail.".to_string()
    };

    Ok((input_path, prompt))
}

fn main() -> Result<()> {
    println!("=== Oxidizing Gemma 4 — Image and Video Multimodal CLI ===");

    // 1. Parse Arguments (Requires an image or video file)
    let (visual_path, user_prompt) = parse_cli_args()?;

    // 2. Locate model directory
    let model_dir = Path::new("models/gemma-4-e2b");
    if !model_dir.exists() {
        bail!("Model directory 'models/gemma-4-e2b' does not exist.");
    }

    let config_path = model_dir.join("config.json");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let safetensor_files = get_safetensors_files(model_dir)?;

    println!("1. Loading multimodal configuration from {:?}", config_path);
    let mm_config: Gemma4MultimodalConfig = serde_json::from_reader(std::fs::File::open(&config_path)?)?;

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

    println!("5. Instantiating Gemma 4 Multimodal Model (Vision Tower + Language Backbone)");
    let load_start = Instant::now();
    let mut model = Gemma4ForConditionalGeneration::new(&mm_config, vb)?;
    println!("   Multimodal model warm in {:.2?}", load_start.elapsed());

    // 6. Load Visual Input Tensor (Image or Video)
    println!("6. Processing visual input: {:?}", visual_path);
    let visual_tensor = load_visual_input(&visual_path, dtype, &device)?;

    let formatted_prompt = format!("<|turn>user\n<|image|>\n{}<turn|>\n<|turn>model\n", user_prompt);
    println!("Formatted Prompt: \"{}\"", formatted_prompt);
    println!("--- Generating Multimodal Response ---");

    let encoding = tokenizer.encode(formatted_prompt.as_str(), true).map_err(anyhow::Error::msg)?;
    let prompt_tokens = encoding.get_ids();
    let prompt_tensor = Tensor::new(prompt_tokens, &device)?.unsqueeze(0)?;

    let mut logits_processor = LogitsProcessor::new(42, Some(0.7), Some(0.9));
    let mut generated_count = 0;
    let max_tokens = 100;

    let gen_start = Instant::now();

    // 7. Multimodal Prefill Phase (Vision Tokens + Prompt Tokens)
    let logits = model.forward(&prompt_tensor, Some(&visual_tensor), 0)?;
    let mut next_token = logits_processor.sample(&logits.squeeze(0)?)?;

    // 8. Autoregressive Streaming Decode
    for step in 0..max_tokens {
        if next_token == eos_token_id || next_token == 1 {
            break;
        }

        let token_str = tokenizer.decode(&[next_token], false).unwrap_or_default();
        print!("{}", token_str);
        use std::io::Write;
        std::io::stdout().flush()?;
        generated_count += 1;

        let logits = model.decode_step(next_token, prompt_tokens.len() + step, &device)?;
        next_token = logits_processor.sample(&logits.squeeze(0)?)?;
    }

    let elapsed = gen_start.elapsed();
    let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
        generated_count as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    println!("\n\nGenerated {} tokens in {:.2?} ({:.2} tok/sec)", generated_count, elapsed, tok_per_sec);
    model.clear_kv_cache();

    Ok(())
}
