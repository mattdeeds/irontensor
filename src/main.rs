use irontensor::{
    GPTModel, MetalContext, ModelConfig, Profiler, ProfilerConfig, TokenDataset,
    TrainConfigSnapshot, Trainer, TrainingConfig,
};
use irontensor::logging::{InferenceTimer, LogConfig, Logger, TrainStepRecord};
use irontensor::train::{TrainCallback, TrainMetrics};
use objc2_metal::MTLDevice;
use rand::Rng;
use std::fs;
use std::path::Path;
use std::time::Instant;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{DecoderWrapper, PreTokenizerWrapper, Tokenizer};

const TINY_SHAKESPEARE_URL: &str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

fn main() {
    // Load environment variables from .env file (if it exists)
    dotenvy::dotenv().ok();

    // Initialize the Metal context
    let ctx = MetalContext::global();
    println!("IronTensor - GPT Training on Tiny Shakespeare");
    println!("==============================================");
    println!("Device: {}\n", ctx.device().name());

    // Check if logging is enabled (profiler is enabled when logging is enabled)
    let logging_enabled = std::env::var("IRONTENSOR_LOG").is_ok();

    // Initialize profiler (enabled when logging is enabled)
    Profiler::init(ProfilerConfig {
        enabled: logging_enabled,
        warmup_steps: 5,
        report_interval: 0,
    });

    // Parse configuration from environment variables
    let model_name = std::env::var("IRONTENSOR_MODEL").unwrap_or_else(|_| "shakespeare".to_string());
    let total_steps: usize = std::env::var("IRONTENSOR_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let batch_size: usize = std::env::var("IRONTENSOR_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    println!("Configuration:");
    println!("  Model: {}", model_name);
    println!("  Steps: {}", total_steps);
    println!("  Batch size: {}", batch_size);
    println!();

    // Create data directory if it doesn't exist
    fs::create_dir_all("data").expect("Failed to create data directory");
    fs::create_dir_all("checkpoints").expect("Failed to create checkpoints directory");

    // =====================================================================
    // Step 1: Download or load Tiny Shakespeare
    // =====================================================================
    println!("=== Step 1: Loading Tiny Shakespeare ===\n");

    let text_path = Path::new("data/tinyshakespeare.txt");
    let text = if text_path.exists() {
        println!("Loading from cached file: {:?}", text_path);
        fs::read_to_string(text_path).expect("Failed to read cached file")
    } else {
        println!("Downloading Tiny Shakespeare from GitHub...");
        let text = download_text(TINY_SHAKESPEARE_URL);
        fs::write(text_path, &text).expect("Failed to cache text");
        println!("Saved to: {:?}", text_path);
        text
    };

    println!("Text length: {} characters", text.len());
    println!("First 200 chars:\n{}\n", &text[..200.min(text.len())]);

    // =====================================================================
    // Step 2: Train BPE Tokenizer
    // =====================================================================
    println!("=== Step 2: Training BPE Tokenizer ===\n");

    let tokenizer_path = Path::new("data/shakespeare_tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        println!("Loading cached tokenizer from: {:?}", tokenizer_path);
        Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer")
    } else {
        println!("Training new BPE tokenizer (vocab_size=2048)...");
        let tokenizer = train_bpe_tokenizer(&text, 2048);
        tokenizer.save(tokenizer_path, false).expect("Failed to save tokenizer");
        println!("Saved tokenizer to: {:?}", tokenizer_path);
        tokenizer
    };

    let vocab_size = tokenizer.get_vocab_size(true);
    println!("Vocabulary size: {}", vocab_size);

    // Test tokenization
    let sample = "ROMEO:\nWherefore art thou Romeo?";
    let encoding = tokenizer.encode(sample, false).expect("Failed to encode");
    let tokens = encoding.get_ids();
    println!("Sample: \"{}\"", sample);
    println!("Tokens: {:?}", tokens);
    let decoded = tokenizer.decode(tokens, true).expect("Failed to decode");
    println!("Decoded: \"{}\"\n", decoded);

    // =====================================================================
    // Step 3: Prepare Datasets
    // =====================================================================
    println!("=== Step 3: Preparing Datasets ===\n");

    let train_path = Path::new("data/shakespeare_train.bin");
    let val_path = Path::new("data/shakespeare_val.bin");

    // Check if datasets already exist
    let datasets_exist = train_path.exists() && val_path.exists();

    if !datasets_exist {
        // Tokenize entire text
        println!("Tokenizing entire corpus...");
        let encoding = tokenizer.encode(text.as_str(), false).expect("Failed to encode");
        let all_tokens: Vec<u32> = encoding.get_ids().to_vec();
        println!("Total tokens: {}", all_tokens.len());

        // Split 90/10 for train/val
        let split_idx = (all_tokens.len() as f64 * 0.9) as usize;
        let train_tokens = &all_tokens[..split_idx];
        let val_tokens = &all_tokens[split_idx..];

        println!("Train tokens: {}", train_tokens.len());
        println!("Val tokens: {}", val_tokens.len());

        // Create binary datasets
        TokenDataset::create(train_path, train_tokens).expect("Failed to create train dataset");
        TokenDataset::create(val_path, val_tokens).expect("Failed to create val dataset");
        println!("Created train dataset: {:?}", train_path);
        println!("Created val dataset: {:?}", val_path);
    } else {
        println!("Using cached datasets:");
        println!("  Train: {:?}", train_path);
        println!("  Val: {:?}", val_path);
    }
    println!();

    // =====================================================================
    // Step 4: Initialize Model
    // =====================================================================
    println!("=== Step 4: Initializing Model ===\n");

    // Create model config matching tokenizer vocab size
    let mut config = match model_name.as_str() {
        "tiny" => ModelConfig::tiny(),
        "small" => ModelConfig::small(),
        "medium" => ModelConfig::medium(),
        _ => ModelConfig::shakespeare(),
    };
    config.vocab_size = vocab_size; // Match actual tokenizer vocab

    println!("Model Configuration:");
    println!("  vocab_size: {}", config.vocab_size);
    println!("  hidden_dim: {}", config.hidden_dim);
    println!("  num_layers: {}", config.num_layers);
    println!("  num_heads: {}", config.num_heads);
    println!("  intermediate_dim: {}", config.intermediate_dim);
    println!("  max_seq_len: {}", config.max_seq_len);
    println!("  tie_weights: {}", config.tie_weights);

    let model = GPTModel::new(&config);
    println!("\n{}", model.summary());

    let expected_initial_loss = (vocab_size as f64).ln();
    println!("Expected initial loss (random): {:.4} (ln({}))\n", expected_initial_loss, vocab_size);

    // =====================================================================
    // Step 5: Training
    // =====================================================================
    println!("=== Step 5: Training ===\n");

    let seq_len = 256; // Sequence length for training

    let train_dataset = TokenDataset::open(train_path, seq_len).expect("Failed to open train dataset");
    let val_dataset = TokenDataset::open(val_path, seq_len).expect("Failed to open val dataset");

    println!("Train sequences: {}", train_dataset.num_sequences());
    println!("Val sequences: {}", val_dataset.num_sequences());
    println!("Batch size: {}", batch_size);
    println!("Sequence length: {}", seq_len);

    // Check for BF16 flag via environment variable
    let use_bf16 = std::env::var("IRONTENSOR_BF16").is_ok();
    // Check for async GPU flag (enabled by default, disable with IRONTENSOR_SYNC_GPU=1)
    let async_gpu = std::env::var("IRONTENSOR_SYNC_GPU").is_err();

    // Training configuration
    let train_config = TrainingConfig {
        learning_rate: 3e-4,
        weight_decay: 0.25,
        beta1: 0.9,
        beta2: 0.99,
        max_grad_norm: 1.0,
        warmup_steps: 50,
        total_steps,
        log_interval: 10,  // More frequent logging
        save_interval: usize::MAX,  // Disable checkpoints during testing
        eval_interval: 50,
        checkpoint_dir: "checkpoints".to_string(),
        use_bf16,
        async_gpu,
    };

    println!("\nTraining Configuration:");
    println!("  learning_rate: {}", train_config.learning_rate);
    println!("  weight_decay: {}", train_config.weight_decay);
    println!("  warmup_steps: {}", train_config.warmup_steps);
    println!("  total_steps: {}", train_config.total_steps);
    println!("  max_grad_norm: {}", train_config.max_grad_norm);
    println!("  use_bf16: {}", train_config.use_bf16);
    println!("  async_gpu: {}", train_config.async_gpu);
    println!();

    // Initialize logger
    Logger::init(LogConfig {
        enabled: logging_enabled,
        log_dir: std::env::var("IRONTENSOR_LOG_DIR").unwrap_or_else(|_| "logs".to_string()),
        model_name: model_name.clone(),
        config: TrainConfigSnapshot {
            learning_rate: train_config.learning_rate,
            weight_decay: train_config.weight_decay,
            batch_size,
            seq_len,
            warmup_steps: train_config.warmup_steps,
            total_steps: train_config.total_steps,
            use_bf16: train_config.use_bf16,
        },
        include_op_breakdown: std::env::var("IRONTENSOR_LOG_OPS").is_ok(),
        ..Default::default()
    });
    if logging_enabled {
        println!("Logging enabled (will write to: {})\n",
            Logger::log_path().map(|p| p.display().to_string()).unwrap_or_else(|| "logs".to_string()));
    }

    // Create trainer
    let mut trainer = Trainer::new(&config, &train_config);

    // Create logging callback
    let mut callback = LoggingCallback::new(batch_size, seq_len);

    // Start training timer
    Logger::start_training();
    let training_start = Instant::now();

    println!("Starting training...");
    println!("{}", "-".repeat(60));

    // Use Trainer::train_epoch for proper training loop
    let num_epochs = (train_config.total_steps / (train_dataset.num_sequences() / batch_size).max(1)).max(1) + 1;
    trainer.train(&train_dataset, Some(&val_dataset), batch_size, num_epochs, &mut callback);

    let total_time_sec = training_start.elapsed().as_secs_f32();
    println!("{}", "-".repeat(60));
    println!("Training complete in {:.1}s!\n", total_time_sec);

    // Get profiler report (included in JSON log if logging enabled)
    let profiler_report = if logging_enabled {
        Some(Profiler::report().to_record())
    } else {
        None
    };

    // Finalize training log
    if logging_enabled {
        Logger::finalize_training(
            trainer.step,  // Actual total steps completed
            callback.last_loss,
            callback.best_val_loss,
            trainer.epoch,
            profiler_report,
        );
    }

    // =====================================================================
    // Step 6: Text Generation
    // =====================================================================
    println!("=== Step 6: Text Generation ===\n");

    let prompts = vec![
        "ROMEO:",
        "To be or not",
        "The king",
        "JULIET:\nO Romeo,",
    ];

    for prompt in prompts {
        println!("Prompt: \"{}\"", prompt);
        let generated = generate(&trainer.model, &tokenizer, prompt, 100, 0.8);
        println!("Generated:\n{}\n", generated);
        println!("{}", "-".repeat(40));
    }

    // Shutdown logger (writes the complete log file)
    Logger::shutdown();

    println!("\nDone!");
}

/// Callback that prints to console and logs to file.
struct LoggingCallback {
    batch_size: usize,
    seq_len: usize,
    step_start: Instant,
    last_loss: f32,
    best_val_loss: Option<f32>,
}

impl LoggingCallback {
    fn new(batch_size: usize, seq_len: usize) -> Self {
        Self {
            batch_size,
            seq_len,
            step_start: Instant::now(),
            last_loss: 0.0,
            best_val_loss: None,
        }
    }
}

impl TrainCallback for LoggingCallback {
    fn on_step(&mut self, metrics: &TrainMetrics) {
        self.last_loss = metrics.loss;

        // Print to console
        println!(
            "Step {:>6} | Loss: {:.4} | Grad norm: {:.4} | LR: {:.2e} | {:.0} tok/s",
            metrics.step,
            metrics.loss,
            metrics.grad_norm,
            metrics.learning_rate,
            metrics.tokens_per_sec
        );

        // Log to file
        if Logger::is_enabled() {
            let elapsed = self.step_start.elapsed().as_secs_f32();
            let step_time_ms = elapsed * 1000.0;

            let record = TrainStepRecord::new(
                metrics.step,
                0, // epoch is tracked by trainer
                metrics.loss,
                metrics.grad_norm,
                metrics.learning_rate,
                metrics.tokens_per_sec,
                step_time_ms,
                self.batch_size,
                self.seq_len,
            );
            Logger::log_train_step(&record);
        }

        self.step_start = Instant::now();
    }

    fn on_eval(&mut self, step: usize, val_loss: f32) {
        println!("Step {:>6} | Validation loss: {:.4}", step, val_loss);

        if self.best_val_loss.is_none() || val_loss < self.best_val_loss.unwrap() {
            self.best_val_loss = Some(val_loss);
        }
    }

    fn on_save(&mut self, step: usize, path: &str) {
        println!("Step {:>6} | Saved checkpoint to: {}", step, path);
    }
}

/// Download text from a URL (simple blocking HTTP)
fn download_text(url: &str) -> String {
    // Use a simple curl command for downloading
    use std::process::Command;

    let output = Command::new("curl")
        .args(["-s", url])
        .output()
        .expect("Failed to execute curl");

    String::from_utf8(output.stdout).expect("Invalid UTF-8 in response")
}

/// Train a BPE tokenizer on the given text
fn train_bpe_tokenizer(text: &str, vocab_size: usize) -> Tokenizer {
    use tokenizers::models::TrainerWrapper;

    // Create a temporary file for training
    let temp_path = std::env::temp_dir().join("bpe_train_text.txt");
    fs::write(&temp_path, text).expect("Failed to write temp file");

    // Create trainer wrapped for the generic Tokenizer
    let trainer = BpeTrainerBuilder::new()
        .vocab_size(vocab_size)
        .min_frequency(2)
        .special_tokens(vec![
            tokenizers::AddedToken::from("<|endoftext|>", true),
            tokenizers::AddedToken::from("<|pad|>", true),
        ])
        .build();
    let mut trainer_wrapper = TrainerWrapper::BpeTrainer(trainer);

    // Build the tokenizer with empty BPE model
    let mut tokenizer = Tokenizer::new(BPE::default());

    // Use byte-level pre-tokenization (like GPT-2)
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::ByteLevel(ByteLevel::default())));

    // Train
    tokenizer
        .train_from_files(&mut trainer_wrapper, vec![temp_path.to_str().unwrap().to_string()])
        .expect("Failed to train tokenizer");

    // Add decoder for proper decoding
    tokenizer.with_decoder(Some(DecoderWrapper::ByteLevel(
        tokenizers::decoders::byte_level::ByteLevel::default(),
    )));

    // Cleanup
    fs::remove_file(&temp_path).ok();

    tokenizer
}

/// Generate text from a prompt with optional performance logging.
fn generate(
    model: &GPTModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> String {
    // Encode prompt
    let encoding = tokenizer.encode(prompt, false).expect("Failed to encode prompt");
    let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_len = tokens.len();

    let max_len = model.config.max_seq_len;

    // Start inference timing
    let mut timer = InferenceTimer::new(prompt_len, temperature);

    // Generate tokens one at a time
    for i in 0..max_tokens {
        // Truncate if necessary
        let context_tokens = if tokens.len() > max_len {
            &tokens[tokens.len() - max_len..]
        } else {
            &tokens[..]
        };

        // Forward pass
        let logits = model.forward(context_tokens, 1, context_tokens.len(), 0);

        // Get logits for last position
        let vocab_size = model.config.vocab_size;
        let last_pos = context_tokens.len() - 1;
        let logits_slice = logits.as_f32_slice();
        let last_logits = &logits_slice[last_pos * vocab_size..(last_pos + 1) * vocab_size];

        // Apply temperature
        let scaled_logits: Vec<f32> = if temperature > 0.0 {
            last_logits.iter().map(|&x| x / temperature).collect()
        } else {
            last_logits.to_vec()
        };

        // Sample from distribution
        let next_token = if temperature > 0.0 {
            sample_from_logits(&scaled_logits)
        } else {
            // Greedy: argmax
            scaled_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap()
        };

        tokens.push(next_token);

        // Track inference timing
        if i == 0 {
            timer.mark_first_token();
        } else {
            timer.token_generated();
        }

        // Stop at end of text token
        if next_token == 0 {
            break;
        }
    }

    // Finish timing and log (automatically logs if Logger is enabled)
    let _ = timer.finish();

    // Decode
    tokenizer.decode(&tokens, true).expect("Failed to decode")
}

/// Sample a token from logits using softmax probabilities
fn sample_from_logits(logits: &[f32]) -> u32 {
    // Compute softmax
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

    // Sample using thread-local RNG
    let r: f32 = rand::rng().random();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u32;
        }
    }

    (probs.len() - 1) as u32
}
