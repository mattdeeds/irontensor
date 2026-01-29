use irontensor::{
    GPTModel, GeneratorConfig, MetalContext, ModelConfig, Profiler, ProfilerConfig, TextGenerator,
    TokenDataset, TrainConfigSnapshot, Trainer, TrainingConfig,
};
use irontensor::logging::{LogConfig, Logger, TrainStepRecord};
use irontensor::train::{TrainCallback, TrainMetrics};
use objc2_metal::MTLDevice;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{DecoderWrapper, PreTokenizerWrapper, Tokenizer};

const TINY_SHAKESPEARE_URL: &str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

fn main() {
    let start_time = Instant::now();

    // Load environment variables from .env file (if it exists)
    dotenvy::dotenv().ok();

    // Initialize the Metal context
    let ctx = MetalContext::global();
    let device_name = ctx.device().name().to_string();

    // Parse configuration from environment variables
    let logging_enabled = std::env::var("IRONTENSOR_LOG").is_ok();
    let inference_enabled = std::env::var("IRONTENSOR_INFERENCE")
        .map(|v| v != "0" && v.to_lowercase() != "false")
        .unwrap_or(true);
    let model_name = std::env::var("IRONTENSOR_MODEL").unwrap_or_else(|_| "small".to_string());
    let total_steps: usize = std::env::var("IRONTENSOR_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let batch_size: usize = std::env::var("IRONTENSOR_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let use_bf16 = std::env::var("IRONTENSOR_BF16").is_ok();
    let async_gpu = std::env::var("IRONTENSOR_SYNC_GPU").is_err();
    let log_dir = std::env::var("IRONTENSOR_LOG_DIR").unwrap_or_else(|_| "logs".to_string());

    // Initialize profiler (enabled when logging is enabled)
    Profiler::init(ProfilerConfig {
        enabled: logging_enabled,
        warmup_steps: 5,
        report_interval: 0,
    });

    // Create directories
    fs::create_dir_all("data").expect("Failed to create data directory");
    fs::create_dir_all("checkpoints").expect("Failed to create checkpoints directory");

    // =========================================================================
    // Load/prepare data (silent)
    // =========================================================================
    let text_path = Path::new("data/tinyshakespeare.txt");
    let text = if text_path.exists() {
        fs::read_to_string(text_path).expect("Failed to read cached file")
    } else {
        let text = download_text(TINY_SHAKESPEARE_URL);
        fs::write(text_path, &text).expect("Failed to cache text");
        text
    };

    // Load/train tokenizer (silent)
    let tokenizer_path = Path::new("data/shakespeare_tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer")
    } else {
        let tokenizer = train_bpe_tokenizer(&text, 2048);
        tokenizer.save(tokenizer_path, false).expect("Failed to save tokenizer");
        tokenizer
    };
    let vocab_size = tokenizer.get_vocab_size(true);

    // Prepare datasets (silent)
    let train_path = Path::new("data/shakespeare_train.bin");
    let val_path = Path::new("data/shakespeare_val.bin");
    let seq_len = 256;

    if !train_path.exists() || !val_path.exists() {
        let encoding = tokenizer.encode(text.as_str(), false).expect("Failed to encode");
        let all_tokens: Vec<u32> = encoding.get_ids().to_vec();
        let split_idx = (all_tokens.len() as f64 * 0.9) as usize;
        TokenDataset::create(train_path, &all_tokens[..split_idx]).expect("Failed to create train dataset");
        TokenDataset::create(val_path, &all_tokens[split_idx..]).expect("Failed to create val dataset");
    }

    let train_dataset = TokenDataset::open(train_path, seq_len).expect("Failed to open train dataset");
    let val_dataset = TokenDataset::open(val_path, seq_len).expect("Failed to open val dataset");

    // Create model
    let mut config = match model_name.as_str() {
        "tiny" => ModelConfig::tiny(),
        "small" => ModelConfig::small(),
        "medium" => ModelConfig::medium(),
        _ => ModelConfig::shakespeare(),
    };
    config.vocab_size = vocab_size;
    let model = GPTModel::new(&config);

    // Training configuration
    let train_config = TrainingConfig {
        learning_rate: 3e-4,
        weight_decay: 0.25,
        beta1: 0.9,
        beta2: 0.99,
        max_grad_norm: 1.0,
        warmup_steps: 50,
        total_steps,
        log_interval: 10,
        save_interval: usize::MAX,
        eval_interval: 50,
        checkpoint_dir: "checkpoints".to_string(),
        use_bf16,
        async_gpu,
        dropout_enabled: true,
        accumulation_steps: 1,
        early_stopping_patience: None,
        early_stopping_min_delta: 0.0,
    };

    // Initialize logger
    Logger::init(LogConfig {
        enabled: logging_enabled,
        log_dir: log_dir.clone(),
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

    // =========================================================================
    // Print summary
    // =========================================================================
    println!(
        r#"
  _                 _____
 (_)               |_   _|
  _ _ __ ___  _ __   | | ___ _ __  ___  ___  _ __
 | | '__/ _ \| '_ \  | |/ _ \ '_ \/ __|/ _ \| '__|
 | | | | (_) | | | | | |  __/ | | \__ \ (_) | |
 |_|_|  \___/|_| |_| \_/\___|_| |_|___/\___/|_|
"#
    );

    println!("Device    {}", device_name);
    println!();
    println!("Environment");
    println!("  IRONTENSOR_MODEL      {}", model_name);
    println!("  IRONTENSOR_STEPS      {}", total_steps);
    println!("  IRONTENSOR_BATCH      {}", batch_size);
    println!("  IRONTENSOR_BF16       {}", if use_bf16 { "1" } else { "-" });
    println!("  IRONTENSOR_LOG        {}", if logging_enabled { "1" } else { "-" });
    println!("  IRONTENSOR_INFERENCE  {}", if inference_enabled { "1" } else { "0" });
    println!();
    println!("Data");
    println!("  Text        {}  ({} chars)", text_path.display(), text.len());
    println!("  Tokenizer   {}  (vocab={})", tokenizer_path.display(), vocab_size);
    println!("  Train       {}  ({} sequences)", train_path.display(), train_dataset.num_sequences());
    println!("  Val         {}  ({} sequences)", val_path.display(), val_dataset.num_sequences());
    println!();
    println!("Model         {} ({:.2}M params)", model_name, model.num_params() as f64 / 1e6);
    println!("  hidden_dim  {}  layers={}  heads={}", config.hidden_dim, config.num_layers, config.num_heads);
    println!("  seq_len     {}  tie_weights={}", config.max_seq_len, config.tie_weights);
    println!();
    println!("Training");
    println!("  lr          {:.0e}  warmup={}  weight_decay={}", train_config.learning_rate, train_config.warmup_steps, train_config.weight_decay);
    println!("  batch       {}  seq_len={}  async_gpu={}", batch_size, seq_len, async_gpu);
    println!("  grad_clip   {:.1}  dropout={}", train_config.max_grad_norm, train_config.dropout_enabled);
    if logging_enabled {
        println!();
        println!("Logger    {}", Logger::log_path().map(|p| p.display().to_string()).unwrap_or_else(|| log_dir));
    }
    println!();
    println!("{}", "=".repeat(60));
    println!();

    // =========================================================================
    // Training
    // =========================================================================
    let mut trainer = Trainer::new(&config, &train_config);
    let mut callback = LoggingCallback::new(batch_size, seq_len);

    Logger::start_training();
    let training_start = Instant::now();

    let num_epochs = (train_config.total_steps / (train_dataset.num_sequences() / batch_size).max(1)).max(1) + 1;
    trainer.train(&train_dataset, Some(&val_dataset), batch_size, num_epochs, &mut callback);

    let training_time = training_start.elapsed();
    println!();
    println!("Training complete in {:.1}s", training_time.as_secs_f32());

    // Finalize training log
    let profiler_report = if logging_enabled {
        Some(Profiler::report().to_record())
    } else {
        None
    };

    if logging_enabled {
        Logger::finalize_training(
            trainer.step,
            callback.last_loss,
            callback.best_val_loss,
            trainer.epoch,
            profiler_report,
        );
    }

    // =========================================================================
    // Inference (optional)
    // =========================================================================
    if inference_enabled {
        println!();
        println!("{}", "=".repeat(60));
        println!();
        let inference_start = Instant::now();

        let prompts = vec![
            "ROMEO:",
            "To be or not",
            "The king",
            "JULIET:\nO Romeo,",
        ];

        let generator = TextGenerator::new(GeneratorConfig {
            temperature: 0.8,
            max_tokens: 100,
            ..Default::default()
        });

        let mut total_generated_tokens = 0usize;

        for prompt in prompts {
            print!("Prompt: \"{}\"\nGenerated: ", prompt);
            std::io::stdout().flush().unwrap();

            let prompt_tokens = tokenizer.encode(prompt, false).map(|e| e.len()).unwrap_or(0);
            let generated = generator.generate_streaming(&mut trainer.model, &tokenizer, prompt, |text| {
                print!("{}", text);
                std::io::stdout().flush().unwrap();
            });
            let total_tokens = tokenizer.encode(generated.as_str(), false).map(|e| e.len()).unwrap_or(0);
            total_generated_tokens += total_tokens.saturating_sub(prompt_tokens);

            println!("\n{}", "-".repeat(40));
        }

        let inference_time = inference_start.elapsed();
        let tokens_per_sec = total_generated_tokens as f32 / inference_time.as_secs_f32();
        println!(
            "Inference complete in {:.1}s ({} tokens, {:.1} tok/s)",
            inference_time.as_secs_f32(),
            total_generated_tokens,
            tokens_per_sec
        );
    }

    // Shutdown logger
    Logger::shutdown();

    let total_duration = start_time.elapsed();
    println!();
    println!("Done! Total time: {:.1}s", total_duration.as_secs_f32());
}

/// Callback for training progress.
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

        println!(
            "Step {:>5} | loss={:.4} | grad={:.4} | lr={:.2e} | {:.0} tok/s",
            metrics.step,
            metrics.loss,
            metrics.grad_norm,
            metrics.learning_rate,
            metrics.tokens_per_sec
        );

        if Logger::is_enabled() {
            let elapsed = self.step_start.elapsed().as_secs_f32();
            let record = TrainStepRecord::new(
                metrics.step,
                0,
                metrics.loss,
                metrics.grad_norm,
                metrics.learning_rate,
                metrics.tokens_per_sec,
                elapsed * 1000.0,
                self.batch_size,
                self.seq_len,
            );
            Logger::log_train_step(&record);
        }

        self.step_start = Instant::now();
    }

    fn on_eval(&mut self, step: usize, val_loss: f32) {
        println!("Step {:>5} | val_loss={:.4}", step, val_loss);

        if self.best_val_loss.is_none() || val_loss < self.best_val_loss.unwrap() {
            self.best_val_loss = Some(val_loss);
        }
    }

    fn on_save(&mut self, step: usize, path: &str) {
        println!("Step {:>5} | checkpoint: {}", step, path);
    }
}

/// Download text from a URL.
fn download_text(url: &str) -> String {
    use std::process::Command;
    let output = Command::new("curl")
        .args(["-s", url])
        .output()
        .expect("Failed to execute curl");
    String::from_utf8(output.stdout).expect("Invalid UTF-8 in response")
}

/// Train a BPE tokenizer on the given text.
fn train_bpe_tokenizer(text: &str, vocab_size: usize) -> Tokenizer {
    use tokenizers::models::TrainerWrapper;

    let temp_path = std::env::temp_dir().join("bpe_train_text.txt");
    fs::write(&temp_path, text).expect("Failed to write temp file");

    let trainer = BpeTrainerBuilder::new()
        .vocab_size(vocab_size)
        .min_frequency(2)
        .special_tokens(vec![
            tokenizers::AddedToken::from("<|endoftext|>", true),
            tokenizers::AddedToken::from("<|pad|>", true),
        ])
        .build();
    let mut trainer_wrapper = TrainerWrapper::BpeTrainer(trainer);

    let mut tokenizer = Tokenizer::new(BPE::default());
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::ByteLevel(ByteLevel::default())));
    tokenizer
        .train_from_files(&mut trainer_wrapper, vec![temp_path.to_str().unwrap().to_string()])
        .expect("Failed to train tokenizer");
    tokenizer.with_decoder(Some(DecoderWrapper::ByteLevel(
        tokenizers::decoders::byte_level::ByteLevel::default(),
    )));

    fs::remove_file(&temp_path).ok();
    tokenizer
}
