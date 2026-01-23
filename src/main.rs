use irontensor::{
    GPTModel, ModelConfig, TokenDataset,
    CosineAnnealingLR, LRScheduler, Trainer, TrainingConfig,
    save_model_weights, Checkpoint,
    MetalContext,
    Profiler, ProfilerConfig,
};
use objc2_metal::MTLDevice;
use rand::Rng;
use std::fs;
use std::path::Path;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{DecoderWrapper, PreTokenizerWrapper, Tokenizer};

const TINY_SHAKESPEARE_URL: &str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

fn main() {
    // Initialize the Metal context
    let ctx = MetalContext::global();
    println!("IronTensor - GPT Training on Tiny Shakespeare");
    println!("==============================================");
    println!("Device: {}\n", ctx.device().name());

    // Initialize profiler (opt-in, enable via environment variable)
    let profiling_enabled = std::env::var("IRONTENSOR_PROFILE").is_ok();
    Profiler::init(ProfilerConfig {
        enabled: profiling_enabled,
        warmup_steps: 5,
        report_interval: 0, // Print at end only
    });
    if profiling_enabled {
        println!("Profiling enabled (set IRONTENSOR_PROFILE=0 to disable)\n");
    }

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
    let mut config = ModelConfig::shakespeare();
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
    let batch_size = 16; // Batch size

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
        weight_decay: 0.1,
        beta1: 0.9,
        beta2: 0.99,
        max_grad_norm: 1.0,
        warmup_steps: 50,
        total_steps: 100,  // Reduced for testing
        log_interval: 10,  // More frequent logging
        save_interval: 100,
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

    // Create trainer
    let mut trainer = Trainer::new(&config, &train_config);

    // Training loop
    let num_batches = train_dataset.num_sequences() / batch_size;
    let steps_per_epoch = num_batches.max(1);

    println!("Starting training...");
    println!("Steps per epoch: {}", steps_per_epoch);
    println!("{}", "-".repeat(60));

    let scheduler = CosineAnnealingLR::with_warmup(
        train_config.learning_rate,
        train_config.warmup_steps,
        train_config.total_steps,
    );

    let mut running_loss = 0.0;
    let mut loss_count = 0;

    for step in 0..train_config.total_steps {
        // Get batch
        let batch_idx = (step % steps_per_epoch) * batch_size;
        let (input_ids, target_ids) = get_batch(&train_dataset, batch_idx, batch_size, seq_len);

        // Training step (handles forward, backward, optimizer update internally)
        let (loss, _grad_norm) = trainer.train_step(&input_ids, &target_ids, batch_size, seq_len);

        let lr = scheduler.get_lr(step);

        running_loss += loss;
        loss_count += 1;

        // Logging
        if (step + 1) % train_config.log_interval == 0 {
            let avg_loss = running_loss / loss_count as f32;
            println!(
                "Step {:5}/{} | Loss: {:.4} | LR: {:.2e}",
                step + 1,
                train_config.total_steps,
                avg_loss,
                lr
            );
            running_loss = 0.0;
            loss_count = 0;
        }

        // Evaluation
        if (step + 1) % train_config.eval_interval == 0 {
            let val_loss = evaluate(&trainer, &val_dataset, batch_size, seq_len, 10);
            println!("  Validation Loss: {:.4}", val_loss);
        }

        // Checkpointing
        if (step + 1) % train_config.save_interval == 0 {
            let checkpoint_path = format!("{}/step_{}.bin", train_config.checkpoint_dir, step + 1);
            let checkpoint = Checkpoint {
                config: config.clone(),
                step: step + 1,
                epoch: step / steps_per_epoch,
                best_val_loss: f32::INFINITY,
                learning_rate: lr,
            };
            save_model_weights(Path::new(&checkpoint_path), &trainer.model, &checkpoint)
                .expect("Failed to save checkpoint");
            println!("  Saved checkpoint: {}", checkpoint_path);
        }
    }

    println!("{}", "-".repeat(60));
    println!("Training complete!\n");

    // Print profiling report if enabled
    if profiling_enabled {
        let report = Profiler::report();
        report.print();
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

    println!("\nDone!");
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

/// Get a batch of training data
fn get_batch(
    dataset: &TokenDataset,
    start_idx: usize,
    batch_size: usize,
    seq_len: usize,
) -> (Vec<u32>, Vec<u32>) {
    let mut input_ids = Vec::with_capacity(batch_size * seq_len);
    let mut target_ids = Vec::with_capacity(batch_size * seq_len);

    for b in 0..batch_size {
        let idx = (start_idx + b) % dataset.num_sequences();
        let (inp, tgt) = dataset.get_batch(idx);
        input_ids.extend_from_slice(&inp[..seq_len.min(inp.len())]);
        target_ids.extend_from_slice(&tgt[..seq_len.min(tgt.len())]);

        // Pad if necessary
        while input_ids.len() < (b + 1) * seq_len {
            input_ids.push(0);
            target_ids.push(0);
        }
    }

    (input_ids, target_ids)
}

/// Evaluate on validation set
fn evaluate(
    trainer: &Trainer,
    val_dataset: &TokenDataset,
    batch_size: usize,
    seq_len: usize,
    num_batches: usize,
) -> f32 {
    let mut total_loss = 0.0;
    let num_batches = num_batches.min(val_dataset.num_sequences() / batch_size).max(1);

    for i in 0..num_batches {
        let (input_ids, target_ids) = get_batch(val_dataset, i * batch_size, batch_size, seq_len);
        let loss = trainer.compute_loss(&input_ids, &target_ids, batch_size, seq_len);
        total_loss += loss;
    }

    total_loss / num_batches as f32
}

/// Generate text from a prompt
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

    let max_len = model.config.max_seq_len;

    // Generate tokens one at a time
    for _ in 0..max_tokens {
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

        // Stop at end of text token
        if next_token == 0 {
            break;
        }
    }

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
