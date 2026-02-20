//! IronTensor benchmark binary for comparison with PyTorch.
//!
//! Runs training and inference benchmarks on synthetic data and outputs
//! JSON results to `benchmarks/results/irontensor.json`.
//!
//! Usage:
//!     cargo run --release --bin benchmark

use irontensor::{
    CommandBatch, GPTModel, InferenceTimer, MetalContext, ModelConfig, Profiler, ProfilerConfig,
    TokenDataset, Trainer, TrainingConfig,
};
use irontensor::train::{TrainCallback, TrainMetrics};

use objc2_metal::MTLDevice;
use rand::Rng;
use serde_json::json;
use std::fs;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const WARMUP_STEPS: usize = 5;
const TIMED_STEPS: usize = 50;
const TOTAL_STEPS: usize = WARMUP_STEPS + TIMED_STEPS;
const BATCH_SIZE: usize = 16;
const SEQ_LEN: usize = 256;
const INFERENCE_TOKENS: usize = 100;
const VOCAB_SIZE: usize = 32000;

// ---------------------------------------------------------------------------
// Benchmark callback: collects per-step timing
// ---------------------------------------------------------------------------

struct BenchmarkCallback {
    batch_size: usize,
    seq_len: usize,
    step_start: Instant,
    warmup_steps: usize,
    step_times_ms: Vec<f64>,
    tokens_per_sec: Vec<f64>,
    losses: Vec<f32>,
    last_loss: f32,
}

impl BenchmarkCallback {
    fn new(batch_size: usize, seq_len: usize, warmup_steps: usize) -> Self {
        Self {
            batch_size,
            seq_len,
            step_start: Instant::now(),
            warmup_steps,
            step_times_ms: Vec::new(),
            tokens_per_sec: Vec::new(),
            losses: Vec::new(),
            last_loss: 0.0,
        }
    }
}

impl TrainCallback for BenchmarkCallback {
    fn on_step(&mut self, metrics: &TrainMetrics) {
        let elapsed_ms = self.step_start.elapsed().as_secs_f64() * 1000.0;
        self.last_loss = metrics.loss;

        let tok_per_sec = (self.batch_size * self.seq_len) as f64 / (elapsed_ms / 1000.0);

        // Only record after warmup
        if metrics.step > self.warmup_steps {
            self.step_times_ms.push(elapsed_ms);
            self.tokens_per_sec.push(tok_per_sec);
            self.losses.push(metrics.loss);
        }

        println!(
            "  Step {:>4}/{} | loss={:.4} | grad={:.4} | lr={:.2e} | {:.0} tok/s | {:.1}ms{}",
            metrics.step,
            TOTAL_STEPS,
            metrics.loss,
            metrics.grad_norm,
            metrics.learning_rate,
            tok_per_sec,
            elapsed_ms,
            if metrics.step <= self.warmup_steps {
                " (warmup)"
            } else {
                ""
            }
        );

        self.step_start = Instant::now();
    }

    fn on_eval(&mut self, _step: usize, _val_loss: f32) {}
    fn on_save(&mut self, _step: usize, _path: &str) {}
}

// ---------------------------------------------------------------------------
// Training benchmark
// ---------------------------------------------------------------------------

fn benchmark_training(config: &ModelConfig) -> serde_json::Value {
    println!("Creating synthetic dataset...");

    // Create synthetic random token data
    let dataset_path = "benchmarks/results/_benchmark_train.bin";
    let num_tokens = BATCH_SIZE * SEQ_LEN * (TOTAL_STEPS + 2) + 1;
    let mut rng = rand::rng();
    let tokens: Vec<u32> = (0..num_tokens)
        .map(|_| rng.random_range(0..VOCAB_SIZE as u32))
        .collect();
    TokenDataset::create(dataset_path, &tokens).expect("Failed to create dataset");
    let dataset = TokenDataset::open(dataset_path, SEQ_LEN).expect("Failed to open dataset");

    println!(
        "Dataset: {} tokens, {} sequences",
        dataset.num_tokens(),
        dataset.num_sequences()
    );

    // Training config
    let train_config = TrainingConfig {
        learning_rate: 3e-4,
        weight_decay: 0.1,
        beta1: 0.9,
        beta2: 0.99,
        max_grad_norm: 1.0,
        warmup_steps: WARMUP_STEPS,
        total_steps: TOTAL_STEPS,
        log_interval: 1, // Log every step for timing
        save_interval: usize::MAX,
        eval_interval: usize::MAX,
        checkpoint_dir: "benchmarks/results/_benchmark_ckpt".to_string(),
        use_bf16: false,
        async_gpu: true,
        dropout_enabled: false, // Disable for fair comparison
        accumulation_steps: 1,
        early_stopping_patience: None,
        early_stopping_min_delta: 0.0,
        checkpoint_config: irontensor::train::CheckpointConfig::default(),
        ..Default::default()
    };

    // Create trainer
    let mut trainer = Trainer::new(config, &train_config);
    let mut callback = BenchmarkCallback::new(BATCH_SIZE, SEQ_LEN, WARMUP_STEPS);

    println!(
        "Model: {:.2}M params, hidden={}, layers={}, heads={}",
        config.num_params() as f64 / 1e6,
        config.hidden_dim,
        config.num_layers,
        config.num_heads,
    );
    println!();

    // Run training
    let num_epochs =
        (TOTAL_STEPS / (dataset.num_sequences() / BATCH_SIZE).max(1)).max(1) + 1;
    trainer.train(&dataset, None, BATCH_SIZE, num_epochs, &mut callback);

    // Ensure all GPU work is complete
    CommandBatch::wait_for_completion();
    CommandBatch::end_async();

    // Memory measurement
    let memory_bytes = irontensor::gpu_memory_allocated();

    // Compute stats
    let step_times = &callback.step_times_ms;
    let avg_step_ms = if step_times.is_empty() {
        0.0
    } else {
        step_times.iter().sum::<f64>() / step_times.len() as f64
    };

    let mut sorted_times = step_times.clone();
    sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_step_ms = if sorted_times.is_empty() {
        0.0
    } else {
        sorted_times[sorted_times.len() / 2]
    };

    let avg_tokens_per_sec = if avg_step_ms > 0.0 {
        (BATCH_SIZE * SEQ_LEN) as f64 / (avg_step_ms / 1000.0)
    } else {
        0.0
    };

    // Clean up temp files
    fs::remove_file(dataset_path).ok();
    fs::remove_dir_all("benchmarks/results/_benchmark_ckpt").ok();

    json!({
        "avg_step_time_ms": avg_step_ms,
        "median_step_time_ms": median_step_ms,
        "min_step_time_ms": sorted_times.first().copied().unwrap_or(0.0),
        "max_step_time_ms": sorted_times.last().copied().unwrap_or(0.0),
        "avg_tokens_per_sec": avg_tokens_per_sec,
        "final_loss": callback.losses.last().copied().unwrap_or(0.0),
        "initial_loss": callback.losses.first().copied().unwrap_or(0.0),
        "peak_memory_bytes": memory_bytes,
        "current_memory_bytes": memory_bytes,
        "timed_steps": step_times.len(),
        "warmup_steps": WARMUP_STEPS,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
    })
}

// ---------------------------------------------------------------------------
// Inference benchmark
// ---------------------------------------------------------------------------

fn benchmark_inference_single(
    model: &mut GPTModel,
    prompt_len: usize,
    max_tokens: usize,
) -> serde_json::Value {
    let vocab_size = model.config.vocab_size;
    let max_seq_len = model.config.max_seq_len;

    // Generate random prompt
    let mut rng = rand::rng();
    let mut tokens: Vec<u32> = (0..prompt_len)
        .map(|_| rng.random_range(0..vocab_size as u32))
        .collect();

    model.set_training(false);

    let mut timer = InferenceTimer::new(prompt_len, 0.0);
    let start = Instant::now();

    for i in 0..max_tokens {
        // Truncate context if needed
        let context = if tokens.len() > max_seq_len {
            &tokens[tokens.len() - max_seq_len..]
        } else {
            &tokens[..]
        };

        // Forward pass (greedy decoding)
        let logits = model.forward(context, 1, context.len(), 0);
        let logits_slice = logits.as_f32_slice();
        let last_pos = context.len() - 1;
        let last_logits = &logits_slice[last_pos * vocab_size..(last_pos + 1) * vocab_size];

        // Argmax
        let next_token = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();

        tokens.push(next_token);

        if i == 0 {
            timer.mark_first_token();
        } else {
            timer.token_generated();
        }
    }

    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let record = timer.finish_no_log();

    model.set_training(true);

    let tokens_per_sec = max_tokens as f64 / (total_ms / 1000.0);
    let inter_token_ms = if max_tokens > 1 {
        (total_ms - record.time_to_first_token_ms as f64) / (max_tokens - 1) as f64
    } else {
        0.0
    };

    println!(
        "  prompt_len={}: TTFT={:.1}ms, {:.1} tok/s, total={:.0}ms",
        prompt_len, record.time_to_first_token_ms, tokens_per_sec, total_ms
    );

    json!({
        "prompt_length": prompt_len,
        "generated_tokens": max_tokens,
        "ttft_ms": record.time_to_first_token_ms,
        "total_time_ms": total_ms,
        "tokens_per_sec": tokens_per_sec,
        "inter_token_latency_ms": inter_token_ms,
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // Initialize Metal context
    let ctx = MetalContext::global();
    let device_name = ctx.device().name().to_string();

    // Initialize profiler (disabled for benchmarking)
    Profiler::init(ProfilerConfig {
        enabled: false,
        warmup_steps: 0,
        report_interval: 0,
    });

    println!();
    println!("IronTensor Benchmark");
    println!("====================");
    println!("Device: {}", device_name);
    println!();

    // Create output directory
    fs::create_dir_all("benchmarks/results").expect("Failed to create results directory");

    // Model config (TINY)
    let model_name = "tiny";
    let config = ModelConfig::tiny();

    // ---- Training benchmark ----
    println!("{}", "=".repeat(60));
    println!(
        "Training benchmark: {} steps (warmup={}, timed={})",
        TOTAL_STEPS, WARMUP_STEPS, TIMED_STEPS
    );
    println!("  batch_size={}, seq_len={}", BATCH_SIZE, SEQ_LEN);
    println!("{}", "=".repeat(60));

    let training_results = benchmark_training(&config);

    println!();
    println!("Training summary:");
    println!(
        "  Avg step time:  {:.1}ms",
        training_results["avg_step_time_ms"].as_f64().unwrap_or(0.0)
    );
    println!(
        "  Avg tokens/sec: {:.0}",
        training_results["avg_tokens_per_sec"].as_f64().unwrap_or(0.0)
    );
    println!(
        "  Peak memory:    {:.0}MB",
        training_results["peak_memory_bytes"].as_f64().unwrap_or(0.0) / 1e6
    );

    // ---- Inference benchmark ----
    println!();
    println!("{}", "=".repeat(60));
    println!(
        "Inference benchmark: {} tokens, no KV cache",
        INFERENCE_TOKENS
    );
    println!("{}", "=".repeat(60));

    // Create a fresh model for inference (same weights aren't important for benchmarking)
    let mut model = GPTModel::new(&config);
    model.set_training(false);

    let prompt_lengths = vec![5, 20];
    let mut inference_results = Vec::new();
    for &prompt_len in &prompt_lengths {
        let result = benchmark_inference_single(&mut model, prompt_len, INFERENCE_TOKENS);
        inference_results.push(result);
    }

    // ---- Save results ----
    let output = json!({
        "framework": "irontensor",
        "model": model_name,
        "model_config": {
            "vocab_size": config.vocab_size,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "num_kv_heads": config.num_kv_heads,
            "intermediate_dim": config.intermediate_dim,
            "max_seq_len": config.max_seq_len,
        },
        "model_params": config.num_params(),
        "device": device_name,
        "precision": "fp32",
        "training": training_results,
        "inference": inference_results,
    });

    let output_path = "benchmarks/results/irontensor.json";
    let json_str = serde_json::to_string_pretty(&output).expect("Failed to serialize JSON");
    fs::write(output_path, &json_str).expect("Failed to write results");

    println!();
    println!("Results saved to {}", output_path);
    println!("Done!");
}
