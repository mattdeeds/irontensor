use super::*;
use crate::ops::{cross_entropy_fused, matmul};
use crate::precision::Precision;

#[test]
fn test_trainer_creation() {
    let model_config = ModelConfig {
        vocab_size: 100,
        hidden_dim: 32,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 2,
        intermediate_dim: 64,
        norm_eps: 1e-5,
        rope_base: 10000.0,
        max_seq_len: 128,
        tie_weights: true,
        precision: Precision::FP32,
        embed_dropout: 0.0,
        attn_dropout: 0.1,
        ffn_dropout: 0.1,
    };

    let train_config = TrainingConfig::default();
    let trainer = Trainer::new(&model_config, &train_config);

    assert_eq!(trainer.step, 0);
    assert_eq!(trainer.epoch, 0);
}

#[test]
fn test_compute_loss() {
    let model_config = ModelConfig {
        vocab_size: 100,
        hidden_dim: 32,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        intermediate_dim: 64,
        norm_eps: 1e-5,
        rope_base: 10000.0,
        max_seq_len: 128,
        tie_weights: true,
        precision: Precision::FP32,
        embed_dropout: 0.0,
        attn_dropout: 0.1,
        ffn_dropout: 0.1,
    };

    let train_config = TrainingConfig::default();
    let trainer = Trainer::new(&model_config, &train_config);

    let batch_size = 2;
    let seq_len = 8;
    let input_ids: Vec<u32> = (0..batch_size * seq_len).map(|i| (i % 100) as u32).collect();
    let target_ids: Vec<u32> = (0..batch_size * seq_len)
        .map(|i| ((i + 1) % 100) as u32)
        .collect();

    let loss = trainer.compute_loss(&input_ids, &target_ids, batch_size, seq_len);

    // Loss should be positive
    assert!(loss > 0.0);
    // Loss should be reasonable for random initialization
    assert!(loss < 10.0);
}

#[test]
fn test_train_step_produces_gradients() {
    let model_config = ModelConfig {
        vocab_size: 100,
        hidden_dim: 32,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        intermediate_dim: 64,
        norm_eps: 1e-5,
        rope_base: 10000.0,
        max_seq_len: 128,
        tie_weights: true,
        precision: Precision::FP32,
        embed_dropout: 0.0,
        attn_dropout: 0.0, // Disable dropout for determinism
        ffn_dropout: 0.0,
    };

    let train_config = TrainingConfig {
        dropout_enabled: false, // Disable dropout
        async_gpu: false,       // Use synchronous mode
        ..Default::default()
    };

    let trainer = Trainer::new(&model_config, &train_config);

    let batch_size = 2;
    let seq_len = 8;
    let vocab_size = model_config.vocab_size;
    let n = batch_size * seq_len;

    let input_ids: Vec<u32> = (0..n).map(|i| (i % 100) as u32).collect();
    let target_ids: Vec<u32> = (0..n).map(|i| ((i + 1) % 100) as u32).collect();

    // Step 1: Forward pass (model.forward reads GPU data, so don't use batching)
    // NOTE: model.forward() has internal as_f32_slice() calls which require sync
    let logits = trainer.model.forward(&input_ids, batch_size, seq_len, 0);

    let logits_data = logits.as_f32_slice();
    let logits_nonzero = logits_data.iter().any(|&x| x != 0.0);
    println!("Logits shape: {:?}, non-zero: {}", logits.shape(), logits_nonzero);
    assert!(logits_nonzero, "Logits should be non-zero");

    // Step 2: Cross-entropy gradient (cross_entropy_fused has internal syncs)
    let logits_2d = logits.view(&[n, vocab_size]);
    let (loss, _, grad_logits) = cross_entropy_fused(&logits_2d, &target_ids);

    let grad_logits_data = grad_logits.as_f32_slice();
    let grad_logits_nonzero = grad_logits_data.iter().any(|&x| x != 0.0);
    let grad_logits_sum: f32 = grad_logits_data.iter().map(|x| x.abs()).sum();
    println!(
        "Loss: {}, grad_logits shape: {:?}, non-zero: {}, abs_sum: {}",
        loss,
        grad_logits.shape(),
        grad_logits_nonzero,
        grad_logits_sum
    );
    assert!(grad_logits_nonzero, "grad_logits should be non-zero");

    // Step 3: Test a simple matmul with grad_logits (matmul syncs internally)
    let embed_fp32 = ensure_fp32(&trainer.model.embed_tokens);
    println!("embed_fp32 shape: {:?}", embed_fp32.shape());

    let grad_hidden = matmul(&grad_logits, &embed_fp32).unwrap();

    let grad_hidden_data = grad_hidden.as_f32_slice();
    let grad_hidden_nonzero = grad_hidden_data.iter().any(|&x| x != 0.0);
    let grad_hidden_sum: f32 = grad_hidden_data.iter().map(|x| x.abs()).sum();
    println!(
        "grad_hidden shape: {:?}, non-zero: {}, abs_sum: {}",
        grad_hidden.shape(),
        grad_hidden_nonzero,
        grad_hidden_sum
    );
    assert!(grad_hidden_nonzero, "grad_hidden should be non-zero");

    // Loss should be positive
    assert!(loss > 0.0, "Loss should be positive, got {}", loss);
}

#[test]
fn test_full_train_step_gradients() {
    use crate::command_batch::CommandBatch;

    // Test that trainer forward_with_cache + backward produces non-zero gradients
    let model_config = ModelConfig {
        vocab_size: 100,
        hidden_dim: 32,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        intermediate_dim: 64,
        norm_eps: 1e-5,
        rope_base: 10000.0,
        max_seq_len: 128,
        tie_weights: true,
        precision: Precision::FP32,
        embed_dropout: 0.0,
        attn_dropout: 0.0, // Disable dropout for determinism
        ffn_dropout: 0.0,
    };

    let train_config = TrainingConfig {
        dropout_enabled: false, // Disable dropout
        async_gpu: false,       // Use synchronous mode
        ..Default::default()
    };

    let trainer = Trainer::new(&model_config, &train_config);

    let batch_size = 2;
    let seq_len = 8;
    let vocab_size = model_config.vocab_size;
    let n = batch_size * seq_len;
    let input_ids: Vec<u32> = (0..n).map(|i| (i % 100) as u32).collect();
    let target_ids: Vec<u32> = (0..n).map(|i| ((i + 1) % 100) as u32).collect();

    // Test 1: Forward pass WITHOUT CommandBatch (immediate mode)
    println!("\n=== Test 1: Forward without batching ===");
    let cache = trainer.forward_with_cache(&input_ids, batch_size, seq_len);

    let fh_data = cache.final_hidden.as_f32_slice();
    let fh_nonzero = fh_data.iter().any(|&x| x != 0.0);
    let fh_sum: f32 = fh_data.iter().map(|x| x.abs()).sum();
    println!(
        "Final hidden shape: {:?}, non-zero: {}, abs_sum: {}",
        cache.final_hidden.shape(),
        fh_nonzero,
        fh_sum
    );
    assert!(fh_nonzero, "Final hidden should be non-zero (immediate mode)");

    // Test 2: Forward pass WITH CommandBatch (batched mode)
    println!("\n=== Test 2: Forward with batching ===");
    CommandBatch::begin();
    let cache2 = trainer.forward_with_cache(&input_ids, batch_size, seq_len);
    CommandBatch::sync();

    let fh_data2 = cache2.final_hidden.as_f32_slice();
    let fh_nonzero2 = fh_data2.iter().any(|&x| x != 0.0);
    let fh_sum2: f32 = fh_data2.iter().map(|x| x.abs()).sum();
    println!(
        "Final hidden shape: {:?}, non-zero: {}, abs_sum: {}",
        cache2.final_hidden.shape(),
        fh_nonzero2,
        fh_sum2
    );
    CommandBatch::end();
    assert!(fh_nonzero2, "Final hidden should be non-zero (batched mode)");

    // Test 3: Logits computation
    println!("\n=== Test 3: Logits computation ===");
    let logits = trainer.compute_logits_from_hidden(&cache.final_hidden);
    let logits_data = logits.as_f32_slice();
    let logits_nonzero = logits_data.iter().any(|&x| x != 0.0);
    let logits_sum: f32 = logits_data.iter().map(|x| x.abs()).sum();
    println!(
        "Logits shape: {:?}, non-zero: {}, abs_sum: {}",
        logits.shape(),
        logits_nonzero,
        logits_sum
    );
    assert!(logits_nonzero, "Logits should be non-zero");

    // Test 4: Cross-entropy gradient
    println!("\n=== Test 4: Cross-entropy gradient ===");
    let logits_2d = logits.view(&[n, vocab_size]);
    let (loss, _, grad_logits) = cross_entropy_fused(&logits_2d, &target_ids);
    let gl_data = grad_logits.as_f32_slice();
    let gl_nonzero = gl_data.iter().any(|&x| x != 0.0);
    let gl_sum: f32 = gl_data.iter().map(|x| x.abs()).sum();
    println!(
        "Loss: {}, grad_logits shape: {:?}, non-zero: {}, abs_sum: {}",
        loss,
        grad_logits.shape(),
        gl_nonzero,
        gl_sum
    );
    assert!(gl_nonzero, "grad_logits should be non-zero");

    // Test 5: Backward matmul (grad_embed_out)
    println!("\n=== Test 5: Backward matmul ===");
    let grad_embed_out = crate::ops::matmul_mps_tn(&grad_logits, &cache.final_hidden);
    let geo_data = grad_embed_out.as_f32_slice();
    let geo_nonzero = geo_data.iter().any(|&x| x != 0.0);
    let geo_sum: f32 = geo_data.iter().map(|x| x.abs()).sum();
    println!(
        "grad_embed_out shape: {:?}, non-zero: {}, abs_sum: {}",
        grad_embed_out.shape(),
        geo_nonzero,
        geo_sum
    );
    assert!(geo_nonzero, "grad_embed_out should be non-zero");

    println!("\n=== All checks passed! ===");
}

#[test]
fn test_gradient_accumulation() {
    // Test that gradient accumulation works correctly
    let model_config = ModelConfig {
        vocab_size: 100,
        hidden_dim: 32,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        intermediate_dim: 64,
        norm_eps: 1e-5,
        rope_base: 10000.0,
        max_seq_len: 128,
        tie_weights: true,
        precision: Precision::FP32,
        embed_dropout: 0.0,
        attn_dropout: 0.0,
        ffn_dropout: 0.0,
    };

    let train_config = TrainingConfig {
        dropout_enabled: false,
        async_gpu: false,
        accumulation_steps: 2, // Accumulate 2 micro-batches
        ..Default::default()
    };

    let mut trainer = Trainer::new(&model_config, &train_config);

    let batch_size = 2;
    let seq_len = 8;
    let n = batch_size * seq_len;
    let input_ids: Vec<u32> = (0..n).map(|i| (i % 100) as u32).collect();
    let target_ids: Vec<u32> = (0..n).map(|i| ((i + 1) % 100) as u32).collect();

    // First micro-batch: should NOT step optimizer
    let (loss1, grad_norm1) = trainer.train_step(&input_ids, &target_ids, batch_size, seq_len);
    assert!(loss1 > 0.0, "Loss should be positive");
    assert_eq!(
        grad_norm1, 0.0,
        "Grad norm should be 0 on first micro-batch (no optimizer step)"
    );
    assert_eq!(trainer.step, 0, "Step should not increment on first micro-batch");

    // Second micro-batch: should step optimizer
    let (loss2, grad_norm2) = trainer.train_step(&input_ids, &target_ids, batch_size, seq_len);
    assert!(loss2 > 0.0, "Loss should be positive");
    assert!(grad_norm2 > 0.0, "Grad norm should be > 0 on optimizer step");
    assert_eq!(trainer.step, 1, "Step should increment after accumulation cycle");

    // Third micro-batch: again should NOT step optimizer
    let (loss3, grad_norm3) = trainer.train_step(&input_ids, &target_ids, batch_size, seq_len);
    assert!(loss3 > 0.0, "Loss should be positive");
    assert_eq!(
        grad_norm3, 0.0,
        "Grad norm should be 0 on first micro-batch of new cycle"
    );
    assert_eq!(trainer.step, 1, "Step should not increment on first micro-batch");

    println!("Gradient accumulation test passed!");
    println!("  loss1={}, grad_norm1={}", loss1, grad_norm1);
    println!("  loss2={}, grad_norm2={}", loss2, grad_norm2);
    println!("  loss3={}, grad_norm3={}", loss3, grad_norm3);
}

#[test]
fn test_activation_checkpointing() {
    use crate::train::CheckpointConfig;

    // Test that training with activation checkpointing produces similar results
    // to training without checkpointing (gradient values should be identical)
    let model_config = ModelConfig {
        vocab_size: 100,
        hidden_dim: 32,
        num_layers: 2, // Need multiple layers to test checkpointing
        num_heads: 2,
        num_kv_heads: 2,
        intermediate_dim: 64,
        norm_eps: 1e-5,
        rope_base: 10000.0,
        max_seq_len: 128,
        tie_weights: true,
        precision: Precision::FP32,
        embed_dropout: 0.0,
        attn_dropout: 0.0, // Disable dropout for deterministic comparison
        ffn_dropout: 0.0,
    };

    // Train without checkpointing
    let train_config_no_ckpt = TrainingConfig {
        dropout_enabled: false,
        async_gpu: false,
        checkpoint_config: CheckpointConfig::default(), // Disabled
        ..Default::default()
    };

    let mut trainer_no_ckpt = Trainer::new(&model_config, &train_config_no_ckpt);

    let batch_size = 2;
    let seq_len = 8;
    let n = batch_size * seq_len;
    let input_ids: Vec<u32> = (0..n).map(|i| (i % 100) as u32).collect();
    let target_ids: Vec<u32> = (0..n).map(|i| ((i + 1) % 100) as u32).collect();

    // Get initial weights for comparison
    let initial_embed = trainer_no_ckpt.model.embed_tokens.as_f32_slice().to_vec();

    let (loss_no_ckpt, grad_norm_no_ckpt) =
        trainer_no_ckpt.train_step(&input_ids, &target_ids, batch_size, seq_len);

    // Get weights after training without checkpointing
    let weights_no_ckpt = trainer_no_ckpt.model.embed_tokens.as_f32_slice().to_vec();

    // Train with checkpointing (checkpoint every layer)
    let train_config_with_ckpt = TrainingConfig {
        dropout_enabled: false,
        async_gpu: false,
        checkpoint_config: CheckpointConfig::enabled(), // All layers checkpointed
        ..Default::default()
    };

    let mut trainer_with_ckpt = Trainer::new(&model_config, &train_config_with_ckpt);

    // Verify initial weights match
    let initial_embed_ckpt = trainer_with_ckpt.model.embed_tokens.as_f32_slice().to_vec();
    assert_eq!(initial_embed, initial_embed_ckpt, "Initial weights should match");

    let (loss_with_ckpt, grad_norm_with_ckpt) =
        trainer_with_ckpt.train_step(&input_ids, &target_ids, batch_size, seq_len);

    // Get weights after training with checkpointing
    let weights_with_ckpt = trainer_with_ckpt.model.embed_tokens.as_f32_slice().to_vec();

    // Compare losses
    let loss_diff = (loss_no_ckpt - loss_with_ckpt).abs();
    println!("Loss without checkpointing: {}", loss_no_ckpt);
    println!("Loss with checkpointing: {}", loss_with_ckpt);
    println!("Loss difference: {}", loss_diff);
    assert!(loss_diff < 1e-4, "Losses should be very close, got diff {}", loss_diff);

    // Compare gradient norms
    let grad_norm_diff = (grad_norm_no_ckpt - grad_norm_with_ckpt).abs();
    println!("Grad norm without checkpointing: {}", grad_norm_no_ckpt);
    println!("Grad norm with checkpointing: {}", grad_norm_with_ckpt);
    println!("Grad norm difference: {}", grad_norm_diff);
    assert!(
        grad_norm_diff < 1e-4,
        "Gradient norms should be very close, got diff {}",
        grad_norm_diff
    );

    // Compare final weights (should be identical since gradients are the same)
    let mut max_weight_diff: f32 = 0.0;
    for (w1, w2) in weights_no_ckpt.iter().zip(weights_with_ckpt.iter()) {
        let diff = (w1 - w2).abs();
        if diff > max_weight_diff {
            max_weight_diff = diff;
        }
    }
    println!("Max weight difference: {}", max_weight_diff);
    assert!(
        max_weight_diff < 1e-4,
        "Weights should be very close, got max diff {}",
        max_weight_diff
    );

    println!("Activation checkpointing test passed!");
}

#[test]
fn test_activation_checkpointing_interval() {
    use crate::train::CheckpointConfig;

    // Test checkpointing with interval > 1 (only some layers checkpointed)
    let model_config = ModelConfig {
        vocab_size: 100,
        hidden_dim: 32,
        num_layers: 4, // Need at least 4 layers to test interval=2
        num_heads: 2,
        num_kv_heads: 2,
        intermediate_dim: 64,
        norm_eps: 1e-5,
        rope_base: 10000.0,
        max_seq_len: 128,
        tie_weights: true,
        precision: Precision::FP32,
        embed_dropout: 0.0,
        attn_dropout: 0.0,
        ffn_dropout: 0.0,
    };

    // Train with checkpointing every 2nd layer (layers 0 and 2 checkpointed)
    let train_config = TrainingConfig {
        dropout_enabled: false,
        async_gpu: false,
        checkpoint_config: CheckpointConfig::with_interval(2),
        ..Default::default()
    };

    let mut trainer = Trainer::new(&model_config, &train_config);

    let batch_size = 2;
    let seq_len = 8;
    let n = batch_size * seq_len;
    let input_ids: Vec<u32> = (0..n).map(|i| (i % 100) as u32).collect();
    let target_ids: Vec<u32> = (0..n).map(|i| ((i + 1) % 100) as u32).collect();

    // Should complete without errors
    let (loss, grad_norm) = trainer.train_step(&input_ids, &target_ids, batch_size, seq_len);

    assert!(loss > 0.0, "Loss should be positive");
    assert!(grad_norm > 0.0, "Grad norm should be positive");
    assert_eq!(trainer.step, 1, "Step should increment");

    println!("Activation checkpointing with interval=2 passed!");
    println!("  loss={}, grad_norm={}", loss, grad_norm);
}

/// Benchmark memory usage with vs without activation checkpointing.
///
/// Run with: cargo test benchmark_checkpointing_memory --release -- --nocapture --ignored
#[test]
#[ignore]
fn benchmark_checkpointing_memory() {
    use crate::device::{format_bytes, gpu_memory_allocated};
    use crate::train::CheckpointConfig;

    // Use a larger model to see meaningful memory differences
    let model_config = ModelConfig {
        vocab_size: 2048,
        hidden_dim: 512,
        num_layers: 8,
        num_heads: 8,
        num_kv_heads: 8,
        intermediate_dim: 512 * 4, // 2048
        norm_eps: 1e-5,
        rope_base: 10000.0,
        max_seq_len: 512,
        tie_weights: true,
        precision: Precision::FP32,
        embed_dropout: 0.0,
        attn_dropout: 0.0,
        ffn_dropout: 0.0,
    };

    let batch_size = 16;
    let seq_len = 256;
    let n = batch_size * seq_len;

    println!("\n{}", "=".repeat(70));
    println!("Activation Checkpointing Memory Benchmark");
    println!("{}", "=".repeat(70));
    println!(
        "\nModel: hidden={}, layers={}, heads={}",
        model_config.hidden_dim, model_config.num_layers, model_config.num_heads
    );
    println!("Batch: batch_size={}, seq_len={}, tokens={}", batch_size, seq_len, n);

    // Prepare input data
    let input_ids: Vec<u32> = (0..n).map(|i| (i % model_config.vocab_size) as u32).collect();

    // ===== Test 1: Without checkpointing =====
    println!("\n--- Without Checkpointing ---");

    let train_config_no_ckpt = TrainingConfig {
        dropout_enabled: false,
        async_gpu: false,
        checkpoint_config: CheckpointConfig::default(), // Disabled
        ..Default::default()
    };

    // Force GC-like behavior by dropping any previous allocations
    std::thread::sleep(std::time::Duration::from_millis(100));

    let mem_before_no_ckpt = gpu_memory_allocated();
    let trainer_no_ckpt = Trainer::new(&model_config, &train_config_no_ckpt);
    let mem_after_model = gpu_memory_allocated();

    println!("  Model memory: {}", format_bytes(mem_after_model - mem_before_no_ckpt));

    // Do forward pass to capture activation memory
    let cache = trainer_no_ckpt.forward_with_cache(&input_ids, batch_size, seq_len);
    let mem_after_forward_no_ckpt = gpu_memory_allocated();
    let activation_mem_no_ckpt = mem_after_forward_no_ckpt - mem_after_model;

    println!(
        "  Activation memory (forward cache): {}",
        format_bytes(activation_mem_no_ckpt)
    );

    // Clean up
    drop(cache);
    drop(trainer_no_ckpt);
    std::thread::sleep(std::time::Duration::from_millis(100));

    // ===== Test 2: With checkpointing (all layers) =====
    println!("\n--- With Checkpointing (all layers) ---");

    let train_config_with_ckpt = TrainingConfig {
        dropout_enabled: false,
        async_gpu: false,
        checkpoint_config: CheckpointConfig::enabled(), // All layers
        ..Default::default()
    };

    let mem_before_ckpt = gpu_memory_allocated();
    let trainer_with_ckpt = Trainer::new(&model_config, &train_config_with_ckpt);
    let mem_after_model_ckpt = gpu_memory_allocated();

    println!(
        "  Model memory: {}",
        format_bytes(mem_after_model_ckpt - mem_before_ckpt)
    );

    // Do forward pass with checkpointing
    let cache_ckpt = trainer_with_ckpt.forward_with_checkpointing(&input_ids, batch_size, seq_len);
    let mem_after_forward_ckpt = gpu_memory_allocated();
    let activation_mem_ckpt = mem_after_forward_ckpt - mem_after_model_ckpt;

    println!(
        "  Activation memory (checkpointed): {}",
        format_bytes(activation_mem_ckpt)
    );

    // Clean up
    drop(cache_ckpt);
    drop(trainer_with_ckpt);

    // ===== Summary =====
    println!("\n--- Summary ---");
    println!("  Without checkpointing: {}", format_bytes(activation_mem_no_ckpt));
    println!("  With checkpointing:    {}", format_bytes(activation_mem_ckpt));

    if activation_mem_no_ckpt > activation_mem_ckpt {
        let savings = activation_mem_no_ckpt - activation_mem_ckpt;
        let ratio = activation_mem_no_ckpt as f64 / activation_mem_ckpt.max(1) as f64;
        println!(
            "  Memory saved:          {} ({:.1}x reduction)",
            format_bytes(savings),
            ratio
        );
    } else {
        println!(
            "  Note: Checkpointed memory >= non-checkpointed (may be due to measurement timing)"
        );
    }

    println!("{}", "=".repeat(70));
}
