use irontensor::{
    // Forward ops
    add, attention, embedding, gelu, matmul, mul, relu, rmsnorm, rope, scale, silu, softmax,
    swiglu,
    // Backward ops
    cross_entropy_fused, embedding_backward, gelu_backward, matmul_backward, mul_backward,
    relu_backward, rmsnorm_backward, rope_backward, scale_backward, silu_backward,
    softmax_backward, swiglu_backward,
    // Optimizer
    clip_grad_norm, grad_norm, zero_gradients, Lion, LionConfig, ParamState,
    // Neural network modules
    GPTModel, ModelConfig, TransformerBlock,
    // Data loading
    TokenDataset,
    // Core types
    MetalContext, Tensor,
};
use objc2_metal::MTLDevice;

fn main() {
    // Initialize the Metal context
    let ctx = MetalContext::global();
    println!("IronTensor - Metal GPU Tensor Library");
    println!("======================================");
    println!("Device: {}\n", ctx.device().name());

    // =====================================================================
    // PHASE 1: Forward Operations
    // =====================================================================
    println!("=== Phase 1: Forward Operations ===\n");

    // --- Matrix Multiplication (GEMM) ---
    println!("1. Matrix Multiplication (matmul)");
    let a = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let c = matmul(&a, &b);
    println!("   A[2x3] @ B[3x2] = C[2x2]");
    println!("   Result: {:?}\n", c.as_f32_slice());

    // --- Element-wise Operations ---
    println!("2. Element-wise Operations");
    let x = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let y = Tensor::from_f32_slice(&[0.5, 1.0, 1.5, 2.0], &[4]);

    let sum = add(&x, &y);
    println!("   add([1,2,3,4], [0.5,1,1.5,2]) = {:?}", sum.as_f32_slice());

    let prod = mul(&x, &y);
    println!("   mul([1,2,3,4], [0.5,1,1.5,2]) = {:?}", prod.as_f32_slice());

    let scaled = scale(&x, 2.0);
    println!("   scale([1,2,3,4], 2.0) = {:?}\n", scaled.as_f32_slice());

    // --- Activation Functions ---
    println!("3. Activation Functions");
    let act_input = Tensor::from_f32_slice(&[-1.0, 0.0, 1.0, 2.0], &[4]);

    let silu_out = silu(&act_input);
    println!("   SiLU([-1,0,1,2]) = {:?}", silu_out.as_f32_slice());

    let gelu_out = gelu(&act_input);
    println!("   GELU([-1,0,1,2]) = {:?}", gelu_out.as_f32_slice());

    let relu_out = relu(&act_input);
    println!("   ReLU([-1,0,1,2]) = {:?}", relu_out.as_f32_slice());

    // SwiGLU (used in Llama-style FFN)
    let gate = Tensor::from_f32_slice(&[1.0, 2.0, -1.0, 0.5], &[4]);
    let up = Tensor::from_f32_slice(&[1.0, 1.0, 1.0, 1.0], &[4]);
    let swiglu_out = swiglu(&gate, &up);
    println!("   SwiGLU(gate, up) = {:?}\n", swiglu_out.as_f32_slice());

    // --- RMSNorm ---
    println!("4. RMSNorm (Layer Normalization)");
    let norm_input = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
    let gamma = Tensor::from_f32_slice(&[1.0, 1.0, 1.0, 1.0], &[4]);
    let normed = rmsnorm(&norm_input, &gamma, 1e-5);
    println!("   Input shape: [2, 4], hidden_dim=4");
    println!("   Output: {:?}\n", normed.as_f32_slice());

    // --- Softmax ---
    println!("5. Softmax");
    let logits = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0], &[2, 4]);
    let probs = softmax(&logits);
    println!("   Input: [1,2,3,4] and [1,1,1,1]");
    println!("   Probs: {:?}\n", probs.as_f32_slice());

    // --- Embedding Lookup ---
    println!("6. Embedding Lookup");
    let vocab_size = 5;
    let embed_dim = 4;
    let weights: Vec<f32> = (0..(vocab_size * embed_dim)).map(|i| i as f32 * 0.1).collect();
    let embed_weights = Tensor::from_f32_slice(&weights, &[vocab_size, embed_dim]);
    let indices = vec![0u32, 2, 4];
    let embedded = embedding(&embed_weights, &indices);
    println!("   Vocab size: {}, Embed dim: {}", vocab_size, embed_dim);
    println!("   Indices: {:?}", indices);
    println!("   Embedded[0]: {:?}", &embedded.as_f32_slice()[0..4]);
    println!("   Embedded[2]: {:?}", &embedded.as_f32_slice()[4..8]);
    println!("   Embedded[4]: {:?}\n", &embedded.as_f32_slice()[8..12]);

    // --- RoPE (Rotary Position Embedding) ---
    println!("7. RoPE (Rotary Position Embedding)");
    let batch = 1;
    let seq_len = 2;
    let num_heads = 2;
    let head_dim = 4;
    let rope_data = vec![1.0f32; batch * seq_len * num_heads * head_dim];
    let rope_input = Tensor::from_f32_slice(&rope_data, &[batch, seq_len, num_heads, head_dim]);
    let rope_out = rope(&rope_input, 10000.0, 0);
    println!("   Input shape: [batch={}, seq={}, heads={}, head_dim={}]", batch, seq_len, num_heads, head_dim);
    println!("   Position 0: {:?}", &rope_out.as_f32_slice()[0..head_dim]);
    println!("   Position 1: {:?}\n", &rope_out.as_f32_slice()[num_heads * head_dim..num_heads * head_dim + head_dim]);

    // --- Attention ---
    println!("8. Attention");
    let batch = 1;
    let heads = 2;
    let seq = 4;
    let head_d = 8;
    let qkv_data: Vec<f32> = (0..(batch * heads * seq * head_d))
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let q = Tensor::from_f32_slice(&qkv_data, &[batch, heads, seq, head_d]);
    let k = Tensor::from_f32_slice(&qkv_data, &[batch, heads, seq, head_d]);
    let v = Tensor::from_f32_slice(&qkv_data, &[batch, heads, seq, head_d]);
    let attn_out = attention(&q, &k, &v, true); // causal=true
    println!("   Q/K/V shape: [batch={}, heads={}, seq={}, head_dim={}]", batch, heads, seq, head_d);
    println!("   Causal masking: enabled");
    println!("   Output shape: {:?}\n", attn_out.shape());

    // =====================================================================
    // PHASE 2: Backward Operations (Autodiff)
    // =====================================================================
    println!("=== Phase 2: Backward Operations ===\n");

    // --- Matmul Backward ---
    println!("1. Matmul Backward");
    let a = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let grad_c = Tensor::from_f32_slice(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let (grad_a, grad_b) = matmul_backward(&grad_c, &a, &b);
    println!("   grad_C[2x2] -> grad_A[2x3], grad_B[3x2]");
    println!("   grad_A: {:?}", grad_a.as_f32_slice());
    println!("   grad_B: {:?}\n", grad_b.as_f32_slice());

    // --- Element-wise Backward ---
    println!("2. Element-wise Backward");
    let x = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let y = Tensor::from_f32_slice(&[0.5, 1.0, 1.5, 2.0], &[4]);
    let grad_out = Tensor::from_f32_slice(&[1.0, 1.0, 1.0, 1.0], &[4]);

    let (grad_x, grad_y) = mul_backward(&grad_out, &x, &y);
    println!("   mul_backward: grad_x={:?}, grad_y={:?}", grad_x.as_f32_slice(), grad_y.as_f32_slice());

    let grad_scaled = scale_backward(&grad_out, 2.0);
    println!("   scale_backward(scalar=2.0): {:?}\n", grad_scaled.as_f32_slice());

    // --- Activation Backward ---
    println!("3. Activation Backward");
    let act_input = Tensor::from_f32_slice(&[-1.0, 0.0, 1.0, 2.0], &[4]);
    let grad_out = Tensor::from_f32_slice(&[1.0, 1.0, 1.0, 1.0], &[4]);

    let grad_silu = silu_backward(&grad_out, &act_input);
    println!("   silu_backward: {:?}", grad_silu.as_f32_slice());

    let grad_gelu = gelu_backward(&grad_out, &act_input);
    println!("   gelu_backward: {:?}", grad_gelu.as_f32_slice());

    let grad_relu = relu_backward(&grad_out, &act_input);
    println!("   relu_backward: {:?}", grad_relu.as_f32_slice());

    let gate = Tensor::from_f32_slice(&[1.0, 2.0, -1.0, 0.5], &[4]);
    let up = Tensor::from_f32_slice(&[1.0, 1.0, 1.0, 1.0], &[4]);
    let (grad_gate, grad_up) = swiglu_backward(&grad_out, &gate, &up);
    println!("   swiglu_backward: grad_gate={:?}", grad_gate.as_f32_slice());
    println!("                    grad_up={:?}\n", grad_up.as_f32_slice());

    // --- RMSNorm Backward ---
    println!("4. RMSNorm Backward");
    let norm_input = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let gamma = Tensor::from_f32_slice(&[1.0, 1.0, 1.0, 1.0], &[4]);
    let grad_out = Tensor::from_f32_slice(&[1.0, 1.0, 1.0, 1.0], &[1, 4]);
    let (grad_input, grad_gamma) = rmsnorm_backward(&grad_out, &norm_input, &gamma, 1e-5);
    println!("   grad_input: {:?}", grad_input.as_f32_slice());
    println!("   grad_gamma: {:?}\n", grad_gamma.as_f32_slice());

    // --- Softmax Backward ---
    println!("5. Softmax Backward");
    let logits = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let probs = softmax(&logits);
    let grad_out = Tensor::from_f32_slice(&[0.1, 0.2, 0.3, 0.4], &[1, 4]);
    let grad_logits = softmax_backward(&grad_out, &probs);
    println!("   Probs: {:?}", probs.as_f32_slice());
    println!("   grad_logits: {:?}\n", grad_logits.as_f32_slice());

    // --- Embedding Backward ---
    println!("6. Embedding Backward");
    let vocab_size = 5;
    let embed_dim = 4;
    let indices = vec![0u32, 2, 0]; // Note: index 0 appears twice
    let grad_out = Tensor::from_f32_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        &[3, embed_dim],
    );
    let grad_weights = embedding_backward(&grad_out, &indices, vocab_size);
    println!("   Indices: {:?} (0 appears twice)", indices);
    println!("   grad_weights[0] (accumulated): {:?}", &grad_weights.as_f32_slice()[0..4]);
    println!("   grad_weights[2]: {:?}\n", &grad_weights.as_f32_slice()[8..12]);

    // --- RoPE Backward ---
    println!("7. RoPE Backward");
    {
        let batch = 1;
        let seq_len = 2;
        let num_heads = 1;
        let head_dim = 4;
        let grad_data = vec![1.0f32; batch * seq_len * num_heads * head_dim];
        let grad_out = Tensor::from_f32_slice(&grad_data, &[batch, seq_len, num_heads, head_dim]);
        let grad_input = rope_backward(&grad_out, 10000.0, 0);
        println!("   RoPE backward applies inverse rotation");
        println!("   grad_input[0]: {:?}\n", &grad_input.as_f32_slice()[0..head_dim]);
    }

    // --- Cross-Entropy Loss (Fused) ---
    println!("8. Cross-Entropy Loss (Fused Softmax + Loss + Backward)");
    let batch_size = 2;
    let vocab_size = 5;
    let logits = Tensor::from_f32_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, // batch 0: highest logit at index 4
            5.0, 4.0, 3.0, 2.0, 1.0, // batch 1: highest logit at index 0
        ],
        &[batch_size, vocab_size],
    );
    let targets = vec![4u32, 0]; // correct predictions
    let (loss, _probs, grad_logits) = cross_entropy_fused(&logits, &targets);
    println!("   Logits shape: [batch={}, vocab={}]", batch_size, vocab_size);
    println!("   Targets: {:?} (correct predictions)", targets);
    println!("   Loss: {:.4}", loss);
    println!("   grad_logits[0]: {:?}", &grad_logits.as_f32_slice()[0..vocab_size]);
    println!("   grad_logits[1]: {:?}\n", &grad_logits.as_f32_slice()[vocab_size..2 * vocab_size]);

    // =====================================================================
    // PHASE 3: Optimizer
    // =====================================================================
    println!("=== Phase 3: Optimizer ===\n");

    // --- Lion Optimizer ---
    println!("1. Lion Optimizer (Sign-based Updates)");
    println!("   Training a simple quadratic: f(x) = sum(x^2)");

    // Initialize weights
    let weights = Tensor::from_f32_slice(&[5.0, -3.0, 2.0, -1.0], &[4]);
    let mut state = ParamState::new(&[4]);

    let optimizer = Lion::new(LionConfig {
        lr: 0.1,
        beta1: 0.9,
        beta2: 0.99,
        weight_decay: 0.0,
    });

    println!("   Initial weights: {:?}", weights.as_f32_slice());

    // Training loop
    for step in 0..10 {
        // Gradient of x^2 is 2x
        let w = weights.as_f32_slice();
        let grads_data: Vec<f32> = w.iter().map(|&x| 2.0 * x).collect();
        let gradients = Tensor::from_f32_slice(&grads_data, &[4]);

        optimizer.step(&weights, &gradients, &mut state);

        if step == 0 || step == 4 || step == 9 {
            println!("   Step {}: weights = {:?}", step + 1, weights.as_f32_slice());
        }
    }

    // --- Gradient Utilities ---
    println!("\n2. Gradient Utilities");

    let gradients = Tensor::from_f32_slice(&[3.0, 4.0, 0.0, 0.0], &[4]);
    let norm = grad_norm(&gradients);
    println!("   grad_norm([3, 4, 0, 0]) = {:.4}", norm);

    let gradients = Tensor::from_f32_slice(&[6.0, 8.0, 0.0, 0.0], &[4]);
    let original_norm = clip_grad_norm(&gradients, 5.0);
    println!("   clip_grad_norm([6, 8, 0, 0], max=5.0):");
    println!("     Original norm: {:.4}", original_norm);
    println!("     Clipped grads: {:?}", gradients.as_f32_slice());
    println!("     New norm: {:.4}", grad_norm(&gradients));

    let gradients = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
    zero_gradients(&gradients);
    println!("   zero_gradients: {:?}", gradients.as_f32_slice());

    // --- Lion with Weight Decay ---
    println!("\n3. Lion with Weight Decay");
    let weights = Tensor::from_f32_slice(&[10.0, -10.0], &[2]);
    let mut state = ParamState::new(&[2]);

    let optimizer = Lion::new(LionConfig {
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.99,
        weight_decay: 0.1, // 10% weight decay
    });

    println!("   Initial: {:?}", weights.as_f32_slice());

    // Zero gradients - only weight decay affects weights
    let zero_grads = Tensor::from_f32_slice(&[0.0, 0.0], &[2]);
    for _ in 0..5 {
        optimizer.step(&weights, &zero_grads, &mut state);
    }
    println!("   After 5 steps (zero grad, 10% decay): {:?}\n", weights.as_f32_slice());

    // =====================================================================
    // PHASE 4: Model & Data
    // =====================================================================
    println!("=== Phase 4: Model & Data ===\n");

    // --- Model Configuration ---
    println!("1. Model Configuration");
    let tiny_config = ModelConfig::tiny();
    println!("   Tiny model: {} layers, {} hidden, {:.2}M params",
        tiny_config.num_layers,
        tiny_config.hidden_dim,
        tiny_config.num_params() as f64 / 1e6
    );

    let small_config = ModelConfig::small();
    println!("   Small model: {} layers, {} hidden, {:.2}M params",
        small_config.num_layers,
        small_config.hidden_dim,
        small_config.num_params() as f64 / 1e6
    );

    let medium_config = ModelConfig::medium();
    println!("   Medium model: {} layers, {} hidden, {:.2}M params\n",
        medium_config.num_layers,
        medium_config.hidden_dim,
        medium_config.num_params() as f64 / 1e6
    );

    // --- GPT Model Forward Pass ---
    println!("2. GPT Model Forward Pass");
    let config = ModelConfig::tiny();
    let model = GPTModel::new(config.clone());
    println!("{}", model.summary());

    let batch = 1;
    let seq_len = 16;
    let input_ids: Vec<u32> = (0..batch * seq_len).map(|i| (i % 100) as u32).collect();

    println!("   Input: {} tokens", input_ids.len());
    let logits = model.forward(&input_ids, batch, seq_len, 0);
    println!("   Output logits shape: {:?}", logits.shape());

    // Show top prediction for first position
    let first_logits = &logits.as_f32_slice()[0..config.vocab_size];
    let (max_idx, max_val) = first_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    println!("   First token prediction: token {} (logit {:.4})\n", max_idx, max_val);

    // --- Transformer Block ---
    println!("3. Transformer Block");
    let block = TransformerBlock::new(64, 4, 4, 128, 10000.0, 1e-5);
    println!("   Block params: {}", block.num_params());

    let block_input_data: Vec<f32> = (0..2 * 8 * 64).map(|i| (i as f32 * 0.01).sin()).collect();
    let block_input = Tensor::from_f32_slice(&block_input_data, &[2, 8, 64]);
    let block_output = block.forward(&block_input, 0, true);
    println!("   Input: [2, 8, 64] -> Output: {:?}\n", block_output.shape());

    // --- Memory-Mapped Dataset ---
    println!("4. Memory-Mapped Dataset");
    let dataset_path = std::env::temp_dir().join("irontensor_demo.bin");

    // Create a small demo dataset
    let demo_tokens: Vec<u32> = (0..1000).collect();
    TokenDataset::create(&dataset_path, &demo_tokens).unwrap();
    println!("   Created dataset with {} tokens", demo_tokens.len());

    let dataset = TokenDataset::open(&dataset_path, 32).unwrap();
    println!("   Sequence length: {}", dataset.seq_len());
    println!("   Number of sequences: {}", dataset.num_sequences());

    // Get a training batch
    let (input_batch, target_batch) = dataset.get_batch(0);
    println!("   Batch 0 input:  {:?}...", &input_batch[0..5]);
    println!("   Batch 0 target: {:?}...", &target_batch[0..5]);

    // Cleanup
    std::fs::remove_file(dataset_path).ok();
    println!();

    // =====================================================================
    // Summary
    // =====================================================================
    println!("=== Summary ===");
    println!("IronTensor provides GPU-accelerated tensor operations for LLM training:");
    println!("- Phase 1: Forward ops (matmul, activations, normalization, attention, etc.)");
    println!("- Phase 2: Backward ops for automatic differentiation");
    println!("- Phase 3: Lion optimizer with gradient clipping and weight decay");
    println!("- Phase 4: GPT model architecture and memory-mapped data loading");
    println!("\nAll operations run on Metal GPU with unified memory (zero-copy on Apple Silicon).");
}
