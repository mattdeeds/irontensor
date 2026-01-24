use crate::ops::{
    add as gpu_add, add3 as gpu_add3, causal_mask_3d_gpu, matmul, matmul_mps_nt, matmul_mps_tn,
    repeat_kv_backward_gpu, repeat_kv_gpu, scale as gpu_scale, scale_tensors_inplace,
    softmax, softmax_backward, to_f32_gpu, transpose_3d_gpu,
};
use crate::precision::Precision;
use crate::profile::Profiler;
use crate::tensor::Tensor;

/// Ensure tensor is FP32 for compute. If BF16, converts to FP32.
/// This enables mixed precision: weights stored in BF16 but computed in FP32.
pub fn ensure_fp32(t: &Tensor) -> Tensor {
    if t.precision() == Precision::BF16 {
        to_f32_gpu(t)
    } else {
        t.clone()
    }
}

/// Scale tensor by a scalar (GPU-accelerated)
pub(crate) fn scale_tensor(t: &Tensor, scale: f32) -> Tensor {
    gpu_scale(t, scale).unwrap()
}

/// Add two tensors element-wise (GPU-accelerated)
pub(crate) fn add_tensors(a: &Tensor, b: &Tensor) -> Tensor {
    gpu_add(a, b).unwrap()
}

/// Add three tensors element-wise (GPU-accelerated, fused kernel)
/// More efficient than chaining two add_tensors calls
pub(crate) fn add3_tensors(a: &Tensor, b: &Tensor, c: &Tensor) -> Tensor {
    gpu_add3(a, b, c).unwrap()
}

/// Scale multiple gradients in-place with the same scalar
/// Useful for gradient clipping to avoid allocating new tensors
pub(crate) fn scale_gradients_inplace(grads: &[&Tensor], scale: f32) {
    scale_tensors_inplace(grads, scale).unwrap()
}

/// Linear forward: output = input @ weight.T
///
/// Uses MPS's native transpose support to avoid explicit transposition.
/// Supports mixed precision: BF16 weights are converted to FP32 for compute.
///
/// Note: MPS does not support BFloat16 input, so conversion is required.
pub(crate) fn linear_forward(input: &Tensor, weight: &Tensor) -> Tensor {
    // weight: [out, in], we want input @ weight.T
    // matmul_mps_nt handles the transpose natively
    let weight_fp32 = ensure_fp32(weight);
    matmul_mps_nt(input, &weight_fp32)
}

/// Linear backward: grad_input = grad_output @ weight, grad_weight = grad_output.T @ input
///
/// Supports mixed precision: BF16 weights are converted to FP32 for compute.
/// Returns FP32 gradients regardless of weight precision.
///
/// Note: MPS does not support BFloat16 input, so conversion is required.
pub(crate) fn linear_backward(
    grad_output: &Tensor,
    input: &Tensor,
    weight: &Tensor,
) -> (Tensor, Tensor) {
    // Convert weight to FP32 if needed for mixed precision
    let weight_fp32 = ensure_fp32(weight);

    // grad_input = grad_output @ weight
    let grad_input = matmul(grad_output, &weight_fp32).unwrap();

    // grad_weight = grad_output.T @ input
    let grad_weight = matmul_tn(grad_output, input);

    (grad_input, grad_weight)
}

/// Matrix multiply with first operand transposed: A.T @ B
///
/// Uses MPS's native transpose support to avoid explicit transposition.
pub(crate) fn matmul_tn(a: &Tensor, b: &Tensor) -> Tensor {
    matmul_mps_tn(a, b)
}

/// Repeat KV heads for GQA (Grouped Query Attention) - GPU accelerated
pub(crate) fn repeat_kv(
    x: &Tensor,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Tensor {
    repeat_kv_gpu(x, batch, seq_len, num_heads, num_kv_heads, head_dim)
}

/// Transpose last two dimensions of a 3D tensor: [batch, m, n] -> [batch, n, m] (GPU-accelerated)
pub(crate) fn transpose_last_two_dims_3d(t: &Tensor, batch: usize, m: usize, n: usize) -> Tensor {
    transpose_3d_gpu(t, batch, m, n)
}

/// Apply causal mask to 3D attention scores: [batch, seq, seq] (GPU-accelerated)
/// Sets positions where col > row to -inf
pub(crate) fn apply_causal_mask_3d(scores: &Tensor, batch: usize, seq_len: usize) -> Tensor {
    causal_mask_3d_gpu(scores, batch, seq_len)
}

/// Attention backward pass that recomputes attention weights from Q, K, V
///
/// This version doesn't require cached attention weights, instead recomputing them
/// during backward. Trades compute for memory - useful with FlashAttention forward.
///
/// Input shapes:
///   - grad_output: [batch, heads, seq, head_dim]
///   - q: [batch, heads, seq, head_dim]
///   - k: [batch, heads, seq, head_dim]
///   - v: [batch, heads, seq, head_dim]
///
/// Returns: (grad_Q, grad_K, grad_V) all with shape [batch, heads, seq, head_dim]
pub(crate) fn attention_backward_recompute(
    grad_output: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> (Tensor, Tensor, Tensor) {
    Profiler::push_tag("attn_bwd");

    let q_shape = q.shape();
    let batch = q_shape[0];
    let heads = q_shape[1];
    let seq_len = q_shape[2];
    let head_dim = q_shape[3];
    let batch_heads = batch * heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Reshape to 3D for batched matmul: [batch*heads, seq, dim]
    let q_flat = q.view(&[batch_heads, seq_len, head_dim]);
    let k_flat = k.view(&[batch_heads, seq_len, head_dim]);
    let v_flat = v.view(&[batch_heads, seq_len, head_dim]);
    let grad_o_flat = grad_output.view(&[batch_heads, seq_len, head_dim]);

    // Recompute attention weights: P = softmax(causal_mask(Q @ K^T / sqrt(d)))
    Profiler::push_tag("recompute");
    // K^T: [batch*heads, head_dim, seq]
    let k_t = transpose_last_two_dims_3d(&k_flat, batch_heads, seq_len, head_dim);

    // scores = Q @ K^T: [batch*heads, seq, seq]
    let scores = matmul(&q_flat, &k_t).unwrap();

    // Scale and apply causal mask
    let scores_scaled = scale_tensor(&scores, scale);
    let scores_masked = apply_causal_mask_3d(&scores_scaled, batch_heads, seq_len);

    // Softmax to get attention weights
    let p_flat = softmax(&scores_masked).unwrap();
    Profiler::pop_tag();

    // Now compute gradients using the recomputed P
    // grad_V = P^T @ grad_O
    Profiler::push_tag("grad_v");
    let p_t = transpose_last_two_dims_3d(&p_flat, batch_heads, seq_len, seq_len);
    let grad_v_flat = matmul(&p_t, &grad_o_flat).unwrap();
    Profiler::pop_tag();

    // grad_P = grad_O @ V^T
    Profiler::push_tag("grad_p");
    let v_t = transpose_last_two_dims_3d(&v_flat, batch_heads, seq_len, head_dim);
    let grad_p_flat = matmul(&grad_o_flat, &v_t).unwrap();

    // grad_S = softmax_backward(grad_P, P)
    let grad_s_flat = softmax_backward(&grad_p_flat, &p_flat);
    Profiler::pop_tag();

    // grad_Q = grad_S @ K / sqrt(d)
    Profiler::push_tag("grad_qk");
    let grad_q_unscaled = matmul(&grad_s_flat, &k_flat).unwrap();
    let grad_q_flat = scale_tensor(&grad_q_unscaled, scale);

    // grad_K = grad_S^T @ Q / sqrt(d)
    let grad_s_t = transpose_last_two_dims_3d(&grad_s_flat, batch_heads, seq_len, seq_len);
    let grad_k_unscaled = matmul(&grad_s_t, &q_flat).unwrap();
    let grad_k_flat = scale_tensor(&grad_k_unscaled, scale);
    Profiler::pop_tag();

    // Reshape back to 4D
    let grad_q = grad_q_flat.view(&[batch, heads, seq_len, head_dim]);
    let grad_k = grad_k_flat.view(&[batch, heads, seq_len, head_dim]);
    let grad_v = grad_v_flat.view(&[batch, heads, seq_len, head_dim]);

    Profiler::pop_tag();

    (grad_q, grad_k, grad_v)
}

/// Backward for repeat_kv - sums gradients from expanded heads back to KV heads (GPU accelerated)
///
/// Forward: [batch, seq, kv_heads, head_dim] -> [batch, seq, num_heads, head_dim]
/// Backward: [batch, seq, num_heads, head_dim] -> [batch, seq, kv_heads, head_dim]
pub(crate) fn repeat_kv_backward(
    grad_expanded: &Tensor,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Tensor {
    repeat_kv_backward_gpu(grad_expanded, batch, seq_len, num_heads, num_kv_heads, head_dim)
}

/// Attention backward pass using custom kernels with pipelined execution.
///
/// NOTE: Benchmarks show this is NOT faster than the MPS-based `attention_backward_recompute`.
/// MPS's AMX coprocessor provides 2-4x faster matmul, which outweighs the sync overhead savings.
/// This implementation is kept for reference and future optimization experiments.
///
/// This version uses custom GEMM kernels that integrate with CommandBatch,
/// allowing independent operations to execute without intermediate GPU syncs.
///
/// Input shapes:
///   - grad_output: [batch, heads, seq, head_dim]
///   - q: [batch, heads, seq, head_dim]
///   - k: [batch, heads, seq, head_dim]
///   - v: [batch, heads, seq, head_dim]
///
/// Returns: (grad_Q, grad_K, grad_V) all with shape [batch, heads, seq, head_dim]
#[cfg(test)]
pub(crate) fn attention_backward_pipelined(
    grad_output: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> (Tensor, Tensor, Tensor) {
    use crate::command_batch::CommandBatch;
    use crate::ops::matmul_custom;

    Profiler::push_tag("attn_bwd_pipelined");

    let q_shape = q.shape();
    let batch = q_shape[0];
    let heads = q_shape[1];
    let seq_len = q_shape[2];
    let head_dim = q_shape[3];
    let batch_heads = batch * heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Reshape to 3D for batched matmul: [batch*heads, seq, dim]
    let q_flat = q.view(&[batch_heads, seq_len, head_dim]);
    let k_flat = k.view(&[batch_heads, seq_len, head_dim]);
    let v_flat = v.view(&[batch_heads, seq_len, head_dim]);
    let grad_o_flat = grad_output.view(&[batch_heads, seq_len, head_dim]);

    // === PHASE 1: Recompute attention weights (sequential, data dependent) ===
    // Must sync after this phase because we need P for phase 2
    Profiler::push_tag("recompute");
    CommandBatch::begin();

    // K^T: [batch*heads, head_dim, seq]
    let k_t = transpose_last_two_dims_3d(&k_flat, batch_heads, seq_len, head_dim);

    // scores = Q @ K^T: [batch*heads, seq, seq] - uses custom kernel, no sync
    let scores = matmul_custom(&q_flat, &k_t);

    // Scale and apply causal mask
    let scores_scaled = scale_tensor(&scores, scale);
    let scores_masked = apply_causal_mask_3d(&scores_scaled, batch_heads, seq_len);

    // Softmax to get attention weights - needs sync before reading
    CommandBatch::sync();
    let p_flat = softmax(&scores_masked).unwrap();
    Profiler::pop_tag();

    // === PHASE 2: Compute grad_V and grad_P (INDEPENDENT - can pipeline) ===
    Profiler::push_tag("grad_vp");
    CommandBatch::begin();

    // V^T: [batch*heads, head_dim, seq] - independent of P^T
    let v_t = transpose_last_two_dims_3d(&v_flat, batch_heads, seq_len, head_dim);
    // P^T: [batch*heads, seq, seq]
    let p_t = transpose_last_two_dims_3d(&p_flat, batch_heads, seq_len, seq_len);

    // grad_V = P^T @ grad_O: [batch*heads, seq, head_dim]
    let grad_v_flat = matmul_custom(&p_t, &grad_o_flat);

    // grad_P = grad_O @ V^T: [batch*heads, seq, seq]
    let grad_p_flat = matmul_custom(&grad_o_flat, &v_t);

    // Must sync before softmax_backward reads grad_p_flat
    CommandBatch::sync();

    // grad_S = softmax_backward(grad_P, P)
    let grad_s_flat = softmax_backward(&grad_p_flat, &p_flat);
    Profiler::pop_tag();

    // === PHASE 3: Compute grad_Q and grad_K (INDEPENDENT - can pipeline) ===
    Profiler::push_tag("grad_qk");
    CommandBatch::begin();

    // grad_S^T: [batch*heads, seq, seq]
    let grad_s_t = transpose_last_two_dims_3d(&grad_s_flat, batch_heads, seq_len, seq_len);

    // grad_Q = grad_S @ K: [batch*heads, seq, head_dim]
    let grad_q_unscaled = matmul_custom(&grad_s_flat, &k_flat);

    // grad_K = grad_S^T @ Q: [batch*heads, seq, head_dim]
    let grad_k_unscaled = matmul_custom(&grad_s_t, &q_flat);

    // Sync before scaling reads the results
    CommandBatch::sync();

    let grad_q_flat = scale_tensor(&grad_q_unscaled, scale);
    let grad_k_flat = scale_tensor(&grad_k_unscaled, scale);
    Profiler::pop_tag();

    // Must sync before view() to ensure data is ready
    CommandBatch::sync();

    // Reshape back to 4D
    let grad_q = grad_q_flat.view(&[batch, heads, seq_len, head_dim]);
    let grad_k = grad_k_flat.view(&[batch, heads, seq_len, head_dim]);
    let grad_v = grad_v_flat.view(&[batch, heads, seq_len, head_dim]);

    CommandBatch::end();
    Profiler::pop_tag();

    (grad_q, grad_k, grad_v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command_batch::CommandBatch;
    use crate::tensor::Tensor;

    /// Benchmark comparing MPS-based attention backward vs pipelined custom kernels
    /// Run with: cargo test benchmark_attention_backward --release -- --nocapture --ignored
    #[test]
    #[ignore]
    fn benchmark_attention_backward() {
        use std::time::Instant;

        // Typical training dimensions
        let test_configs = [
            // (batch, heads, seq_len, head_dim) - description
            (16, 12, 256, 64),   // Standard config (batch=16, heads=12)
            (8, 12, 512, 64),    // Longer sequence
            (32, 8, 128, 64),    // Larger batch, fewer heads
            (4, 16, 256, 64),    // More heads
        ];

        println!("\n{}", "=".repeat(90));
        println!("Attention Backward Benchmark: MPS vs Pipelined Custom Kernels");
        println!("{}", "=".repeat(90));
        println!("{:>6} {:>6} {:>6} {:>6} | {:>12} {:>12} | {:>10}",
                 "batch", "heads", "seq", "dim", "MPS(ms)", "Pipelined(ms)", "Speedup");
        println!("{}", "-".repeat(90));

        let warmup_iters = 3;
        let bench_iters = 10;

        for (batch, heads, seq_len, head_dim) in test_configs {
            // Create test tensors
            let size = batch * heads * seq_len * head_dim;
            let grad_o_data: Vec<f32> = (0..size).map(|i| ((i % 17) as f32) * 0.01 - 0.08).collect();
            let q_data: Vec<f32> = (0..size).map(|i| ((i % 13) as f32) * 0.01 - 0.06).collect();
            let k_data: Vec<f32> = (0..size).map(|i| ((i % 11) as f32) * 0.01 - 0.05).collect();
            let v_data: Vec<f32> = (0..size).map(|i| ((i % 7) as f32) * 0.01 - 0.03).collect();

            let grad_o = Tensor::from_f32_slice(&grad_o_data, &[batch, heads, seq_len, head_dim]);
            let q = Tensor::from_f32_slice(&q_data, &[batch, heads, seq_len, head_dim]);
            let k = Tensor::from_f32_slice(&k_data, &[batch, heads, seq_len, head_dim]);
            let v = Tensor::from_f32_slice(&v_data, &[batch, heads, seq_len, head_dim]);

            // Warmup MPS path
            for _ in 0..warmup_iters {
                let _ = attention_backward_recompute(&grad_o, &q, &k, &v);
            }
            CommandBatch::sync();

            // Benchmark MPS path
            let start = Instant::now();
            for _ in 0..bench_iters {
                let _ = attention_backward_recompute(&grad_o, &q, &k, &v);
            }
            CommandBatch::sync();
            let mps_time = start.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;

            // Warmup pipelined path
            for _ in 0..warmup_iters {
                let _ = attention_backward_pipelined(&grad_o, &q, &k, &v);
            }
            CommandBatch::sync();

            // Benchmark pipelined path
            let start = Instant::now();
            for _ in 0..bench_iters {
                let _ = attention_backward_pipelined(&grad_o, &q, &k, &v);
            }
            CommandBatch::sync();
            let pipelined_time = start.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;

            // Verify correctness (check one result)
            let (mps_gq, mps_gk, mps_gv) = attention_backward_recompute(&grad_o, &q, &k, &v);
            CommandBatch::sync();
            let (pipe_gq, pipe_gk, pipe_gv) = attention_backward_pipelined(&grad_o, &q, &k, &v);
            CommandBatch::sync();

            let max_diff_q = mps_gq.as_f32_slice().iter()
                .zip(pipe_gq.as_f32_slice().iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let max_diff_k = mps_gk.as_f32_slice().iter()
                .zip(pipe_gk.as_f32_slice().iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let max_diff_v = mps_gv.as_f32_slice().iter()
                .zip(pipe_gv.as_f32_slice().iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            let max_diff = max_diff_q.max(max_diff_k).max(max_diff_v);
            let correct = if max_diff < 1e-3 { "✓" } else { "✗" };

            let speedup = mps_time / pipelined_time;

            println!("{:>6} {:>6} {:>6} {:>6} | {:>12.3} {:>12.3} | {:>8.2}x {}",
                     batch, heads, seq_len, head_dim,
                     mps_time, pipelined_time, speedup, correct);
        }

        println!("{}", "=".repeat(90));
        println!("\nNote: Speedup > 1.0 means pipelined is faster");
        println!("      Max diff should be < 1e-3 for numerical equivalence");
    }

    /// Test correctness of pipelined attention backward
    #[test]
    fn test_attention_backward_pipelined_correctness() {
        let batch = 2;
        let heads = 4;
        let seq_len = 16;
        let head_dim = 8;

        let size = batch * heads * seq_len * head_dim;
        let grad_o_data: Vec<f32> = (0..size).map(|i| ((i % 17) as f32) * 0.01 - 0.08).collect();
        let q_data: Vec<f32> = (0..size).map(|i| ((i % 13) as f32) * 0.01 - 0.06).collect();
        let k_data: Vec<f32> = (0..size).map(|i| ((i % 11) as f32) * 0.01 - 0.05).collect();
        let v_data: Vec<f32> = (0..size).map(|i| ((i % 7) as f32) * 0.01 - 0.03).collect();

        let grad_o = Tensor::from_f32_slice(&grad_o_data, &[batch, heads, seq_len, head_dim]);
        let q = Tensor::from_f32_slice(&q_data, &[batch, heads, seq_len, head_dim]);
        let k = Tensor::from_f32_slice(&k_data, &[batch, heads, seq_len, head_dim]);
        let v = Tensor::from_f32_slice(&v_data, &[batch, heads, seq_len, head_dim]);

        let (mps_gq, mps_gk, mps_gv) = attention_backward_recompute(&grad_o, &q, &k, &v);
        CommandBatch::sync();

        let (pipe_gq, pipe_gk, pipe_gv) = attention_backward_pipelined(&grad_o, &q, &k, &v);
        CommandBatch::sync();

        // Check shapes match
        assert_eq!(mps_gq.shape(), pipe_gq.shape());
        assert_eq!(mps_gk.shape(), pipe_gk.shape());
        assert_eq!(mps_gv.shape(), pipe_gv.shape());

        // Check values are close
        let tolerance = 1e-3;

        for (i, (mps, pipe)) in mps_gq.as_f32_slice().iter()
            .zip(pipe_gq.as_f32_slice().iter()).enumerate() {
            assert!(
                (mps - pipe).abs() < tolerance,
                "grad_Q mismatch at {}: MPS={}, pipelined={}, diff={}",
                i, mps, pipe, (mps - pipe).abs()
            );
        }

        for (i, (mps, pipe)) in mps_gk.as_f32_slice().iter()
            .zip(pipe_gk.as_f32_slice().iter()).enumerate() {
            assert!(
                (mps - pipe).abs() < tolerance,
                "grad_K mismatch at {}: MPS={}, pipelined={}, diff={}",
                i, mps, pipe, (mps - pipe).abs()
            );
        }

        for (i, (mps, pipe)) in mps_gv.as_f32_slice().iter()
            .zip(pipe_gv.as_f32_slice().iter()).enumerate() {
            assert!(
                (mps - pipe).abs() < tolerance,
                "grad_V mismatch at {}: MPS={}, pipelined={}, diff={}",
                i, mps, pipe, (mps - pipe).abs()
            );
        }
    }
}
