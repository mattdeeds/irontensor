use crate::ops::{matmul, matmul_mps_nt, matmul_mps_tn, softmax, softmax_backward, to_f32_gpu};
use crate::precision::Precision;
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

/// Scale tensor by a scalar
pub(crate) fn scale_tensor(t: &Tensor, scale: f32) -> Tensor {
    let data = t.as_f32_slice();
    let result: Vec<f32> = data.iter().map(|x| x * scale).collect();
    Tensor::from_f32_slice(&result, t.shape())
}

/// Add two tensors element-wise
pub(crate) fn add_tensors(a: &Tensor, b: &Tensor) -> Tensor {
    let a_data = a.as_f32_slice();
    let b_data = b.as_f32_slice();

    // Handle shape mismatch by using the smaller size
    let len = a_data.len().min(b_data.len());
    let result: Vec<f32> = a_data[..len]
        .iter()
        .zip(b_data[..len].iter())
        .map(|(x, y)| x + y)
        .collect();

    Tensor::from_f32_slice(&result, a.shape())
}

/// Compute total L2 norm of multiple gradient tensors
pub(crate) fn compute_total_grad_norm(grads: &[&Tensor]) -> f32 {
    let mut sum_sq = 0.0f32;
    for g in grads {
        for &val in g.as_f32_slice() {
            sum_sq += val * val;
        }
    }
    sum_sq.sqrt()
}

/// Linear forward: output = input @ weight.T
///
/// Uses MPS's native transpose support to avoid explicit transposition.
/// Supports mixed precision: BF16 weights are converted to FP32 for compute.
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
pub(crate) fn linear_backward(
    grad_output: &Tensor,
    input: &Tensor,
    weight: &Tensor,
) -> (Tensor, Tensor) {
    // Convert weight to FP32 if needed for mixed precision
    let weight_fp32 = ensure_fp32(weight);

    // grad_input = grad_output @ weight
    let grad_input = matmul(grad_output, &weight_fp32);

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

/// Repeat KV heads for GQA (Grouped Query Attention)
pub(crate) fn repeat_kv(
    x: &Tensor,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Tensor {
    let repeats = num_heads / num_kv_heads;
    let data = x.as_f32_slice();
    let mut expanded = vec![0.0f32; batch * seq_len * num_heads * head_dim];
    for b in 0..batch {
        for s in 0..seq_len {
            for kv_h in 0..num_kv_heads {
                for r in 0..repeats {
                    let h = kv_h * repeats + r;
                    for d in 0..head_dim {
                        let src = b * seq_len * num_kv_heads * head_dim
                            + s * num_kv_heads * head_dim
                            + kv_h * head_dim
                            + d;
                        let dst = b * seq_len * num_heads * head_dim
                            + s * num_heads * head_dim
                            + h * head_dim
                            + d;
                        expanded[dst] = data[src];
                    }
                }
            }
        }
    }
    Tensor::from_f32_slice(&expanded, &[batch, seq_len, num_heads, head_dim])
}

/// Transpose last two dimensions of a 3D tensor: [batch, m, n] -> [batch, n, m]
pub(crate) fn transpose_last_two_dims_3d(t: &Tensor, batch: usize, m: usize, n: usize) -> Tensor {
    let input = t.as_f32_slice();
    let mut output = vec![0.0f32; batch * n * m];

    for b in 0..batch {
        for i in 0..m {
            for j in 0..n {
                output[b * n * m + j * m + i] = input[b * m * n + i * n + j];
            }
        }
    }

    Tensor::from_f32_slice(&output, &[batch, n, m])
}

/// Apply causal mask to 3D attention scores: [batch, seq, seq]
/// Sets positions where col > row to -inf
pub(crate) fn apply_causal_mask_3d(scores: &Tensor, batch: usize, seq_len: usize) -> Tensor {
    let data = scores.as_f32_slice();
    let mut output = vec![0.0f32; batch * seq_len * seq_len];

    for b in 0..batch {
        for row in 0..seq_len {
            for col in 0..seq_len {
                let idx = b * seq_len * seq_len + row * seq_len + col;
                if col > row {
                    output[idx] = f32::NEG_INFINITY;
                } else {
                    output[idx] = data[idx];
                }
            }
        }
    }

    Tensor::from_f32_slice(&output, &[batch, seq_len, seq_len])
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
    // K^T: [batch*heads, head_dim, seq]
    let k_t = transpose_last_two_dims_3d(&k_flat, batch_heads, seq_len, head_dim);

    // scores = Q @ K^T: [batch*heads, seq, seq]
    let scores = matmul(&q_flat, &k_t);

    // Scale and apply causal mask
    let scores_scaled = scale_tensor(&scores, scale);
    let scores_masked = apply_causal_mask_3d(&scores_scaled, batch_heads, seq_len);

    // Softmax to get attention weights
    let p_flat = softmax(&scores_masked);

    // Now compute gradients using the recomputed P
    // grad_V = P^T @ grad_O
    let p_t = transpose_last_two_dims_3d(&p_flat, batch_heads, seq_len, seq_len);
    let grad_v_flat = matmul(&p_t, &grad_o_flat);

    // grad_P = grad_O @ V^T
    let v_t = transpose_last_two_dims_3d(&v_flat, batch_heads, seq_len, head_dim);
    let grad_p_flat = matmul(&grad_o_flat, &v_t);

    // grad_S = softmax_backward(grad_P, P)
    let grad_s_flat = softmax_backward(&grad_p_flat, &p_flat);

    // grad_Q = grad_S @ K / sqrt(d)
    let grad_q_unscaled = matmul(&grad_s_flat, &k_flat);
    let grad_q_flat = scale_tensor(&grad_q_unscaled, scale);

    // grad_K = grad_S^T @ Q / sqrt(d)
    let grad_s_t = transpose_last_two_dims_3d(&grad_s_flat, batch_heads, seq_len, seq_len);
    let grad_k_unscaled = matmul(&grad_s_t, &q_flat);
    let grad_k_flat = scale_tensor(&grad_k_unscaled, scale);

    // Reshape back to 4D
    let grad_q = grad_q_flat.view(&[batch, heads, seq_len, head_dim]);
    let grad_k = grad_k_flat.view(&[batch, heads, seq_len, head_dim]);
    let grad_v = grad_v_flat.view(&[batch, heads, seq_len, head_dim]);

    (grad_q, grad_k, grad_v)
}

/// Backward for repeat_kv - sums gradients from expanded heads back to KV heads
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
    let repeats = num_heads / num_kv_heads;
    let data = grad_expanded.as_f32_slice();
    let mut grad_kv = vec![0.0f32; batch * seq_len * num_kv_heads * head_dim];

    for b in 0..batch {
        for s in 0..seq_len {
            for kv_h in 0..num_kv_heads {
                for d in 0..head_dim {
                    let dst = b * seq_len * num_kv_heads * head_dim
                        + s * num_kv_heads * head_dim
                        + kv_h * head_dim
                        + d;
                    // Sum gradients from all repeated heads
                    for r in 0..repeats {
                        let h = kv_h * repeats + r;
                        let src = b * seq_len * num_heads * head_dim
                            + s * num_heads * head_dim
                            + h * head_dim
                            + d;
                        grad_kv[dst] += data[src];
                    }
                }
            }
        }
    }
    Tensor::from_f32_slice(&grad_kv, &[batch, seq_len, num_kv_heads, head_dim])
}
