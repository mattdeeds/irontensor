use crate::ops::{matmul, softmax_backward, transpose_2d};
use crate::tensor::Tensor;

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
pub(crate) fn linear_forward(input: &Tensor, weight: &Tensor) -> Tensor {
    // weight: [out, in], need [in, out] for matmul
    let weight_t = transpose_2d(weight);
    matmul(input, &weight_t)
}

/// Linear backward: grad_input = grad_output @ weight, grad_weight = grad_output.T @ input
pub(crate) fn linear_backward(
    grad_output: &Tensor,
    input: &Tensor,
    weight: &Tensor,
) -> (Tensor, Tensor) {
    // grad_input = grad_output @ weight
    let grad_input = matmul(grad_output, weight);

    // grad_weight = grad_output.T @ input
    let grad_weight = matmul_tn(grad_output, input);

    (grad_input, grad_weight)
}

/// Matrix multiply with first operand transposed: A.T @ B
pub(crate) fn matmul_tn(a: &Tensor, b: &Tensor) -> Tensor {
    let a_t = transpose_2d(a);
    matmul(&a_t, b)
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

/// Attention backward pass
///
/// Forward:
///   S = Q @ K^T / sqrt(d)
///   P = softmax(S)  (with causal mask applied before softmax)
///   O = P @ V
///
/// Backward:
///   grad_V = P^T @ grad_O
///   grad_P = grad_O @ V^T
///   grad_S = softmax_backward(grad_P, P)
///   grad_Q = grad_S @ K / sqrt(d)
///   grad_K = grad_S^T @ Q / sqrt(d)
///
/// Inputs:
///   - grad_output: [batch, heads, seq, head_dim]
///   - q: [batch, heads, seq, head_dim]
///   - k: [batch, heads, seq, head_dim]
///   - v: [batch, heads, seq, head_dim]
///   - attn_weights: [batch, heads, seq, seq] (softmax output P)
///
/// Returns: (grad_Q, grad_K, grad_V) all with shape [batch, heads, seq, head_dim]
pub(crate) fn attention_backward(
    grad_output: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attn_weights: &Tensor,
) -> (Tensor, Tensor, Tensor) {
    let q_shape = q.shape();
    let batch = q_shape[0];
    let heads = q_shape[1];
    let seq_len = q_shape[2];
    let head_dim = q_shape[3];
    let batch_heads = batch * heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Reshape to 3D for batched matmul: [batch*heads, seq, dim]
    let grad_o_flat = grad_output.view(&[batch_heads, seq_len, head_dim]);
    let q_flat = q.view(&[batch_heads, seq_len, head_dim]);
    let k_flat = k.view(&[batch_heads, seq_len, head_dim]);
    let v_flat = v.view(&[batch_heads, seq_len, head_dim]);
    let p_flat = attn_weights.view(&[batch_heads, seq_len, seq_len]);

    // grad_V = P^T @ grad_O
    // P: [batch*heads, seq, seq], grad_O: [batch*heads, seq, head_dim]
    // P^T: [batch*heads, seq, seq]
    let p_t = transpose_last_two_dims_3d(&p_flat, batch_heads, seq_len, seq_len);
    let grad_v_flat = matmul(&p_t, &grad_o_flat);

    // grad_P = grad_O @ V^T
    // grad_O: [batch*heads, seq, head_dim], V: [batch*heads, seq, head_dim]
    // V^T: [batch*heads, head_dim, seq]
    let v_t = transpose_last_two_dims_3d(&v_flat, batch_heads, seq_len, head_dim);
    let grad_p_flat = matmul(&grad_o_flat, &v_t);

    // grad_S = softmax_backward(grad_P, P)
    // Both are [batch*heads, seq, seq]
    let grad_s_flat = softmax_backward(&grad_p_flat, &p_flat);

    // grad_Q = grad_S @ K / sqrt(d)
    // grad_S: [batch*heads, seq, seq], K: [batch*heads, seq, head_dim]
    let grad_q_unscaled = matmul(&grad_s_flat, &k_flat);
    let grad_q_flat = scale_tensor(&grad_q_unscaled, scale);

    // grad_K = grad_S^T @ Q / sqrt(d)
    // grad_S^T: [batch*heads, seq, seq], Q: [batch*heads, seq, head_dim]
    let grad_s_t = transpose_last_two_dims_3d(&grad_s_flat, batch_heads, seq_len, seq_len);
    let grad_k_unscaled = matmul(&grad_s_t, &q_flat);
    let grad_k_flat = scale_tensor(&grad_k_unscaled, scale);

    // Reshape back to 4D
    let grad_q = Tensor::from_f32_slice(
        grad_q_flat.as_f32_slice(),
        &[batch, heads, seq_len, head_dim],
    );
    let grad_k = Tensor::from_f32_slice(
        grad_k_flat.as_f32_slice(),
        &[batch, heads, seq_len, head_dim],
    );
    let grad_v = Tensor::from_f32_slice(
        grad_v_flat.as_f32_slice(),
        &[batch, heads, seq_len, head_dim],
    );

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
