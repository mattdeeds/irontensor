use crate::ops::{matmul, transpose_2d};
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
