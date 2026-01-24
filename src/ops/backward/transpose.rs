use crate::ops::{transpose_2d, transpose_for_attention, transpose_from_attention};
use crate::profile::{timed, OpCategory};
use crate::tensor::Tensor;

/// Backward for transpose_2d - transpose is its own inverse for 2D tensors
pub fn transpose_2d_backward(grad_output: &Tensor) -> Tensor {
    let _timer = timed(OpCategory::TransposeBackward, grad_output.numel());
    transpose_2d(grad_output)
}

/// Backward for transpose_for_attention
///
/// Forward: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
/// Backward: [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim]
pub fn transpose_for_attention_backward(
    grad_output: &Tensor,
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> Tensor {
    let _timer = timed(OpCategory::TransposeBackward, grad_output.numel());
    // The backward of transpose_for_attention is transpose_from_attention
    transpose_from_attention(grad_output, batch, seq, heads, head_dim)
}

/// Backward for transpose_from_attention
///
/// Forward: [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim]
/// Backward: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
pub fn transpose_from_attention_backward(grad_output: &Tensor) -> Tensor {
    let _timer = timed(OpCategory::TransposeBackward, grad_output.numel());
    // The backward of transpose_from_attention is transpose_for_attention
    transpose_for_attention(grad_output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_2d_backward() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let grad = Tensor::from_f32_slice(&data, &[2, 3]);

        // Forward: [2, 3] -> [3, 2]
        let transposed = transpose_2d(&grad);
        assert_eq!(transposed.shape(), &[3, 2]);

        // Backward: [3, 2] -> [2, 3]
        let grad_back = transpose_2d_backward(&transposed);
        assert_eq!(grad_back.shape(), &[2, 3]);

        // Should recover original
        let result = grad_back.as_f32_slice();
        assert_eq!(result, &data);
    }

    #[test]
    fn test_transpose_for_attention_backward() {
        let batch = 2;
        let seq = 4;
        let heads = 3;
        let head_dim = 2;

        let data: Vec<f32> = (0..(batch * seq * heads * head_dim)).map(|i| i as f32).collect();
        let input = Tensor::from_f32_slice(&data, &[batch, seq, heads, head_dim]);

        // Forward: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        let transposed = transpose_for_attention(&input);
        assert_eq!(transposed.shape(), &[batch, heads, seq, head_dim]);

        // Backward: [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim]
        let grad_back = transpose_for_attention_backward(&transposed, batch, seq, heads, head_dim);
        assert_eq!(grad_back.shape(), &[batch, seq, heads, head_dim]);

        // Should recover original
        let result = grad_back.as_f32_slice();
        assert_eq!(result, &data);
    }

    #[test]
    fn test_transpose_from_attention_backward() {
        let batch = 2;
        let seq = 4;
        let heads = 3;
        let head_dim = 2;

        let data: Vec<f32> = (0..(batch * heads * seq * head_dim)).map(|i| i as f32).collect();
        let input = Tensor::from_f32_slice(&data, &[batch, heads, seq, head_dim]);

        // Forward: [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim]
        let transposed = transpose_from_attention(&input, batch, seq, heads, head_dim);
        assert_eq!(transposed.shape(), &[batch, seq, heads, head_dim]);

        // Backward: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        let grad_back = transpose_from_attention_backward(&transposed);
        assert_eq!(grad_back.shape(), &[batch, heads, seq, head_dim]);

        // Should recover original
        let result = grad_back.as_f32_slice();
        assert_eq!(result, &data);
    }
}
