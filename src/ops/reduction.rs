//! GPU-accelerated reduction operations for computing norms and sums.
//!
//! # Performance Note (Apple Silicon)
//!
//! On Apple Silicon with unified memory, the CPU-based gradient norm computation
//! in `train/helpers.rs` is actually **faster** than this GPU implementation.
//!
//! Benchmarks showed ~70% slower performance with GPU reduction due to:
//! - Kernel dispatch overhead for many small tensors (37+ gradient tensors)
//! - Extra sync point required to read partial sums
//! - Unified memory allows fast CPU reads (~10 GB/s) without copies
//!
//! This module is kept for:
//! - Potential use on discrete GPUs where CPU-GPU transfers are expensive
//! - Single large tensor reductions where dispatch overhead is amortized
//! - Reference implementation for future optimization attempts

use objc2_metal::MTLSize;

use crate::command_batch::CommandBatch;
use crate::define_pipelines;
use crate::ops::kernel::{dispatch_threadgroups, slice_buffer, BufferBinding};
use crate::precision::Precision;
use crate::profile::{timed, OpCategory};
use crate::tensor::Tensor;

const SHADER: &str = include_str!("../shaders/reduction.metal");
const THREADGROUP_SIZE: usize = 256;

define_pipelines!(Pipelines, SHADER, "reduction", {
    sum_squares => "sum_squares_f32",
});

/// Compute the sum of squares of all elements in a tensor (GPU-accelerated).
///
/// Returns the sum of x^2 for all x in the tensor.
pub fn sum_squares_gpu(input: &Tensor) -> f32 {
    let _timer = timed(OpCategory::GradientClip, input.numel());

    let count = input.numel();
    if count == 0 {
        return 0.0;
    }

    let pipelines = get_pipelines();

    // Calculate number of threadgroups needed
    let num_threadgroups = count.div_ceil(THREADGROUP_SIZE).max(1);

    // Create buffer for partial sums (one per threadgroup)
    let partial_sums = Tensor::zeros(&[num_threadgroups], Precision::FP32);
    let count_buf = slice_buffer(&[count as u32]);

    let threadgroup_size = MTLSize {
        width: THREADGROUP_SIZE,
        height: 1,
        depth: 1,
    };
    let threadgroup_count = MTLSize {
        width: num_threadgroups,
        height: 1,
        depth: 1,
    };

    dispatch_threadgroups(
        &pipelines.sum_squares,
        [
            BufferBinding::from(input),
            BufferBinding::from(&partial_sums),
            BufferBinding::from(&count_buf),
        ],
        threadgroup_count,
        threadgroup_size,
    );

    // Sync to read partial sums
    CommandBatch::sync();

    // Sum partial results on CPU (small number, typically < 1000)
    partial_sums.as_f32_slice().iter().sum()
}

/// Compute the L2 norm of a tensor: sqrt(sum(x^2))
pub fn l2_norm_gpu(input: &Tensor) -> f32 {
    sum_squares_gpu(input).sqrt()
}

/// Compute the total L2 norm of multiple tensors: sqrt(sum of all squared elements)
///
/// This is equivalent to treating all tensors as a single flattened vector
/// and computing its L2 norm.
///
/// Optimized to dispatch all GPU kernels first, then sync once.
pub fn total_l2_norm_gpu(tensors: &[&Tensor]) -> f32 {
    let _timer = timed(
        OpCategory::GradientClip,
        tensors.iter().map(|t| t.numel()).sum(),
    );

    if tensors.is_empty() {
        return 0.0;
    }

    let pipelines = get_pipelines();

    // Pre-allocate partial sum buffers for all tensors
    // We'll dispatch all kernels first, then sync once and read all results
    let mut partial_buffers: Vec<Tensor> = Vec::with_capacity(tensors.len());

    let threadgroup_size = MTLSize {
        width: THREADGROUP_SIZE,
        height: 1,
        depth: 1,
    };

    // Phase 1: Dispatch all GPU kernels without syncing
    for input in tensors {
        let count = input.numel();
        if count == 0 {
            continue;
        }

        let num_threadgroups = count.div_ceil(THREADGROUP_SIZE).max(1);
        let partial_sums = Tensor::zeros(&[num_threadgroups], Precision::FP32);
        let count_buf = slice_buffer(&[count as u32]);

        let threadgroup_count = MTLSize {
            width: num_threadgroups,
            height: 1,
            depth: 1,
        };

        dispatch_threadgroups(
            &pipelines.sum_squares,
            [
                BufferBinding::from(*input),
                BufferBinding::from(&partial_sums),
                BufferBinding::from(&count_buf),
            ],
            threadgroup_count,
            threadgroup_size,
        );

        // Store buffer for later reading (after sync)
        partial_buffers.push(partial_sums);
    }

    // Phase 2: Single sync to wait for all kernels to complete
    CommandBatch::sync();

    // Phase 3: Read all partial sums and compute final result on CPU
    let mut total_sum_sq = 0.0f32;
    for partial_sums in &partial_buffers {
        total_sum_sq += partial_sums.as_f32_slice().iter().sum::<f32>();
    }

    total_sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_squares_simple() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_f32_slice(&data, &[4]);
        let result = sum_squares_gpu(&tensor);
        let expected: f32 = data.iter().map(|x| x * x).sum();
        assert!(
            (result - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_sum_squares_large() {
        let data: Vec<f32> = (0..10000).map(|i| (i as f32) * 0.01).collect();
        let tensor = Tensor::from_f32_slice(&data, &[10000]);
        let result = sum_squares_gpu(&tensor);
        let expected: f32 = data.iter().map(|x| x * x).sum();
        assert!(
            (result - expected).abs() / expected < 1e-4,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_l2_norm() {
        let data = vec![3.0f32, 4.0];
        let tensor = Tensor::from_f32_slice(&data, &[2]);
        let result = l2_norm_gpu(&tensor);
        assert!(
            (result - 5.0).abs() < 1e-5,
            "Expected 5.0, got {}",
            result
        );
    }

    #[test]
    fn test_total_l2_norm_multiple_tensors() {
        let data1 = vec![1.0f32, 2.0, 3.0];
        let data2 = vec![4.0f32, 5.0];
        let t1 = Tensor::from_f32_slice(&data1, &[3]);
        let t2 = Tensor::from_f32_slice(&data2, &[2]);

        let result = total_l2_norm_gpu(&[&t1, &t2]);

        // Expected: sqrt(1 + 4 + 9 + 16 + 25) = sqrt(55)
        let expected = 55.0f32.sqrt();
        assert!(
            (result - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_empty_tensor() {
        let tensor = Tensor::zeros(&[0], Precision::FP32);
        let result = sum_squares_gpu(&tensor);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_empty_tensor_list() {
        let result = total_l2_norm_gpu(&[]);
        assert_eq!(result, 0.0);
    }
}
