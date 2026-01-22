use std::ptr::NonNull;
use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::device::MetalContext;
use crate::precision::Precision;
use crate::tensor::Tensor;

const BACKWARD_GEMM_SHADER: &str = include_str!("../../shaders/backward/gemm.metal");

#[repr(C)]
struct GemmParams {
    m: u32,
    n: u32,
    k: u32,
}

struct GemmBackwardPipelines {
    grad_a: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    grad_b: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    batched_grad_a: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    batched_grad_b: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static GEMM_BACKWARD_PIPELINES: OnceLock<GemmBackwardPipelines> = OnceLock::new();

fn get_pipelines() -> &'static GemmBackwardPipelines {
    GEMM_BACKWARD_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(BACKWARD_GEMM_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile backward gemm shader: {e}"));

        let make_pipeline = |name: &str| {
            let func = library
                .newFunctionWithName(&objc2_foundation::NSString::from_str(name))
                .unwrap_or_else(|| panic!("{} function not found", name));
            device
                .newComputePipelineStateWithFunction_error(&func)
                .unwrap_or_else(|e| panic!("Failed to create {} pipeline: {e}", name))
        };

        GemmBackwardPipelines {
            grad_a: make_pipeline("gemm_grad_a_f32"),
            grad_b: make_pipeline("gemm_grad_b_f32"),
            batched_grad_a: make_pipeline("gemm_batched_grad_a_f32"),
            batched_grad_b: make_pipeline("gemm_batched_grad_b_f32"),
        }
    })
}

/// Backward pass for matmul: C = A @ B
/// Given grad_C, computes grad_A and grad_B
///
/// For 2D: A[M,K] @ B[K,N] = C[M,N]
/// - grad_A = grad_C @ B^T  (shape [M,K])
/// - grad_B = A^T @ grad_C  (shape [K,N])
///
/// For 3D (batched): same formula applied per batch
pub fn matmul_backward(grad_c: &Tensor, a: &Tensor, b: &Tensor) -> (Tensor, Tensor) {
    assert_eq!(grad_c.precision(), Precision::FP32);
    assert_eq!(a.precision(), Precision::FP32);
    assert_eq!(b.precision(), Precision::FP32);

    match (a.shape().len(), b.shape().len()) {
        (2, 2) => matmul_backward_2d(grad_c, a, b),
        (3, 3) => matmul_backward_batched(grad_c, a, b),
        (4, 4) => matmul_backward_4d(grad_c, a, b),
        _ => panic!(
            "matmul_backward requires 2D, 3D, or 4D tensors, got shapes {:?} and {:?}",
            a.shape(), b.shape()
        ),
    }
}

/// Compute only grad_A from matmul backward
pub fn matmul_backward_a(grad_c: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(grad_c.precision(), Precision::FP32);
    assert_eq!(b.precision(), Precision::FP32);

    let grad_c_shape = grad_c.shape();
    let b_shape = b.shape();

    match (grad_c_shape.len(), b_shape.len()) {
        (2, 2) => {
            let m = grad_c_shape[0];
            let n = grad_c_shape[1];
            let k = b_shape[0];
            assert_eq!(b_shape[1], n);
            compute_grad_a_2d(grad_c, b, m, n, k)
        }
        (3, 3) => {
            let batch = grad_c_shape[0];
            let m = grad_c_shape[1];
            let n = grad_c_shape[2];
            let k = b_shape[1];
            assert_eq!(b_shape[0], batch);
            assert_eq!(b_shape[2], n);
            compute_grad_a_batched(grad_c, b, batch, m, n, k)
        }
        _ => panic!("Unsupported shapes for matmul_backward_a"),
    }
}

/// Compute only grad_B from matmul backward
pub fn matmul_backward_b(grad_c: &Tensor, a: &Tensor) -> Tensor {
    assert_eq!(grad_c.precision(), Precision::FP32);
    assert_eq!(a.precision(), Precision::FP32);

    let grad_c_shape = grad_c.shape();
    let a_shape = a.shape();

    match (grad_c_shape.len(), a_shape.len()) {
        (2, 2) => {
            let m = grad_c_shape[0];
            let n = grad_c_shape[1];
            let k = a_shape[1];
            assert_eq!(a_shape[0], m);
            compute_grad_b_2d(grad_c, a, m, n, k)
        }
        (3, 3) => {
            let batch = grad_c_shape[0];
            let m = grad_c_shape[1];
            let n = grad_c_shape[2];
            let k = a_shape[2];
            assert_eq!(a_shape[0], batch);
            assert_eq!(a_shape[1], m);
            compute_grad_b_batched(grad_c, a, batch, m, n, k)
        }
        _ => panic!("Unsupported shapes for matmul_backward_b"),
    }
}

fn matmul_backward_2d(grad_c: &Tensor, a: &Tensor, b: &Tensor) -> (Tensor, Tensor) {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let grad_c_shape = grad_c.shape();

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    assert_eq!(b_shape[0], k);
    assert_eq!(grad_c_shape, &[m, n]);

    let grad_a = compute_grad_a_2d(grad_c, b, m, n, k);
    let grad_b = compute_grad_b_2d(grad_c, a, m, n, k);

    (grad_a, grad_b)
}

fn compute_grad_a_2d(grad_c: &Tensor, b: &Tensor, m: usize, n: usize, k: usize) -> Tensor {
    let grad_a = Tensor::zeros(&[m, k], Precision::FP32);

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let params = GemmParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
    };
    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<GemmParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create compute encoder");

    encoder.setComputePipelineState(&pipelines.grad_a);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(grad_c.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(grad_a.buffer()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
    }

    let grid_size = MTLSize { width: k, height: m, depth: 1 };
    let thread_width = pipelines.grad_a.threadExecutionWidth();
    let max_threads = pipelines.grad_a.maxTotalThreadsPerThreadgroup();
    let threadgroup_size = MTLSize {
        width: thread_width.min(k),
        height: (max_threads / thread_width).min(m).max(1),
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    grad_a
}

fn compute_grad_b_2d(grad_c: &Tensor, a: &Tensor, m: usize, n: usize, k: usize) -> Tensor {
    let grad_b = Tensor::zeros(&[k, n], Precision::FP32);

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let params = GemmParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
    };
    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<GemmParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create compute encoder");

    encoder.setComputePipelineState(&pipelines.grad_b);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(grad_c.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(grad_b.buffer()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
    }

    let grid_size = MTLSize { width: n, height: k, depth: 1 };
    let thread_width = pipelines.grad_b.threadExecutionWidth();
    let max_threads = pipelines.grad_b.maxTotalThreadsPerThreadgroup();
    let threadgroup_size = MTLSize {
        width: thread_width.min(n),
        height: (max_threads / thread_width).min(k).max(1),
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    grad_b
}

fn matmul_backward_batched(grad_c: &Tensor, a: &Tensor, b: &Tensor) -> (Tensor, Tensor) {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let grad_c_shape = grad_c.shape();

    let batch = a_shape[0];
    let m = a_shape[1];
    let k = a_shape[2];
    let n = b_shape[2];

    assert_eq!(b_shape[0], batch);
    assert_eq!(b_shape[1], k);
    assert_eq!(grad_c_shape, &[batch, m, n]);

    let grad_a = compute_grad_a_batched(grad_c, b, batch, m, n, k);
    let grad_b = compute_grad_b_batched(grad_c, a, batch, m, n, k);

    (grad_a, grad_b)
}

fn compute_grad_a_batched(grad_c: &Tensor, b: &Tensor, batch: usize, m: usize, n: usize, k: usize) -> Tensor {
    let grad_a = Tensor::zeros(&[batch, m, k], Precision::FP32);

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let params = GemmParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
    };
    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<GemmParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    let batch_u32: u32 = batch as u32;
    let batch_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&batch_u32 as *const _ as *mut _).unwrap(),
            std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create batch buffer");

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create compute encoder");

    encoder.setComputePipelineState(&pipelines.batched_grad_a);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(grad_c.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(grad_a.buffer()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(&batch_buffer), 0, 4);
    }

    let grid_size = MTLSize { width: k, height: m, depth: batch };
    let thread_width = pipelines.batched_grad_a.threadExecutionWidth();
    let max_threads = pipelines.batched_grad_a.maxTotalThreadsPerThreadgroup();
    let threadgroup_size = MTLSize {
        width: thread_width.min(k),
        height: (max_threads / thread_width).min(m).max(1),
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    grad_a
}

fn compute_grad_b_batched(grad_c: &Tensor, a: &Tensor, batch: usize, m: usize, n: usize, k: usize) -> Tensor {
    let grad_b = Tensor::zeros(&[batch, k, n], Precision::FP32);

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    let params = GemmParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
    };
    let params_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&params as *const _ as *mut _).unwrap(),
            std::mem::size_of::<GemmParams>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer");

    let batch_u32: u32 = batch as u32;
    let batch_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&batch_u32 as *const _ as *mut _).unwrap(),
            std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create batch buffer");

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create compute encoder");

    encoder.setComputePipelineState(&pipelines.batched_grad_b);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(grad_c.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(grad_b.buffer()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(&batch_buffer), 0, 4);
    }

    let grid_size = MTLSize { width: n, height: k, depth: batch };
    let thread_width = pipelines.batched_grad_b.threadExecutionWidth();
    let max_threads = pipelines.batched_grad_b.maxTotalThreadsPerThreadgroup();
    let threadgroup_size = MTLSize {
        width: thread_width.min(n),
        height: (max_threads / thread_width).min(k).max(1),
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    grad_b
}

/// Backward pass for 4D batched matmul
/// Reshapes to 3D by collapsing first two dims, computes backward, then reshapes back
fn matmul_backward_4d(grad_c: &Tensor, a: &Tensor, b: &Tensor) -> (Tensor, Tensor) {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let grad_c_shape = grad_c.shape();

    // 4D: [batch1, batch2, m, k] @ [batch1, batch2, k, n] = [batch1, batch2, m, n]
    let batch1 = a_shape[0];
    let batch2 = a_shape[1];
    let m = a_shape[2];
    let k = a_shape[3];
    let n = b_shape[3];

    assert_eq!(b_shape[0], batch1);
    assert_eq!(b_shape[1], batch2);
    assert_eq!(b_shape[2], k);
    assert_eq!(grad_c_shape, &[batch1, batch2, m, n]);

    // Collapse first two dims into one batch dimension
    let new_batch = batch1 * batch2;

    let a_3d = Tensor::from_f32_slice(a.as_f32_slice(), &[new_batch, m, k]);
    let b_3d = Tensor::from_f32_slice(b.as_f32_slice(), &[new_batch, k, n]);
    let gc_3d = Tensor::from_f32_slice(grad_c.as_f32_slice(), &[new_batch, m, n]);

    let (ga_3d, gb_3d) = matmul_backward_batched(&gc_3d, &a_3d, &b_3d);

    // Reshape back to 4D
    let grad_a = Tensor::from_f32_slice(ga_3d.as_f32_slice(), a_shape);
    let grad_b = Tensor::from_f32_slice(gb_3d.as_f32_slice(), b_shape);

    (grad_a, grad_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    

    #[test]
    fn test_matmul_backward_2d() {
        let m = 3;
        let k = 4;
        let n = 2;

        let a_data: Vec<f32> = (0..(m * k)).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..(k * n)).map(|i| i as f32 * 0.1).collect();
        let grad_c_data: Vec<f32> = vec![1.0f32; m * n];

        let a = Tensor::from_f32_slice(&a_data, &[m, k]);
        let b = Tensor::from_f32_slice(&b_data, &[k, n]);
        let grad_c = Tensor::from_f32_slice(&grad_c_data, &[m, n]);

        let (grad_a, grad_b) = matmul_backward(&grad_c, &a, &b);

        // Verify shapes
        assert_eq!(grad_a.shape(), &[m, k]);
        assert_eq!(grad_b.shape(), &[k, n]);

        // Verify against CPU reference
        // grad_A[i,j] = sum_l grad_C[i,l] * B[j,l] (B transposed)
        let grad_a_result = grad_a.as_f32_slice();
        for i in 0..m {
            for j in 0..k {
                let mut expected = 0.0f32;
                for l in 0..n {
                    expected += grad_c_data[i * n + l] * b_data[j * n + l];
                }
                assert!(
                    (grad_a_result[i * k + j] - expected).abs() < 1e-4,
                    "grad_A mismatch at [{},{}]: expected {}, got {}",
                    i, j, expected, grad_a_result[i * k + j]
                );
            }
        }

        // grad_B[i,j] = sum_l A[l,i] * grad_C[l,j] (A transposed)
        let grad_b_result = grad_b.as_f32_slice();
        for i in 0..k {
            for j in 0..n {
                let mut expected = 0.0f32;
                for l in 0..m {
                    expected += a_data[l * k + i] * grad_c_data[l * n + j];
                }
                assert!(
                    (grad_b_result[i * n + j] - expected).abs() < 1e-4,
                    "grad_B mismatch at [{},{}]: expected {}, got {}",
                    i, j, expected, grad_b_result[i * n + j]
                );
            }
        }
    }

    #[test]
    fn test_matmul_backward_batched() {
        let batch = 2;
        let m = 3;
        let k = 4;
        let n = 2;

        let a_data: Vec<f32> = (0..(batch * m * k)).map(|i| i as f32 * 0.01).collect();
        let b_data: Vec<f32> = (0..(batch * k * n)).map(|i| i as f32 * 0.01).collect();
        let grad_c_data: Vec<f32> = vec![1.0f32; batch * m * n];

        let a = Tensor::from_f32_slice(&a_data, &[batch, m, k]);
        let b = Tensor::from_f32_slice(&b_data, &[batch, k, n]);
        let grad_c = Tensor::from_f32_slice(&grad_c_data, &[batch, m, n]);

        let (grad_a, grad_b) = matmul_backward(&grad_c, &a, &b);

        assert_eq!(grad_a.shape(), &[batch, m, k]);
        assert_eq!(grad_b.shape(), &[batch, k, n]);

        // Verify first batch against CPU reference
        let grad_a_result = grad_a.as_f32_slice();
        for i in 0..m {
            for j in 0..k {
                let mut expected = 0.0f32;
                for l in 0..n {
                    expected += grad_c_data[i * n + l] * b_data[j * n + l];
                }
                assert!(
                    (grad_a_result[i * k + j] - expected).abs() < 1e-3,
                    "grad_A batch 0 mismatch at [{},{}]",
                    i, j
                );
            }
        }
    }

    #[test]
    fn test_matmul_backward_4d() {
        let batch1 = 2;
        let batch2 = 3;
        let m = 4;
        let k = 5;
        let n = 6;

        let a_data: Vec<f32> = (0..(batch1 * batch2 * m * k))
            .map(|i| (i as f32 * 0.01) % 1.0)
            .collect();
        let b_data: Vec<f32> = (0..(batch1 * batch2 * k * n))
            .map(|i| (i as f32 * 0.01) % 1.0)
            .collect();
        let grad_c_data: Vec<f32> = vec![1.0f32; batch1 * batch2 * m * n];

        let a = Tensor::from_f32_slice(&a_data, &[batch1, batch2, m, k]);
        let b = Tensor::from_f32_slice(&b_data, &[batch1, batch2, k, n]);
        let grad_c = Tensor::from_f32_slice(&grad_c_data, &[batch1, batch2, m, n]);

        let (grad_a, grad_b) = matmul_backward(&grad_c, &a, &b);

        // Verify shapes
        assert_eq!(grad_a.shape(), &[batch1, batch2, m, k]);
        assert_eq!(grad_b.shape(), &[batch1, batch2, k, n]);

        // Verify first element of first batch against CPU reference
        let grad_a_result = grad_a.as_f32_slice();
        // For grad_a[0,0,0,0], we compute sum over n of grad_c[0,0,0,l] * b[0,0,0,l]
        let mut expected = 0.0f32;
        for l in 0..n {
            expected += grad_c_data[l] * b_data[l];
        }
        assert!(
            (grad_a_result[0] - expected).abs() < 1e-3,
            "grad_A[0,0,0,0] mismatch: expected {}, got {}",
            expected, grad_a_result[0]
        );
    }
}
