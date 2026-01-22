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
use crate::profile::{timed, OpCategory};
use crate::tensor::Tensor;

const GEMM_SHADER: &str = include_str!("../shaders/gemm.metal");
const TILE_SIZE: usize = 16;

#[repr(C)]
struct GemmParams {
    m: u32,
    n: u32,
    k: u32,
}

struct GemmPipelines {
    simple: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    tiled: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    batched: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static GEMM_PIPELINES: OnceLock<GemmPipelines> = OnceLock::new();

fn get_pipelines() -> &'static GemmPipelines {
    GEMM_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(GEMM_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile GEMM shader: {e}"));

        let simple_fn = library
            .newFunctionWithName(ns_string!("gemm_f32"))
            .expect("gemm_f32 function not found");
        let tiled_fn = library
            .newFunctionWithName(ns_string!("gemm_f32_tiled"))
            .expect("gemm_f32_tiled function not found");
        let batched_fn = library
            .newFunctionWithName(ns_string!("gemm_f32_batched"))
            .expect("gemm_f32_batched function not found");

        let simple = device
            .newComputePipelineStateWithFunction_error(&simple_fn)
            .expect("Failed to create simple GEMM pipeline");
        let tiled = device
            .newComputePipelineStateWithFunction_error(&tiled_fn)
            .expect("Failed to create tiled GEMM pipeline");
        let batched = device
            .newComputePipelineStateWithFunction_error(&batched_fn)
            .expect("Failed to create batched GEMM pipeline");

        GemmPipelines {
            simple,
            tiled,
            batched,
        }
    })
}

/// Matrix multiplication: C = A @ B
///
/// Supports:
/// - 2D tensors: [M, K] @ [K, N] -> [M, N]
/// - 3D tensors (batched): [B, M, K] @ [B, K, N] -> [B, M, N]
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.precision(), Precision::FP32, "matmul currently only supports FP32");
    assert_eq!(b.precision(), Precision::FP32, "matmul currently only supports FP32");

    let a_shape = a.shape();
    let b_shape = b.shape();

    // Compute output elements for profiling
    let output_elements = match (a_shape.len(), b_shape.len()) {
        (2, 2) => a_shape[0] * b_shape[1],
        (3, 3) => a_shape[0] * a_shape[1] * b_shape[2],
        _ => 0,
    };
    let _timer = timed(OpCategory::Matmul, output_elements);

    match (a_shape.len(), b_shape.len()) {
        (2, 2) => matmul_2d(a, b),
        (3, 3) => matmul_batched(a, b),
        _ => panic!(
            "matmul requires 2D or 3D tensors, got shapes {:?} and {:?}",
            a_shape, b_shape
        ),
    }
}

fn matmul_2d(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();

    let m = a_shape[0];
    let k = a_shape[1];
    let k2 = b_shape[0];
    let n = b_shape[1];

    assert_eq!(k, k2, "Inner dimensions must match: A[{}, {}] @ B[{}, {}]", m, k, k2, n);

    let c = Tensor::zeros(&[m, n], Precision::FP32);

    let ctx = MetalContext::global();
    let pipelines = get_pipelines();

    // Use tiled kernel for larger matrices, simple for small ones
    let use_tiled = m >= TILE_SIZE && n >= TILE_SIZE && k >= TILE_SIZE;

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

    if use_tiled {
        encoder.setComputePipelineState(&pipelines.tiled);
    } else {
        encoder.setComputePipelineState(&pipelines.simple);
    }

    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(c.buffer()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
    }

    if use_tiled {
        let grid_size = MTLSize {
            width: (n + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE,
            height: (m + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE,
            depth: 1,
        };
        let threadgroup_size = MTLSize {
            width: TILE_SIZE,
            height: TILE_SIZE,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    } else {
        let grid_size = MTLSize {
            width: n,
            height: m,
            depth: 1,
        };
        let max_threads = pipelines.simple.maxTotalThreadsPerThreadgroup();
        let thread_width = pipelines.simple.threadExecutionWidth();
        let threadgroup_size = MTLSize {
            width: thread_width.min(n),
            height: (max_threads / thread_width).min(m),
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    }

    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    c
}

fn matmul_batched(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();

    let batch = a_shape[0];
    let m = a_shape[1];
    let k = a_shape[2];
    let batch2 = b_shape[0];
    let k2 = b_shape[1];
    let n = b_shape[2];

    assert_eq!(batch, batch2, "Batch sizes must match: {} vs {}", batch, batch2);
    assert_eq!(k, k2, "Inner dimensions must match: A[{}, {}, {}] @ B[{}, {}, {}]", batch, m, k, batch2, k2, n);

    let c = Tensor::zeros(&[batch, m, n], Precision::FP32);

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

    let batch_size: u32 = batch as u32;
    let batch_buffer = unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(&batch_size as *const _ as *mut _).unwrap(),
            std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create batch buffer");

    let command_buffer = ctx.command_queue().commandBuffer().expect("Failed to create command buffer");
    let encoder = command_buffer.computeCommandEncoder().expect("Failed to create compute encoder");

    encoder.setComputePipelineState(&pipelines.batched);

    unsafe {
        encoder.setBuffer_offset_atIndex(Some(a.buffer()), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(b.buffer()), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(c.buffer()), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&params_buffer), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(&batch_buffer), 0, 4);
    }

    let grid_size = MTLSize {
        width: n,
        height: m,
        depth: batch,
    };
    let max_threads = pipelines.batched.maxTotalThreadsPerThreadgroup();
    let thread_width = pipelines.batched.threadExecutionWidth();
    let threadgroup_size = MTLSize {
        width: thread_width.min(n),
        height: (max_threads / thread_width).min(m),
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_2x2() {
        // A = [[1, 2], [3, 4]]
        // B = [[5, 6], [7, 8]]
        // C = [[19, 22], [43, 50]]
        let a = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_f32_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = matmul(&a, &b);

        let result = c.as_f32_slice();
        assert_eq!(result.len(), 4);
        assert!((result[0] - 19.0).abs() < 1e-5);
        assert!((result[1] - 22.0).abs() < 1e-5);
        assert!((result[2] - 43.0).abs() < 1e-5);
        assert!((result[3] - 50.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_larger() {
        // Test larger matrix to exercise tiled kernel
        let m = 64;
        let k = 32;
        let n = 48;

        // Create identity-like patterns for verification
        let mut a_data = vec![0.0f32; m * k];
        let mut b_data = vec![0.0f32; k * n];

        // Fill with simple values
        for i in 0..m {
            for j in 0..k {
                a_data[i * k + j] = (i + j) as f32 * 0.01;
            }
        }
        for i in 0..k {
            for j in 0..n {
                b_data[i * n + j] = (i * j) as f32 * 0.01;
            }
        }

        let a = Tensor::from_f32_slice(&a_data, &[m, k]);
        let b = Tensor::from_f32_slice(&b_data, &[k, n]);
        let c = matmul(&a, &b);

        // Verify against CPU computation
        let result = c.as_f32_slice();
        for i in 0..m {
            for j in 0..n {
                let mut expected = 0.0f32;
                for kk in 0..k {
                    expected += a_data[i * k + kk] * b_data[kk * n + j];
                }
                let actual = result[i * n + j];
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "Mismatch at [{}, {}]: expected {}, got {}",
                    i, j, expected, actual
                );
            }
        }
    }

    #[test]
    fn test_matmul_batched() {
        let batch = 2;
        let m = 3;
        let k = 4;
        let n = 2;

        // Simple test data
        let a_data: Vec<f32> = (0..(batch * m * k)).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..(batch * k * n)).map(|i| i as f32 * 0.1).collect();

        let a = Tensor::from_f32_slice(&a_data, &[batch, m, k]);
        let b = Tensor::from_f32_slice(&b_data, &[batch, k, n]);
        let c = matmul(&a, &b);

        // Verify against CPU computation
        let result = c.as_f32_slice();
        for bb in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut expected = 0.0f32;
                    for kk in 0..k {
                        expected += a_data[bb * m * k + i * k + kk] * b_data[bb * k * n + kk * n + j];
                    }
                    let actual = result[bb * m * n + i * n + j];
                    assert!(
                        (actual - expected).abs() < 1e-3,
                        "Mismatch at batch {}, [{}, {}]: expected {}, got {}",
                        bb, i, j, expected, actual
                    );
                }
            }
        }
    }
}
