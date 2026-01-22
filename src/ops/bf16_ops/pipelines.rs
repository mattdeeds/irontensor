//! BF16 pipeline management and shader compilation.

use std::ptr::NonNull;
use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions};

use crate::device::MetalContext;

const BF16_SHADER: &str = include_str!("../../shaders/bf16_ops.metal");

pub(super) const TILE_SIZE: usize = 16;

/// Compiled Metal pipelines for BF16 operations.
pub(super) struct BF16Pipelines {
    pub gemm_tiled: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub gemm_batched: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub add: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub mul: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub scale: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub silu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub swiglu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub rmsnorm_fast: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub softmax: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub f32_to_bf16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub bf16_to_f32: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static BF16_PIPELINES: OnceLock<BF16Pipelines> = OnceLock::new();

/// Get or initialize the BF16 pipelines.
pub(super) fn get_pipelines() -> &'static BF16Pipelines {
    BF16_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(BF16_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile BF16 shader: {e}"));

        let make_pipeline = |name: &str| {
            let func = library
                .newFunctionWithName(&objc2_foundation::NSString::from_str(name))
                .unwrap_or_else(|| panic!("{} function not found", name));
            device
                .newComputePipelineStateWithFunction_error(&func)
                .unwrap_or_else(|e| panic!("Failed to create {} pipeline: {e}", name))
        };

        BF16Pipelines {
            gemm_tiled: make_pipeline("gemm_bf16_tiled"),
            gemm_batched: make_pipeline("gemm_bf16_batched"),
            add: make_pipeline("add_bf16"),
            mul: make_pipeline("mul_bf16"),
            scale: make_pipeline("scale_bf16"),
            silu: make_pipeline("silu_bf16"),
            swiglu: make_pipeline("swiglu_bf16"),
            rmsnorm_fast: make_pipeline("rmsnorm_bf16_fast"),
            softmax: make_pipeline("softmax_bf16"),
            f32_to_bf16: make_pipeline("f32_to_bf16"),
            bf16_to_f32: make_pipeline("bf16_to_f32"),
        }
    })
}

/// Create a Metal buffer from a single value.
pub(super) fn create_buffer<T>(
    ctx: &MetalContext,
    data: &T,
) -> Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>> {
    unsafe {
        ctx.device()
            .newBufferWithBytes_length_options(
                NonNull::new(data as *const T as *mut _).unwrap(),
                std::mem::size_of::<T>(),
                MTLResourceOptions::StorageModeShared,
            )
    }
    .expect("Failed to create buffer")
}
