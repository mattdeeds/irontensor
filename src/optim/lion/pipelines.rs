//! Lion optimizer pipeline management and shader compilation.

use std::ptr::NonNull;
use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions,
};

use crate::device::MetalContext;

const LION_SHADER: &str = include_str!("../../shaders/lion.metal");
const BF16_SHADER: &str = include_str!("../../shaders/bf16_ops.metal");

/// Parameters for FP32 Lion kernels.
#[repr(C)]
pub(super) struct LionParams {
    pub count: u32,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
}

/// Parameters for BF16 Lion kernel (matches bf16_ops.metal layout).
#[repr(C)]
pub(super) struct LionParamsBF16 {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
    pub lr_scale: f32,
}

/// Compiled Metal pipelines for Lion optimizer operations.
pub(super) struct LionPipelines {
    pub lion_step: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub lion_step_scaled: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub lion_step_bf16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub zero_gradients: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub grad_norm_squared: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub grad_clip: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

static LION_PIPELINES: OnceLock<LionPipelines> = OnceLock::new();

/// Get or initialize the Lion pipelines.
pub(super) fn get_pipelines() -> &'static LionPipelines {
    LION_PIPELINES.get_or_init(|| {
        let ctx = MetalContext::global();
        let device = ctx.device();

        let library = device
            .newLibraryWithSource_options_error(ns_string!(LION_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile Lion shader: {e}"));

        let bf16_library = device
            .newLibraryWithSource_options_error(ns_string!(BF16_SHADER), None)
            .unwrap_or_else(|e| panic!("Failed to compile BF16 shader: {e}"));

        let make_pipeline = |lib: &ProtocolObject<dyn MTLLibrary>, name: &str| {
            let func = lib
                .newFunctionWithName(&objc2_foundation::NSString::from_str(name))
                .unwrap_or_else(|| panic!("{} function not found", name));
            device
                .newComputePipelineStateWithFunction_error(&func)
                .unwrap_or_else(|e| panic!("Failed to create {} pipeline: {e}", name))
        };

        LionPipelines {
            lion_step: make_pipeline(&library, "lion_step_f32"),
            lion_step_scaled: make_pipeline(&library, "lion_step_scaled_f32"),
            lion_step_bf16: make_pipeline(&bf16_library, "lion_step_bf16"),
            zero_gradients: make_pipeline(&library, "zero_gradients_f32"),
            grad_norm_squared: make_pipeline(&library, "grad_norm_squared_f32"),
            grad_clip: make_pipeline(&library, "grad_clip_f32"),
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
