//! Shared utilities for Metal kernel dispatch - eliminates boilerplate across ops

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

use crate::command_batch::CommandBatch;
use crate::device::MetalContext;

/// Pipeline registry - compiles shader once, caches pipelines
pub struct PipelineRegistry {
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

impl PipelineRegistry {
    /// Compile a shader source into a library
    pub fn new(shader_source: &str, name: &str) -> Self {
        let ctx = MetalContext::global();
        let library = ctx
            .device()
            .newLibraryWithSource_options_error(&NSString::from_str(shader_source), None)
            .unwrap_or_else(|e| panic!("Failed to compile {name} shader: {e}"));
        Self { library }
    }

    /// Get a compute pipeline for a kernel function
    pub fn pipeline(
        &self,
        func_name: &str,
    ) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        let ctx = MetalContext::global();
        let func = self
            .library
            .newFunctionWithName(&NSString::from_str(func_name))
            .unwrap_or_else(|| panic!("{func_name} function not found"));
        ctx.device()
            .newComputePipelineStateWithFunction_error(&func)
            .unwrap_or_else(|e| panic!("Failed to create {func_name} pipeline: {e}"))
    }
}

/// Macro to define a lazily-initialized pipeline set
#[macro_export]
macro_rules! define_pipelines {
    ($name:ident, $shader:expr, $shader_name:expr, { $($field:ident => $func:expr),+ $(,)? }) => {
        struct $name {
            $($field: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>,)+
        }

        static PIPELINES: std::sync::OnceLock<$name> = std::sync::OnceLock::new();

        fn get_pipelines() -> &'static $name {
            PIPELINES.get_or_init(|| {
                let registry = $crate::ops::kernel::PipelineRegistry::new($shader, $shader_name);
                $name {
                    $($field: registry.pipeline($func),)+
                }
            })
        }
    };
}

/// Create a params buffer from any #[repr(C)] struct
pub fn params_buffer<T>(params: &T) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let ctx = MetalContext::global();
    unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(params as *const T as *mut _).unwrap(),
            std::mem::size_of::<T>(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create params buffer")
}

/// Create a buffer from a slice
pub fn slice_buffer<T>(data: &[T]) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let ctx = MetalContext::global();
    unsafe {
        ctx.device().newBufferWithBytes_length_options(
            NonNull::new(data.as_ptr() as *mut _).unwrap(),
            std::mem::size_of_val(data),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("Failed to create buffer")
}

/// Compute optimal threadgroup size for a 2D grid
pub fn threadgroup_2d(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    width: usize,
    height: usize,
) -> (MTLSize, MTLSize) {
    let thread_width = pipeline.threadExecutionWidth() as usize;
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;

    let grid = MTLSize {
        width,
        height,
        depth: 1,
    };
    let threadgroup = MTLSize {
        width: thread_width.min(width).max(1),
        height: (max_threads / thread_width).min(height).max(1),
        depth: 1,
    };
    (grid, threadgroup)
}

/// Compute optimal threadgroup size for a 1D grid
pub fn threadgroup_1d(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    count: usize,
) -> (MTLSize, MTLSize) {
    let thread_width = pipeline.threadExecutionWidth() as usize;

    let grid = MTLSize {
        width: count,
        height: 1,
        depth: 1,
    };
    let threadgroup = MTLSize {
        width: thread_width.min(count).max(1),
        height: 1,
        depth: 1,
    };
    (grid, threadgroup)
}

/// Compute optimal threadgroup size for a 3D grid
pub fn threadgroup_3d(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    width: usize,
    height: usize,
    depth: usize,
) -> (MTLSize, MTLSize) {
    let thread_width = pipeline.threadExecutionWidth() as usize;
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;

    let grid = MTLSize {
        width,
        height,
        depth,
    };

    // Distribute threads across dimensions
    let tg_width = thread_width.min(width).max(1);
    let remaining = max_threads / tg_width;
    let tg_height = remaining.min(height).max(1);
    let tg_depth = (remaining / tg_height).min(depth).max(1);

    let threadgroup = MTLSize {
        width: tg_width,
        height: tg_height,
        depth: tg_depth,
    };
    (grid, threadgroup)
}

/// Buffer binding for dispatch - can be a Tensor buffer or a raw MTLBuffer
pub enum BufferBinding<'a> {
    Tensor(&'a crate::tensor::Tensor),
    Raw(&'a Retained<ProtocolObject<dyn MTLBuffer>>),
}

impl<'a> From<&'a crate::tensor::Tensor> for BufferBinding<'a> {
    fn from(t: &'a crate::tensor::Tensor) -> Self {
        BufferBinding::Tensor(t)
    }
}

impl<'a> From<&'a Retained<ProtocolObject<dyn MTLBuffer>>> for BufferBinding<'a> {
    fn from(b: &'a Retained<ProtocolObject<dyn MTLBuffer>>) -> Self {
        BufferBinding::Raw(b)
    }
}

/// Dispatch helper that binds buffers in order
pub fn dispatch<'a>(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: impl IntoIterator<Item = BufferBinding<'a>>,
    grid: MTLSize,
    threadgroup: MTLSize,
) {
    let buffers: Vec<_> = buffers.into_iter().collect();
    CommandBatch::dispatch(
        pipeline,
        |encoder| unsafe {
            for (i, binding) in buffers.iter().enumerate() {
                match binding {
                    BufferBinding::Tensor(t) => {
                        encoder.setBuffer_offset_atIndex(Some(t.buffer()), 0, i);
                    }
                    BufferBinding::Raw(b) => {
                        encoder.setBuffer_offset_atIndex(Some(b), 0, i);
                    }
                }
            }
        },
        grid,
        threadgroup,
    );
}

/// Dispatch with threadgroups (for kernels using threadgroup memory)
pub fn dispatch_threadgroups<'a>(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: impl IntoIterator<Item = BufferBinding<'a>>,
    threadgroup_count: MTLSize,
    threadgroup_size: MTLSize,
) {
    let buffers: Vec<_> = buffers.into_iter().collect();
    CommandBatch::dispatch_threadgroups(
        pipeline,
        |encoder| unsafe {
            for (i, binding) in buffers.iter().enumerate() {
                match binding {
                    BufferBinding::Tensor(t) => {
                        encoder.setBuffer_offset_atIndex(Some(t.buffer()), 0, i);
                    }
                    BufferBinding::Raw(b) => {
                        encoder.setBuffer_offset_atIndex(Some(b), 0, i);
                    }
                }
            }
        },
        threadgroup_count,
        threadgroup_size,
    );
}
