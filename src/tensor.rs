use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::device::MetalContext;
use crate::precision::Precision;

pub struct Tensor {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    shape: Vec<usize>,
    precision: Precision,
}

impl Tensor {
    pub fn zeros(shape: &[usize], precision: Precision) -> Self {
        let numel: usize = shape.iter().product();
        let byte_size = numel * precision.byte_size();

        // Metal doesn't allow zero-size buffers, so allocate at least 1 byte
        let alloc_size = byte_size.max(1);

        let ctx = MetalContext::global();
        let buffer = ctx
            .device()
            .newBufferWithLength_options(alloc_size, MTLResourceOptions::StorageModeShared)
            .expect("Failed to allocate Metal buffer");

        // Zero the buffer (unified memory allows direct CPU access)
        if byte_size > 0 {
            unsafe {
                std::ptr::write_bytes(buffer.contents().as_ptr(), 0, byte_size);
            }
        }

        Self {
            buffer,
            shape: shape.to_vec(),
            precision,
        }
    }

    pub fn from_f32_slice(data: &[f32], shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "Data length {} does not match shape {:?} (numel={})",
            data.len(),
            shape,
            numel
        );

        let byte_size = numel * std::mem::size_of::<f32>();
        let ctx = MetalContext::global();

        let buffer = unsafe {
            ctx.device().newBufferWithBytes_length_options(
                NonNull::new(data.as_ptr() as *mut _).unwrap(),
                byte_size,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("Failed to allocate Metal buffer");

        Self {
            buffer,
            shape: shape.to_vec(),
            precision: Precision::FP32,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_size(&self) -> usize {
        self.numel() * self.precision.byte_size()
    }

    pub fn as_f32_slice(&self) -> &[f32] {
        assert_eq!(
            self.precision,
            Precision::FP32,
            "Tensor precision is not FP32"
        );
        unsafe {
            std::slice::from_raw_parts(
                self.buffer.contents().as_ptr() as *const f32,
                self.numel(),
            )
        }
    }

    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        assert_eq!(
            self.precision,
            Precision::FP32,
            "Tensor precision is not FP32"
        );
        unsafe {
            std::slice::from_raw_parts_mut(self.buffer.contents().as_ptr() as *mut f32, self.numel())
        }
    }

    pub fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    /// Reshape tensor in-place (only changes shape metadata, not data)
    pub fn reshape(&mut self, new_shape: &[usize]) {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            self.numel(),
            new_numel,
            "Cannot reshape tensor of {} elements to shape {:?} ({} elements)",
            self.numel(),
            new_shape,
            new_numel
        );
        self.shape = new_shape.to_vec();
    }
}
