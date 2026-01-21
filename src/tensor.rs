use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::device::MetalContext;
use crate::precision::{bf16_slice_to_f32, f32_slice_to_bf16, Precision};

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

    // =========================================================================
    // BF16 Support
    // =========================================================================

    /// Create a tensor from BF16 data (stored as u16)
    pub fn from_bf16_slice(data: &[u16], shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "Data length {} does not match shape {:?} (numel={})",
            data.len(),
            shape,
            numel
        );

        let byte_size = numel * std::mem::size_of::<u16>();
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
            precision: Precision::BF16,
        }
    }

    /// Create a BF16 tensor from f32 data (converts automatically)
    pub fn from_f32_as_bf16(data: &[f32], shape: &[usize]) -> Self {
        let bf16_data = f32_slice_to_bf16(data);
        Self::from_bf16_slice(&bf16_data, shape)
    }

    /// Get the raw BF16 data as a u16 slice
    pub fn as_bf16_slice(&self) -> &[u16] {
        assert_eq!(
            self.precision,
            Precision::BF16,
            "Tensor precision is not BF16"
        );
        unsafe {
            std::slice::from_raw_parts(self.buffer.contents().as_ptr() as *const u16, self.numel())
        }
    }

    /// Get mutable access to BF16 data as a u16 slice
    pub fn as_bf16_slice_mut(&mut self) -> &mut [u16] {
        assert_eq!(
            self.precision,
            Precision::BF16,
            "Tensor precision is not BF16"
        );
        unsafe {
            std::slice::from_raw_parts_mut(self.buffer.contents().as_ptr() as *mut u16, self.numel())
        }
    }

    /// Convert a FP32 tensor to BF16
    ///
    /// Creates a new tensor with BF16 precision.
    pub fn to_bf16(&self) -> Self {
        assert_eq!(
            self.precision,
            Precision::FP32,
            "to_bf16() requires FP32 tensor"
        );
        let f32_data = self.as_f32_slice();
        let bf16_data = f32_slice_to_bf16(f32_data);
        Self::from_bf16_slice(&bf16_data, &self.shape)
    }

    /// Convert a BF16 tensor to FP32
    ///
    /// Creates a new tensor with FP32 precision.
    pub fn to_f32(&self) -> Self {
        assert_eq!(
            self.precision,
            Precision::BF16,
            "to_f32() requires BF16 tensor"
        );
        let bf16_data = self.as_bf16_slice();
        let f32_data = bf16_slice_to_f32(bf16_data);
        Self::from_f32_slice(&f32_data, &self.shape)
    }

    /// Check if this tensor is BF16
    pub fn is_bf16(&self) -> bool {
        self.precision == Precision::BF16
    }

    /// Check if this tensor is FP32
    pub fn is_f32(&self) -> bool {
        self.precision == Precision::FP32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision::bf16_to_f32;

    #[test]
    fn test_bf16_tensor_creation() {
        let data = vec![0x3F80u16, 0x4000, 0x4040, 0x4080]; // 1.0, 2.0, 3.0, 4.0 in BF16
        let tensor = Tensor::from_bf16_slice(&data, &[2, 2]);

        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.precision(), Precision::BF16);
        assert!(tensor.is_bf16());
        assert!(!tensor.is_f32());
    }

    #[test]
    fn test_bf16_from_f32() {
        let f32_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_f32_as_bf16(&f32_data, &[4]);

        assert_eq!(tensor.precision(), Precision::BF16);
        let bf16_data = tensor.as_bf16_slice();
        assert_eq!(bf16_data.len(), 4);

        // Convert back and check
        let back: Vec<f32> = bf16_data.iter().map(|&x| bf16_to_f32(x)).collect();
        for (i, (&orig, &conv)) in f32_data.iter().zip(back.iter()).enumerate() {
            assert!(
                (orig - conv).abs() < 0.01,
                "Mismatch at {}: {} vs {}",
                i,
                orig,
                conv
            );
        }
    }

    #[test]
    fn test_bf16_to_f32_conversion() {
        let f32_data = vec![1.5f32, -2.5, 0.0, 100.0];
        let bf16_tensor = Tensor::from_f32_as_bf16(&f32_data, &[4]);
        let f32_tensor = bf16_tensor.to_f32();

        assert_eq!(f32_tensor.precision(), Precision::FP32);
        let result = f32_tensor.as_f32_slice();

        for (i, (&orig, &conv)) in f32_data.iter().zip(result.iter()).enumerate() {
            assert!(
                (orig - conv).abs() < 0.01,
                "Mismatch at {}: {} vs {}",
                i,
                orig,
                conv
            );
        }
    }

    #[test]
    fn test_f32_to_bf16_conversion() {
        let f32_tensor = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let bf16_tensor = f32_tensor.to_bf16();

        assert_eq!(bf16_tensor.precision(), Precision::BF16);
        assert_eq!(bf16_tensor.shape(), &[2, 2]);
    }
}
