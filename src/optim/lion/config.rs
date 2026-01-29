//! Lion optimizer configuration and parameter state.

use std::io::{Read, Write};

use crate::precision::Precision;
use crate::tensor::Tensor;

/// Momentum state for a parameter tensor.
pub struct ParamState {
    /// Momentum tensor (same shape as parameter).
    pub momentum: Tensor,
}

impl ParamState {
    /// Create a new parameter state with zeroed momentum.
    pub fn new(shape: &[usize]) -> Self {
        Self {
            momentum: Tensor::zeros(shape, Precision::FP32),
        }
    }

    /// Save the parameter state to a writer.
    pub fn save<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        save_tensor(&self.momentum, writer)
    }

    /// Load the parameter state from a reader.
    pub fn load<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let momentum = load_tensor(reader)?;
        Ok(Self { momentum })
    }
}

/// Save a tensor to a writer (FP32 format).
fn save_tensor<W: Write>(tensor: &Tensor, writer: &mut W) -> std::io::Result<()> {
    let shape = tensor.shape();
    let ndim = shape.len() as u32;
    writer.write_all(&ndim.to_le_bytes())?;

    for &dim in shape {
        writer.write_all(&(dim as u64).to_le_bytes())?;
    }

    let data = tensor.as_f32_slice();
    for &val in data {
        writer.write_all(&val.to_le_bytes())?;
    }

    Ok(())
}

/// Load a tensor from a reader (FP32 format).
fn load_tensor<R: Read>(reader: &mut R) -> std::io::Result<Tensor> {
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    reader.read_exact(&mut buf4)?;
    let ndim = u32::from_le_bytes(buf4) as usize;

    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        reader.read_exact(&mut buf8)?;
        shape.push(u64::from_le_bytes(buf8) as usize);
    }

    let numel: usize = shape.iter().product();
    let mut data = Vec::with_capacity(numel);

    for _ in 0..numel {
        reader.read_exact(&mut buf4)?;
        data.push(f32::from_le_bytes(buf4));
    }

    Ok(Tensor::from_f32_slice(&data, &shape))
}

/// Lion optimizer configuration.
#[derive(Clone, Debug)]
pub struct LionConfig {
    /// Learning rate (default: 1e-4).
    pub lr: f32,
    /// Momentum decay for update computation (default: 0.9).
    pub beta1: f32,
    /// Momentum decay for momentum update (default: 0.99).
    pub beta2: f32,
    /// Weight decay coefficient (default: 0.0).
    pub weight_decay: f32,
}

impl Default for LionConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
        }
    }
}

impl LionConfig {
    /// Create a new Lion config with the given learning rate.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            ..Default::default()
        }
    }

    /// Set weight decay.
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Set beta parameters.
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }
}
