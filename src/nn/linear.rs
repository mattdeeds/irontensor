use crate::ops::{matmul_mps_nt, to_f32_gpu};
use crate::optim::ParamState;
use crate::precision::Precision;
use crate::tensor::Tensor;

/// Add bias to output with broadcasting: output[b, o] += bias[o]
/// This is a relatively small operation compared to matmul, so CPU is acceptable.
fn add_bias_broadcast(output: &Tensor, bias: &Tensor, batch_size: usize, out_features: usize) -> Tensor {
    let output_data = output.as_f32_slice();
    let bias_data = bias.as_f32_slice();
    let mut result = output_data.to_vec();
    for b in 0..batch_size {
        for o in 0..out_features {
            result[b * out_features + o] += bias_data[o];
        }
    }
    Tensor::from_f32_slice(&result, &[batch_size, out_features])
}

/// Linear layer: y = xW^T + b
///
/// For input of shape [..., in_features], produces output of shape [..., out_features]
pub struct Linear {
    /// Weight matrix [out_features, in_features]
    pub weight: Tensor,
    /// Optional bias [out_features]
    pub bias: Option<Tensor>,
    /// Input features
    pub in_features: usize,
    /// Output features
    pub out_features: usize,
}

impl Linear {
    /// Create a new linear layer with Xavier/Glorot initialization
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // Xavier uniform initialization: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
        let limit = (6.0 / (in_features + out_features) as f32).sqrt();
        let weight_data: Vec<f32> = (0..out_features * in_features)
            .map(|i| {
                // Simple deterministic pseudo-random based on index
                let x = ((i as f32 * 0.618033988749895) % 1.0) * 2.0 - 1.0;
                x * limit
            })
            .collect();
        let weight = Tensor::from_f32_slice(&weight_data, &[out_features, in_features]);

        let bias = if bias {
            Some(Tensor::zeros(&[out_features], Precision::FP32))
        } else {
            None
        };

        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Create a linear layer with given weights (for weight tying)
    pub fn from_weight(weight: Tensor, bias: Option<Tensor>) -> Self {
        assert_eq!(weight.shape().len(), 2);
        let out_features = weight.shape()[0];
        let in_features = weight.shape()[1];

        if let Some(ref b) = bias {
            assert_eq!(b.shape(), &[out_features]);
        }

        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Forward pass: y = xW^T + b
    ///
    /// Input shape: [..., in_features]
    /// Output shape: [..., out_features]
    /// Supports mixed precision: BF16 weights/inputs are converted to FP32.
    ///
    /// Uses MPS native transpose for efficient GPU computation.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let input_shape = input.shape();
        assert!(
            !input_shape.is_empty(),
            "Input must have at least one dimension"
        );
        assert_eq!(
            input_shape[input_shape.len() - 1],
            self.in_features,
            "Input last dimension {} doesn't match in_features {}",
            input_shape[input_shape.len() - 1],
            self.in_features
        );

        // Convert BF16 input to FP32 if needed
        let input = if input.precision() == Precision::BF16 {
            to_f32_gpu(input)
        } else {
            input.clone()
        };
        let input_shape = input.shape().to_vec();

        // Reshape input to 2D using zero-copy view: [batch, in_features]
        let batch_size: usize = input_shape[..input_shape.len() - 1].iter().product();
        let batch_size = batch_size.max(1);
        let input_2d = input.view(&[batch_size, self.in_features]);

        // Convert BF16 weight to FP32 if needed
        let weight = if self.weight.precision() == Precision::BF16 {
            to_f32_gpu(&self.weight)
        } else {
            self.weight.clone()
        };

        // Compute xW^T using MPS native transpose: [batch, in] @ [out, in]^T = [batch, out]
        // matmul_mps_nt handles the transpose natively without materializing W^T
        let mut output = matmul_mps_nt(&input_2d, &weight);

        // Add bias if present
        if let Some(ref bias) = self.bias {
            // Convert BF16 bias to FP32 if needed
            let bias = if bias.precision() == Precision::BF16 {
                to_f32_gpu(bias)
            } else {
                bias.clone()
            };
            // Broadcast bias across batch dimension
            output = add_bias_broadcast(&output, &bias, batch_size, self.out_features);
        }

        // Reshape back to original batch dimensions using zero-copy view
        if input_shape.len() > 2 {
            let mut output_shape = input_shape[..input_shape.len() - 1].to_vec();
            output_shape.push(self.out_features);
            output.view(&output_shape)
        } else {
            output
        }
    }

    /// Get parameter count
    pub fn num_params(&self) -> usize {
        let weight_params = self.in_features * self.out_features;
        let bias_params = if self.bias.is_some() {
            self.out_features
        } else {
            0
        };
        weight_params + bias_params
    }

    /// Convert weights to BF16
    pub fn to_bf16(&mut self) {
        if self.weight.is_bf16() {
            return;
        }
        self.weight = self.weight.to_bf16();
        if let Some(ref b) = self.bias {
            self.bias = Some(b.to_bf16());
        }
    }

    /// Convert weights to FP32
    pub fn to_f32(&mut self) {
        if self.weight.is_f32() {
            return;
        }
        self.weight = self.weight.to_f32();
        if let Some(ref b) = self.bias {
            self.bias = Some(b.to_f32());
        }
    }
}

/// Optimizer state for a Linear layer
pub struct LinearState {
    pub weight_state: ParamState,
    pub bias_state: Option<ParamState>,
}

impl LinearState {
    pub fn new(layer: &Linear) -> Self {
        Self {
            weight_state: ParamState::new(layer.weight.shape()),
            bias_state: layer.bias.as_ref().map(|b| ParamState::new(b.shape())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward_2d() {
        let layer = Linear::new(4, 3, false);

        let input = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);

        let output = layer.forward(&input);

        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_linear_forward_3d() {
        let layer = Linear::new(4, 3, false);

        // [batch=2, seq=3, features=4]
        let input_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
        let input = Tensor::from_f32_slice(&input_data, &[2, 3, 4]);

        let output = layer.forward(&input);

        assert_eq!(output.shape(), &[2, 3, 3]);
    }

    #[test]
    fn test_linear_with_bias() {
        // Create layer with known weights for verification
        let weight = Tensor::from_f32_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let bias = Some(Tensor::from_f32_slice(&[1.0, 2.0], &[2]));
        let layer = Linear::from_weight(weight, bias);

        let input = Tensor::from_f32_slice(&[1.0, 2.0], &[1, 2]);
        let output = layer.forward(&input);

        // y = [1,2] @ [[1,0],[0,1]] + [1,2] = [1,2] + [1,2] = [2,4]
        let result = output.as_f32_slice();
        assert!((result[0] - 2.0).abs() < 1e-5);
        assert!((result[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_num_params() {
        let layer = Linear::new(512, 256, true);
        assert_eq!(layer.num_params(), 512 * 256 + 256);

        let layer_no_bias = Linear::new(512, 256, false);
        assert_eq!(layer_no_bias.num_params(), 512 * 256);
    }
}
