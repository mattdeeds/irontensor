//! Gradient Checkpointing for Memory-Efficient Training
//!
//! Gradient checkpointing (activation checkpointing) trades compute for memory by not storing
//! all intermediate activations during the forward pass. Instead, we store checkpoints at
//! layer boundaries and recompute activations during the backward pass when needed.
//!
//! This implementation focuses on transformer block boundaries, which provides a good balance
//! between memory savings and recomputation overhead.

use crate::tensor::Tensor;

/// Configuration for gradient checkpointing
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Enable gradient checkpointing
    pub enabled: bool,
    /// Checkpoint every N transformer blocks (1 = checkpoint all, 2 = every other, etc.)
    pub checkpoint_every: usize,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            checkpoint_every: 1, // Checkpoint all blocks when enabled
        }
    }
}

impl CheckpointConfig {
    /// Create config with checkpointing enabled
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            checkpoint_every: 1,
        }
    }

    /// Create config with checkpointing at specified interval
    pub fn with_interval(checkpoint_every: usize) -> Self {
        Self {
            enabled: true,
            checkpoint_every: checkpoint_every.max(1),
        }
    }

    /// Check if a given layer should be checkpointed
    pub fn should_checkpoint(&self, layer_idx: usize) -> bool {
        self.enabled && layer_idx.is_multiple_of(self.checkpoint_every)
    }
}

/// Stores checkpointed activations at transformer block boundaries
///
/// During forward pass: Store only inputs at checkpoint boundaries
/// During backward pass: Recompute activations from checkpoints as needed
pub struct GradientCheckpoints {
    /// Stored activations at checkpoint boundaries
    /// Key: layer index, Value: input tensor to that layer
    checkpoints: Vec<Option<Tensor>>,
    /// Configuration
    config: CheckpointConfig,
    /// Number of layers
    num_layers: usize,
}

impl GradientCheckpoints {
    /// Create a new gradient checkpoint store
    pub fn new(num_layers: usize, config: CheckpointConfig) -> Self {
        let mut checkpoints = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            checkpoints.push(None);
        }
        Self {
            checkpoints,
            config,
            num_layers,
        }
    }

    /// Store a checkpoint at the given layer
    pub fn store(&mut self, layer_idx: usize, activation: Tensor) {
        if self.config.should_checkpoint(layer_idx) && layer_idx < self.num_layers {
            self.checkpoints[layer_idx] = Some(activation);
        }
    }

    /// Retrieve a checkpoint at the given layer
    pub fn get(&self, layer_idx: usize) -> Option<&Tensor> {
        if layer_idx < self.num_layers {
            self.checkpoints[layer_idx].as_ref()
        } else {
            None
        }
    }

    /// Clear all checkpoints (call after backward pass)
    pub fn clear(&mut self) {
        for checkpoint in &mut self.checkpoints {
            *checkpoint = None;
        }
    }

    /// Check if checkpointing is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get memory usage estimate (in bytes)
    pub fn memory_bytes(&self) -> usize {
        self.checkpoints
            .iter()
            .filter_map(|c| c.as_ref())
            .map(|t| t.byte_size())
            .sum()
    }

    /// Get number of stored checkpoints
    pub fn num_checkpoints(&self) -> usize {
        self.checkpoints.iter().filter(|c| c.is_some()).count()
    }
}

/// Segment information for recomputation during backward pass
#[derive(Debug, Clone)]
pub struct RecomputeSegment {
    /// Start layer index (inclusive)
    pub start_layer: usize,
    /// End layer index (exclusive)
    pub end_layer: usize,
}

impl GradientCheckpoints {
    /// Get segments that need to be recomputed for backward pass
    ///
    /// Returns segments between checkpoints that need forward recomputation
    pub fn get_recompute_segments(&self) -> Vec<RecomputeSegment> {
        let mut segments = Vec::new();
        let mut last_checkpoint = 0;

        for i in 0..self.num_layers {
            if self.checkpoints[i].is_some() {
                if i > last_checkpoint {
                    segments.push(RecomputeSegment {
                        start_layer: last_checkpoint,
                        end_layer: i,
                    });
                }
                last_checkpoint = i;
            }
        }

        // Add final segment
        if last_checkpoint < self.num_layers {
            segments.push(RecomputeSegment {
                start_layer: last_checkpoint,
                end_layer: self.num_layers,
            });
        }

        segments
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_config_default() {
        let config = CheckpointConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.checkpoint_every, 1);
    }

    #[test]
    fn test_checkpoint_config_enabled() {
        let config = CheckpointConfig::enabled();
        assert!(config.enabled);
        assert!(config.should_checkpoint(0));
        assert!(config.should_checkpoint(1));
        assert!(config.should_checkpoint(5));
    }

    #[test]
    fn test_checkpoint_config_interval() {
        let config = CheckpointConfig::with_interval(2);
        assert!(config.enabled);
        assert!(config.should_checkpoint(0)); // 0 % 2 == 0
        assert!(!config.should_checkpoint(1)); // 1 % 2 == 1
        assert!(config.should_checkpoint(2)); // 2 % 2 == 0
        assert!(!config.should_checkpoint(3)); // 3 % 2 == 1
    }

    #[test]
    fn test_gradient_checkpoints_store_retrieve() {
        let config = CheckpointConfig::enabled();
        let mut checkpoints = GradientCheckpoints::new(4, config);

        let tensor = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        checkpoints.store(0, tensor);

        assert!(checkpoints.get(0).is_some());
        assert!(checkpoints.get(1).is_none());
        assert_eq!(checkpoints.num_checkpoints(), 1);
    }

    #[test]
    fn test_gradient_checkpoints_clear() {
        let config = CheckpointConfig::enabled();
        let mut checkpoints = GradientCheckpoints::new(4, config);

        // Create separate tensors for each store
        let tensor1 = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let tensor2 = Tensor::from_f32_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        checkpoints.store(0, tensor1);
        checkpoints.store(2, tensor2);

        assert_eq!(checkpoints.num_checkpoints(), 2);
        checkpoints.clear();
        assert_eq!(checkpoints.num_checkpoints(), 0);
    }

    #[test]
    fn test_recompute_segments() {
        let config = CheckpointConfig::with_interval(2);
        let mut checkpoints = GradientCheckpoints::new(6, config);

        // Simulate storing checkpoints at layers 0, 2, 4
        checkpoints.store(0, Tensor::from_f32_slice(&[1.0], &[1]));
        checkpoints.store(2, Tensor::from_f32_slice(&[2.0], &[1]));
        checkpoints.store(4, Tensor::from_f32_slice(&[3.0], &[1]));

        let segments = checkpoints.get_recompute_segments();

        // Should have segments: [0,2), [2,4), [4,6)
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].start_layer, 0);
        assert_eq!(segments[0].end_layer, 2);
        assert_eq!(segments[1].start_layer, 2);
        assert_eq!(segments[1].end_layer, 4);
        assert_eq!(segments[2].start_layer, 4);
        assert_eq!(segments[2].end_layer, 6);
    }

    #[test]
    fn test_memory_estimate() {
        use crate::precision::Precision;

        let config = CheckpointConfig::enabled();
        let mut checkpoints = GradientCheckpoints::new(4, config);

        // 4 elements * 4 bytes = 16 bytes each
        checkpoints.store(0, Tensor::zeros(&[2, 2], Precision::FP32));
        checkpoints.store(2, Tensor::zeros(&[2, 2], Precision::FP32));

        assert_eq!(checkpoints.memory_bytes(), 32); // 2 tensors * 16 bytes
    }
}
