//! Error types for IronTensor operations.

use std::fmt;

/// Error type for tensor operations.
#[derive(Debug, Clone)]
pub enum TensorError {
    /// Shape mismatch between tensors.
    ShapeMismatch {
        operation: &'static str,
        expected: String,
        got: String,
    },
    /// Unsupported tensor dimensions for operation.
    UnsupportedDimensions {
        operation: &'static str,
        expected: &'static str,
        got: Vec<usize>,
    },
    /// Precision mismatch between tensors.
    PrecisionMismatch {
        operation: &'static str,
        expected: &'static str,
        got: &'static str,
    },
    /// Inner dimensions don't match for matrix multiplication.
    InnerDimensionMismatch {
        operation: &'static str,
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
    },
    /// Empty tensor where non-empty required.
    EmptyTensor {
        operation: &'static str,
    },
    /// Invalid value for operation.
    InvalidValue {
        operation: &'static str,
        message: String,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch { operation, expected, got } => {
                write!(f, "{}: shape mismatch, expected {}, got {}", operation, expected, got)
            }
            TensorError::UnsupportedDimensions { operation, expected, got } => {
                write!(f, "{}: unsupported dimensions, expected {}, got {:?}", operation, expected, got)
            }
            TensorError::PrecisionMismatch { operation, expected, got } => {
                write!(f, "{}: precision mismatch, expected {}, got {}", operation, expected, got)
            }
            TensorError::InnerDimensionMismatch { operation, a_shape, b_shape } => {
                write!(f, "{}: inner dimensions don't match, A={:?}, B={:?}", operation, a_shape, b_shape)
            }
            TensorError::EmptyTensor { operation } => {
                write!(f, "{}: empty tensor not supported", operation)
            }
            TensorError::InvalidValue { operation, message } => {
                write!(f, "{}: {}", operation, message)
            }
        }
    }
}

impl std::error::Error for TensorError {}

/// Result type alias for tensor operations.
pub type TensorResult<T> = Result<T, TensorError>;
