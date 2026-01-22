//! Operation categories and training phases for profiling.

use std::fmt;

/// Training phases.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
    Forward,
    Backward,
    Optimizer,
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Phase::Forward => write!(f, "Forward"),
            Phase::Backward => write!(f, "Backward"),
            Phase::Optimizer => write!(f, "Optimizer"),
        }
    }
}

/// Operation categories for profiling.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OpCategory {
    // Forward operations
    Embedding,
    Matmul,
    RmsNorm,
    RoPE,
    Softmax,
    Attention,
    FlashAttention,
    FusedLinearCE,
    Elementwise(String),
    Transpose,

    // Backward operations
    EmbeddingBackward,
    MatmulBackward,
    RmsNormBackward,
    RoPEBackward,
    SoftmaxBackward,
    ElementwiseBackward(String),
    CrossEntropyBackward,
    TransposeBackward,

    // Optimizer operations
    LionStep,
    GradientClip,
    GradientNorm,
    ZeroGradients,
}

impl fmt::Display for OpCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpCategory::Embedding => write!(f, "Embedding"),
            OpCategory::Matmul => write!(f, "Matmul"),
            OpCategory::RmsNorm => write!(f, "RmsNorm"),
            OpCategory::RoPE => write!(f, "RoPE"),
            OpCategory::Softmax => write!(f, "Softmax"),
            OpCategory::Attention => write!(f, "Attention"),
            OpCategory::FlashAttention => write!(f, "FlashAttention"),
            OpCategory::FusedLinearCE => write!(f, "FusedLinearCE"),
            OpCategory::Elementwise(name) => write!(f, "Elementwise({})", name),
            OpCategory::Transpose => write!(f, "Transpose"),

            OpCategory::EmbeddingBackward => write!(f, "EmbeddingBackward"),
            OpCategory::MatmulBackward => write!(f, "MatmulBackward"),
            OpCategory::RmsNormBackward => write!(f, "RmsNormBackward"),
            OpCategory::RoPEBackward => write!(f, "RoPEBackward"),
            OpCategory::SoftmaxBackward => write!(f, "SoftmaxBackward"),
            OpCategory::ElementwiseBackward(name) => write!(f, "ElementwiseBackward({})", name),
            OpCategory::CrossEntropyBackward => write!(f, "CrossEntropyBackward"),
            OpCategory::TransposeBackward => write!(f, "TransposeBackward"),

            OpCategory::LionStep => write!(f, "LionStep"),
            OpCategory::GradientClip => write!(f, "GradientClip"),
            OpCategory::GradientNorm => write!(f, "GradientNorm"),
            OpCategory::ZeroGradients => write!(f, "ZeroGradients"),
        }
    }
}

impl OpCategory {
    /// Get a short name for the operation (used in reports).
    pub fn short_name(&self) -> String {
        match self {
            OpCategory::Elementwise(name) => format!("ew:{}", name),
            OpCategory::ElementwiseBackward(name) => format!("ew_bw:{}", name),
            _ => format!("{}", self),
        }
    }
}
