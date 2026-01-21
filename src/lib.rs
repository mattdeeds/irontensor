pub mod data;
pub mod device;
pub mod nn;
pub mod ops;
pub mod optim;
pub mod precision;
pub mod tensor;

pub use data::{DatasetIterator, TokenDataset};
pub use device::MetalContext;
pub use nn::{
    FeedForward, FeedForwardState, GPTModel, GPTModelState, Linear, LinearState, ModelConfig,
    MultiHeadAttention, MultiHeadAttentionState, TransformerBlock, TransformerBlockState,
};
pub use ops::{
    // Forward ops
    add, add_scalar, attention, embedding, gelu, matmul, mul, relu, rmsnorm, rope, rope_default,
    scale, silu, softmax, swiglu,
    // Optimized ops (Phase 5)
    flash_attention, fused_linear_cross_entropy, fused_linear_cross_entropy_forward_only,
    // Backward ops
    cross_entropy, cross_entropy_backward, cross_entropy_fused, embedding_backward, gelu_backward,
    matmul_backward, matmul_backward_a, matmul_backward_b, mul_backward, relu_backward,
    rmsnorm_backward, rope_backward, scale_backward, silu_backward, softmax_backward,
    swiglu_backward,
};
pub use optim::{clip_grad_norm, grad_norm, zero_gradients, Lion, LionConfig, ParamState};
pub use precision::Precision;
pub use tensor::Tensor;
