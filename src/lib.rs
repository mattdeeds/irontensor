pub mod device;
pub mod ops;
pub mod precision;
pub mod tensor;

pub use device::MetalContext;
pub use ops::{
    // Forward ops
    add, add_scalar, attention, embedding, gelu, matmul, mul, relu, rmsnorm, rope, rope_default,
    scale, silu, softmax, swiglu,
    // Backward ops
    cross_entropy, cross_entropy_backward, cross_entropy_fused, embedding_backward, gelu_backward,
    matmul_backward, matmul_backward_a, matmul_backward_b, mul_backward, relu_backward,
    rmsnorm_backward, rope_backward, scale_backward, silu_backward, softmax_backward,
    swiglu_backward,
};
pub use precision::Precision;
pub use tensor::Tensor;
