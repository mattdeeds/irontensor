pub mod device;
pub mod ops;
pub mod precision;
pub mod tensor;

pub use device::MetalContext;
pub use ops::{
    add, add_scalar, attention, embedding, gelu, matmul, mul, relu, rmsnorm, rope, rope_default,
    scale, silu, softmax, swiglu,
};
pub use precision::Precision;
pub use tensor::Tensor;
