mod attention;
pub mod backward;
pub mod bf16_ops;
mod elementwise;
mod embedding;
mod flash_attention;
mod fused_linear_cross_entropy;
mod fused_rmsnorm_linear;
mod gemm;
mod mps_gemm;
mod norm;
mod reduction;
mod repeat_kv;
mod rope;
mod softmax;
mod transpose;

pub use attention::{attention, causal_mask_3d_gpu, transpose_3d_gpu};
pub use backward::{
    cross_entropy, cross_entropy_backward, cross_entropy_fused, embedding_backward, gelu_backward,
    matmul_backward, matmul_backward_a, matmul_backward_b, mul_backward, relu_backward,
    rmsnorm_backward, rope_backward, scale_backward, silu_backward, softmax_backward,
    swiglu_backward, transpose_2d_backward, transpose_for_attention_backward,
    transpose_from_attention_backward,
};
pub use bf16_ops::{
    add_bf16, matmul_bf16, matmul_bf16_batched, mul_bf16, rmsnorm_bf16, scale_bf16, silu_bf16,
    softmax_bf16, swiglu_bf16, to_bf16_gpu, to_f32_gpu,
};
pub use elementwise::{add, add_scalar, gelu, mul, relu, scale, silu, swiglu};
pub use embedding::embedding;
pub use flash_attention::flash_attention;
pub use fused_linear_cross_entropy::{fused_linear_cross_entropy, fused_linear_cross_entropy_forward_only};
pub use fused_rmsnorm_linear::fused_rmsnorm_linear;
pub use gemm::matmul;
pub use mps_gemm::{matmul_mps, matmul_mps_nt, matmul_mps_tn};
pub use norm::rmsnorm;
pub use rope::{rope, rope_default};
pub use softmax::softmax;
pub use reduction::{l2_norm_gpu, sum_squares_gpu, total_l2_norm_gpu};
pub use repeat_kv::{repeat_kv_gpu, repeat_kv_backward_gpu};
pub use transpose::{transpose_2d, transpose_for_attention, transpose_from_attention};
