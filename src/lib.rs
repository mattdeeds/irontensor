/// Golden ratio conjugate (φ - 1 = 1/φ ≈ 0.618), used for low-discrepancy sequences
/// in weight initialization and shuffling algorithms.
pub(crate) const PHI_FRAC: f32 = 0.618_034;

pub mod command_batch;
pub mod data;
pub mod device;
pub mod error;
pub mod logging;
pub mod nn;
pub mod ops;
pub mod optim;
pub mod precision;
pub mod profile;
pub mod rng;
pub mod tensor;
pub mod train;

pub use data::{DatasetIterator, TokenDataset};
pub use device::MetalContext;
pub use nn::{
    FeedForward, FeedForwardState, GPTModel, GPTModelState, Linear, LinearState, ModelConfig,
    MultiHeadAttention, MultiHeadAttentionState, TransformerBlock, TransformerBlockState,
};
pub use ops::{
    // Forward ops (all return TensorResult<Tensor>)
    add, add_scalar, attention, embedding, gelu, matmul, mul, relu, rmsnorm, rope, rope_default,
    scale, silu, softmax, swiglu,
    // Optimized ops (Phase 5)
    flash_attention, fused_linear_cross_entropy, fused_linear_cross_entropy_forward_only,
    // Transpose ops (GPU-accelerated)
    transpose_2d, transpose_for_attention, transpose_from_attention,
    // Backward ops
    cross_entropy, cross_entropy_backward, cross_entropy_fused, embedding_backward, gelu_backward,
    matmul_backward, matmul_backward_a, matmul_backward_b, mul_backward, relu_backward,
    rmsnorm_backward, rope_backward, scale_backward, silu_backward, softmax_backward,
    swiglu_backward, transpose_2d_backward, transpose_for_attention_backward,
    transpose_from_attention_backward,
};
pub use optim::{clip_grad_norm, grad_norm, zero_gradients, Lion, LionConfig, ParamState};
pub use precision::Precision;
pub use tensor::Tensor;
pub use train::{
    load_model_weights, save_model_weights, Checkpoint, ConstantLR, CosineAnnealingLR,
    InverseSqrtLR, LRScheduler, LinearDecayLR, PrintCallback, TrainCallback, TrainMetrics,
    Trainer, TrainingConfig, WarmupConstantLR,
};
pub use profile::{
    OpCategory, Phase, ProfileReport, Profiler, ProfilerConfig,
};
pub use logging::{
    InferenceRecord, InferenceTimer, LogConfig, Logger, ProfileReportRecord, RunLog,
    TrainConfigSnapshot, TrainStepRecord, TrainingLog,
};
pub use command_batch::CommandBatch;
pub use error::{TensorError, TensorResult};
