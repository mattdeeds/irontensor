//! Shared parameter structs for forward and backward ops
//!
//! These structs are used by both forward and backward passes to avoid duplication.

#[repr(C)]
pub struct EmbeddingParams {
    pub num_indices: u32,
    pub embed_dim: u32,
}

#[repr(C)]
pub struct RMSNormParams {
    pub batch_seq: u32,
    pub hidden_dim: u32,
    pub eps: f32,
}

#[repr(C)]
pub struct SoftmaxParams {
    pub batch_seq: u32,
    pub dim: u32,
}

#[repr(C)]
pub struct RoPEParams {
    pub batch_size: u32,
    pub seq_len: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub base: f32,
    pub position_offset: u32,
}

#[repr(C)]
pub struct AttentionParams {
    pub batch_size: u32,
    pub seq_len: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub scale: f32,
}

#[repr(C)]
pub struct DropoutParams {
    pub numel: u32,
    pub p: f32,
    pub scale: f32,
    pub seed: u32,
}
