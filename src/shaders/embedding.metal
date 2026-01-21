#include <metal_stdlib>
using namespace metal;

// Embedding lookup: output[i] = weights[indices[i]]
// weights: [vocab_size, embed_dim]
// indices: [seq_len] or [batch, seq_len]
// output: same shape as indices but with embed_dim appended

struct EmbeddingParams {
    uint num_indices;  // Total number of indices to look up
    uint embed_dim;    // Embedding dimension
};

// Each thread handles one element of the output
kernel void embedding_f32(
    device const float* weights [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant EmbeddingParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint idx_pos = gid.y;  // Which index in the sequence
    uint dim_pos = gid.x;  // Which dimension in the embedding

    if (idx_pos >= params.num_indices || dim_pos >= params.embed_dim) return;

    uint token_id = indices[idx_pos];
    output[idx_pos * params.embed_dim + dim_pos] = weights[token_id * params.embed_dim + dim_pos];
}

// Embedding backward: accumulate gradients into weight gradient
// grad_output: [num_indices, embed_dim]
// indices: [num_indices]
// grad_weights: [vocab_size, embed_dim] - accumulated
kernel void embedding_backward_f32(
    device const float* grad_output [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device float* grad_weights [[buffer(2)]],
    constant EmbeddingParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint idx_pos = gid.y;
    uint dim_pos = gid.x;

    if (idx_pos >= params.num_indices || dim_pos >= params.embed_dim) return;

    uint token_id = indices[idx_pos];
    float grad = grad_output[idx_pos * params.embed_dim + dim_pos];

    // Atomic add since multiple indices might map to the same token
    atomic_fetch_add_explicit(
        (device atomic_float*)(&grad_weights[token_id * params.embed_dim + dim_pos]),
        grad,
        memory_order_relaxed
    );
}
