#include <metal_stdlib>
using namespace metal;

// Embedding backward: scatter gradients to weight matrix
// Forward: output[i] = weights[indices[i]]
// Backward: grad_weights[indices[i]] += grad_output[i]

struct EmbeddingParams {
    uint num_indices;
    uint embed_dim;
};

// Embedding backward - accumulate gradients into weight gradient
// Uses atomics since multiple indices might map to same token
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

    // Atomic add since multiple positions might reference same token
    atomic_fetch_add_explicit(
        (device atomic_float*)(&grad_weights[token_id * params.embed_dim + dim_pos]),
        grad,
        memory_order_relaxed
    );
}
