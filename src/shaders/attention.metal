#include <metal_stdlib>
using namespace metal;

// Attention operations for supporting basic attention implementation
// Full attention will be composed of: matmul, scale, mask (optional), softmax, matmul
// These kernels provide supporting operations

struct AttentionParams {
    uint batch_size;
    uint num_heads;
    uint seq_len;
    uint head_dim;
    float scale;  // 1/sqrt(head_dim)
};

// Scale attention scores: scores = scores * scale
// Input/output shape: [batch, num_heads, seq_len, seq_len]
kernel void attention_scale_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        output[gid] = input[gid] * scale;
    }
}

// Apply causal mask: set future positions to -inf
// Shape: [batch, num_heads, seq_len, seq_len]
// For position (q_pos, k_pos), mask if k_pos > q_pos
kernel void causal_mask_f32(
    device float* scores [[buffer(0)]],
    constant AttentionParams& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch_head = gid.z;  // Flattened batch * num_heads
    uint q_pos = gid.y;
    uint k_pos = gid.x;

    uint batch = batch_head / params.num_heads;
    uint head = batch_head % params.num_heads;

    if (batch >= params.batch_size ||
        q_pos >= params.seq_len ||
        k_pos >= params.seq_len) return;

    // Causal mask: can only attend to positions <= current position
    if (k_pos > q_pos) {
        uint offset = batch * params.num_heads * params.seq_len * params.seq_len
                    + head * params.seq_len * params.seq_len
                    + q_pos * params.seq_len
                    + k_pos;
        scores[offset] = -INFINITY;
    }
}

// Transpose key from [batch, seq, heads, dim] to [batch, heads, dim, seq]
// This is needed for efficient Q @ K^T computation
kernel void transpose_key_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant AttentionParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z / params.num_heads;
    uint head = gid.z % params.num_heads;
    uint dim = gid.y;
    uint seq = gid.x;

    if (batch >= params.batch_size ||
        dim >= params.head_dim ||
        seq >= params.seq_len) return;

    // Input: [batch, seq, heads, dim]
    uint in_offset = batch * params.seq_len * params.num_heads * params.head_dim
                   + seq * params.num_heads * params.head_dim
                   + head * params.head_dim
                   + dim;

    // Output: [batch, heads, dim, seq]
    uint out_offset = batch * params.num_heads * params.head_dim * params.seq_len
                    + head * params.head_dim * params.seq_len
                    + dim * params.seq_len
                    + seq;

    output[out_offset] = input[in_offset];
}

// Transpose from [batch, seq, heads, dim] to [batch, heads, seq, dim]
// Standard attention layout transformation
kernel void transpose_qkv_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant AttentionParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z / params.num_heads;
    uint head = gid.z % params.num_heads;
    uint seq = gid.y;
    uint dim = gid.x;

    if (batch >= params.batch_size ||
        seq >= params.seq_len ||
        dim >= params.head_dim) return;

    // Input: [batch, seq, heads, dim]
    uint in_offset = batch * params.seq_len * params.num_heads * params.head_dim
                   + seq * params.num_heads * params.head_dim
                   + head * params.head_dim
                   + dim;

    // Output: [batch, heads, seq, dim]
    uint out_offset = batch * params.num_heads * params.seq_len * params.head_dim
                    + head * params.seq_len * params.head_dim
                    + seq * params.head_dim
                    + dim;

    output[out_offset] = input[in_offset];
}

// Transpose from [batch, heads, seq, dim] back to [batch, seq, heads, dim]
kernel void transpose_output_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant AttentionParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z / params.num_heads;
    uint head = gid.z % params.num_heads;
    uint seq = gid.y;
    uint dim = gid.x;

    if (batch >= params.batch_size ||
        seq >= params.seq_len ||
        dim >= params.head_dim) return;

    // Input: [batch, heads, seq, dim]
    uint in_offset = batch * params.num_heads * params.seq_len * params.head_dim
                   + head * params.seq_len * params.head_dim
                   + seq * params.head_dim
                   + dim;

    // Output: [batch, seq, heads, dim]
    uint out_offset = batch * params.seq_len * params.num_heads * params.head_dim
                    + seq * params.num_heads * params.head_dim
                    + head * params.head_dim
                    + dim;

    output[out_offset] = input[in_offset];
}

// Parameters for 3D tensor operations
struct Transpose3DParams {
    uint batch;   // Number of batches
    uint m;       // First spatial dimension
    uint n;       // Second spatial dimension
};

// Transpose last two dimensions of 3D tensor: [batch, m, n] -> [batch, n, m]
// Used in attention backward for transposing K, V, gradients, etc.
kernel void transpose_3d_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant Transpose3DParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.z;   // batch index
    uint j = gid.y;   // output row (was column)
    uint i = gid.x;   // output column (was row)

    if (b >= params.batch || j >= params.n || i >= params.m) return;

    // Input: [batch, m, n] at [b, i, j]
    uint in_offset = b * params.m * params.n + i * params.n + j;

    // Output: [batch, n, m] at [b, j, i]
    uint out_offset = b * params.n * params.m + j * params.m + i;

    output[out_offset] = input[in_offset];
}

// Apply causal mask to 3D attention scores: [batch, seq, seq]
// Sets positions where col > row to -inf
kernel void causal_mask_3d_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint b = gid.z;    // batch index
    uint row = gid.y;  // query position
    uint col = gid.x;  // key position

    if (b >= batch || row >= seq_len || col >= seq_len) return;

    uint idx = b * seq_len * seq_len + row * seq_len + col;

    // Causal mask: can only attend to positions <= current position
    if (col > row) {
        output[idx] = -INFINITY;
    } else {
        output[idx] = input[idx];
    }
}
