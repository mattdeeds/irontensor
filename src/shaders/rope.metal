#include <metal_stdlib>
using namespace metal;

// RoPE (Rotary Position Embedding)
// Applies rotation to pairs of dimensions based on position
// x'[2i] = x[2i] * cos(p * θ_i) - x[2i+1] * sin(p * θ_i)
// x'[2i+1] = x[2i] * sin(p * θ_i) + x[2i+1] * cos(p * θ_i)
// where θ_i = 1 / (base^(2i/d))

struct RoPEParams {
    uint batch_size;
    uint seq_len;
    uint num_heads;
    uint head_dim;
    float base;         // Typically 10000.0
    uint position_offset;  // For KV cache continuation
};

// Apply RoPE to input tensor
// Input shape: [batch, seq_len, num_heads, head_dim]
// Output shape: same as input
kernel void rope_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant RoPEParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z / params.num_heads;
    uint head = gid.z % params.num_heads;
    uint seq_pos = gid.y;
    uint pair_idx = gid.x;  // Which pair of dimensions (0 to head_dim/2 - 1)

    if (batch >= params.batch_size ||
        seq_pos >= params.seq_len ||
        pair_idx >= params.head_dim / 2) return;

    // Compute position for this token
    uint position = seq_pos + params.position_offset;

    // Compute theta for this dimension pair
    // θ_i = 1 / (base^(2i/d))
    float dim_idx = float(pair_idx * 2);
    float theta = 1.0f / pow(params.base, dim_idx / float(params.head_dim));
    float angle = float(position) * theta;

    float cos_angle = cos(angle);
    float sin_angle = sin(angle);

    // Compute input offset
    uint offset = batch * params.seq_len * params.num_heads * params.head_dim
                + seq_pos * params.num_heads * params.head_dim
                + head * params.head_dim
                + pair_idx * 2;

    float x0 = input[offset];
    float x1 = input[offset + 1];

    // Apply rotation
    output[offset] = x0 * cos_angle - x1 * sin_angle;
    output[offset + 1] = x0 * sin_angle + x1 * cos_angle;
}

// Precompute cos/sin tables for efficiency
// freqs_cos, freqs_sin: [max_seq_len, head_dim/2]
kernel void rope_precompute_freqs_f32(
    device float* freqs_cos [[buffer(0)]],
    device float* freqs_sin [[buffer(1)]],
    constant uint& max_seq_len [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant float& base [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint pos = gid.y;
    uint pair_idx = gid.x;

    if (pos >= max_seq_len || pair_idx >= head_dim / 2) return;

    float dim_idx = float(pair_idx * 2);
    float theta = 1.0f / pow(base, dim_idx / float(head_dim));
    float angle = float(pos) * theta;

    uint offset = pos * (head_dim / 2) + pair_idx;
    freqs_cos[offset] = cos(angle);
    freqs_sin[offset] = sin(angle);
}

// Apply RoPE using precomputed frequencies
// Input shape: [batch, seq_len, num_heads, head_dim]
// freqs_cos, freqs_sin: [seq_len, head_dim/2]
kernel void rope_with_freqs_f32(
    device const float* input [[buffer(0)]],
    device const float* freqs_cos [[buffer(1)]],
    device const float* freqs_sin [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant RoPEParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z / params.num_heads;
    uint head = gid.z % params.num_heads;
    uint seq_pos = gid.y;
    uint pair_idx = gid.x;

    if (batch >= params.batch_size ||
        seq_pos >= params.seq_len ||
        pair_idx >= params.head_dim / 2) return;

    uint position = seq_pos + params.position_offset;

    // Get precomputed cos/sin
    uint freq_offset = position * (params.head_dim / 2) + pair_idx;
    float cos_angle = freqs_cos[freq_offset];
    float sin_angle = freqs_sin[freq_offset];

    // Compute input offset
    uint offset = batch * params.seq_len * params.num_heads * params.head_dim
                + seq_pos * params.num_heads * params.head_dim
                + head * params.head_dim
                + pair_idx * 2;

    float x0 = input[offset];
    float x1 = input[offset + 1];

    // Apply rotation
    output[offset] = x0 * cos_angle - x1 * sin_angle;
    output[offset + 1] = x0 * sin_angle + x1 * cos_angle;
}
