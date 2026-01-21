#include <metal_stdlib>
using namespace metal;

// RoPE backward pass
// Forward rotation:
// x'[2i] = x[2i] * cos(θ) - x[2i+1] * sin(θ)
// x'[2i+1] = x[2i] * sin(θ) + x[2i+1] * cos(θ)
//
// This is a rotation matrix: [cos, -sin; sin, cos]
// The inverse (transpose for rotation) is: [cos, sin; -sin, cos]
//
// Backward (gradient flows backward through rotation):
// grad_x[2i] = grad_out[2i] * cos(θ) + grad_out[2i+1] * sin(θ)
// grad_x[2i+1] = -grad_out[2i] * sin(θ) + grad_out[2i+1] * cos(θ)

struct RoPEParams {
    uint batch_size;
    uint seq_len;
    uint num_heads;
    uint head_dim;
    float base;
    uint position_offset;
};

kernel void rope_backward_f32(
    device const float* grad_output [[buffer(0)]],
    device float* grad_input [[buffer(1)]],
    constant RoPEParams& params [[buffer(2)]],
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

    // Compute theta for this dimension pair
    float dim_idx = float(pair_idx * 2);
    float theta = 1.0f / pow(params.base, dim_idx / float(params.head_dim));
    float angle = float(position) * theta;

    float cos_angle = cos(angle);
    float sin_angle = sin(angle);

    // Compute offset
    uint offset = batch * params.seq_len * params.num_heads * params.head_dim
                + seq_pos * params.num_heads * params.head_dim
                + head * params.head_dim
                + pair_idx * 2;

    float go0 = grad_output[offset];
    float go1 = grad_output[offset + 1];

    // Apply inverse rotation (transpose of forward rotation matrix)
    grad_input[offset] = go0 * cos_angle + go1 * sin_angle;
    grad_input[offset + 1] = -go0 * sin_angle + go1 * cos_angle;
}
