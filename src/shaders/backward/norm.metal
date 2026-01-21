#include <metal_stdlib>
using namespace metal;

// RMSNorm backward pass
// Forward: y = (x / rms) * gamma, where rms = sqrt(mean(x^2) + eps)
//
// Let s = 1/rms = 1/sqrt(mean(x^2) + eps)
// y_i = x_i * s * gamma_i
//
// Gradients:
// grad_gamma_i = sum_batch(grad_y_i * x_i * s)
// grad_x_i = grad_y_i * gamma_i * s - (1/n) * s^3 * x_i * sum_j(grad_y_j * gamma_j * x_j)

struct RMSNormParams {
    uint batch_seq;
    uint hidden_dim;
    float eps;
};

// RMSNorm backward - computes grad_x and grad_gamma
// Each threadgroup handles one row for grad_x, accumulates to grad_gamma
kernel void rmsnorm_backward_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device float* grad_input [[buffer(3)]],
    device float* grad_gamma [[buffer(4)]],  // Accumulated across batch
    constant RMSNormParams& params [[buffer(5)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= params.batch_seq) return;

    uint offset = row * params.hidden_dim;
    float n = float(params.hidden_dim);

    // Compute sum of squares for rms
    float sum_sq = 0.0f;
    for (uint i = 0; i < params.hidden_dim; i++) {
        float xi = input[offset + i];
        sum_sq += xi * xi;
    }
    float rms = sqrt(sum_sq / n + params.eps);
    float s = 1.0f / rms;  // 1/rms
    float s3 = s * s * s;  // 1/rms^3

    // Compute sum_j(grad_y_j * gamma_j * x_j) for this row
    float dot_sum = 0.0f;
    for (uint i = 0; i < params.hidden_dim; i++) {
        dot_sum += grad_output[offset + i] * gamma[i] * input[offset + i];
    }

    // Compute grad_input and accumulate grad_gamma
    for (uint i = 0; i < params.hidden_dim; i++) {
        float go = grad_output[offset + i];
        float xi = input[offset + i];
        float gi = gamma[i];

        // grad_x_i = grad_y_i * gamma_i * s - (1/n) * s^3 * x_i * dot_sum
        grad_input[offset + i] = go * gi * s - (1.0f / n) * s3 * xi * dot_sum;

        // grad_gamma_i += grad_y_i * x_i * s (atomic add since multiple rows contribute)
        atomic_fetch_add_explicit(
            (device atomic_float*)(&grad_gamma[i]),
            go * xi * s,
            memory_order_relaxed
        );
    }
}

// Separate kernel to zero grad_gamma before accumulation
kernel void zero_buffer_f32(
    device float* buffer [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        buffer[gid] = 0.0f;
    }
}
