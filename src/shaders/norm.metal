#include <metal_stdlib>
using namespace metal;

// RMSNorm: output = (x / sqrt(mean(x^2) + eps)) * gamma
// Input shape: [batch, seq_len, hidden_dim]
// Operates along the last dimension (hidden_dim)

struct RMSNormParams {
    uint batch_seq;     // batch * seq_len (number of rows to normalize)
    uint hidden_dim;    // hidden dimension (length to normalize over)
    float eps;          // epsilon for numerical stability
};

// Note: The simple serial rmsnorm_f32 and the optimized rmsnorm_fast_f32 kernels
// below are the main implementations used. No two-pass approach needed.

// Single-kernel RMSNorm (serial per row, but parallelized across rows)
// This is simpler and works well for moderate hidden dimensions
kernel void rmsnorm_f32(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant RMSNormParams& params [[buffer(3)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= params.batch_seq) return;

    uint offset = row * params.hidden_dim;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (uint i = 0; i < params.hidden_dim; i++) {
        float val = input[offset + i];
        sum_sq += val * val;
    }

    // Compute RMS
    float rms = sqrt(sum_sq / float(params.hidden_dim) + params.eps);
    float inv_rms = 1.0f / rms;

    // Normalize and scale by gamma
    for (uint i = 0; i < params.hidden_dim; i++) {
        output[offset + i] = input[offset + i] * inv_rms * gamma[i];
    }
}

// Optimized RMSNorm using threadgroup reduction
// Each threadgroup processes one row
#define RMSNORM_THREADS 256

kernel void rmsnorm_fast_f32(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant RMSNormParams& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    if (row >= params.batch_seq) return;

    threadgroup float shared_sum[RMSNORM_THREADS];

    uint offset = row * params.hidden_dim;

    // Each thread computes partial sum of squares
    float local_sum = 0.0f;
    for (uint i = tid; i < params.hidden_dim; i += RMSNORM_THREADS) {
        float val = input[offset + i];
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = RMSNORM_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 has the total sum
    float rms = sqrt(shared_sum[0] / float(params.hidden_dim) + params.eps);
    float inv_rms = 1.0f / rms;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All threads normalize their elements
    for (uint i = tid; i < params.hidden_dim; i += RMSNORM_THREADS) {
        output[offset + i] = input[offset + i] * inv_rms * gamma[i];
    }
}

// LayerNorm (for completeness): output = ((x - mean) / sqrt(var + eps)) * gamma + beta
struct LayerNormParams {
    uint batch_seq;
    uint hidden_dim;
    float eps;
};

kernel void layernorm_f32(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant LayerNormParams& params [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= params.batch_seq) return;

    uint offset = row * params.hidden_dim;

    // Compute mean
    float sum = 0.0f;
    for (uint i = 0; i < params.hidden_dim; i++) {
        sum += input[offset + i];
    }
    float mean = sum / float(params.hidden_dim);

    // Compute variance
    float var_sum = 0.0f;
    for (uint i = 0; i < params.hidden_dim; i++) {
        float diff = input[offset + i] - mean;
        var_sum += diff * diff;
    }
    float var = var_sum / float(params.hidden_dim);
    float inv_std = 1.0f / sqrt(var + params.eps);

    // Normalize and scale
    for (uint i = 0; i < params.hidden_dim; i++) {
        output[offset + i] = (input[offset + i] - mean) * inv_std * gamma[i] + beta[i];
    }
}
