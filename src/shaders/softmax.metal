#include <metal_stdlib>
using namespace metal;

// Softmax: output_i = exp(input_i - max) / sum(exp(input - max))
// Operates along the last dimension

struct SoftmaxParams {
    uint batch_seq;    // Number of rows (product of all dims except last)
    uint dim;          // Length of dimension to apply softmax over
};

// Simple softmax: one thread per row
// Works well for small dimensions
kernel void softmax_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= params.batch_seq) return;

    uint offset = row * params.dim;

    // Find max for numerical stability
    float max_val = input[offset];
    for (uint i = 1; i < params.dim; i++) {
        max_val = max(max_val, input[offset + i]);
    }

    // Compute exp and sum
    float sum_exp = 0.0f;
    for (uint i = 0; i < params.dim; i++) {
        float exp_val = exp(input[offset + i] - max_val);
        output[offset + i] = exp_val;  // Store exp values temporarily
        sum_exp += exp_val;
    }

    // Normalize
    float inv_sum = 1.0f / sum_exp;
    for (uint i = 0; i < params.dim; i++) {
        output[offset + i] *= inv_sum;
    }
}

// Optimized softmax using threadgroup reduction
// Each threadgroup processes one row
#define SOFTMAX_THREADS 256

kernel void softmax_fast_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    if (row >= params.batch_seq) return;

    threadgroup float shared_max[SOFTMAX_THREADS];
    threadgroup float shared_sum[SOFTMAX_THREADS];

    uint offset = row * params.dim;

    // Phase 1: Find local max
    float local_max = -INFINITY;
    for (uint i = tid; i < params.dim; i += SOFTMAX_THREADS) {
        local_max = max(local_max, input[offset + i]);
    }
    shared_max[tid] = local_max;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce to find global max
    for (uint stride = SOFTMAX_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float global_max = shared_max[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute exp and local sum
    float local_sum = 0.0f;
    for (uint i = tid; i < params.dim; i += SOFTMAX_THREADS) {
        float exp_val = exp(input[offset + i] - global_max);
        output[offset + i] = exp_val;  // Store temporarily
        local_sum += exp_val;
    }
    shared_sum[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce to find global sum
    for (uint stride = SOFTMAX_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float global_sum = shared_sum[0];
    float inv_sum = 1.0f / global_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Normalize
    for (uint i = tid; i < params.dim; i += SOFTMAX_THREADS) {
        output[offset + i] *= inv_sum;
    }
}

// Causal softmax with masking
// Applies softmax only to valid positions (for autoregressive attention)
// mask_pos: position up to which to include (exclusive), -1 means no mask
struct CausalSoftmaxParams {
    uint batch_seq;
    uint dim;
    int mask_pos;  // Position limit for causal masking
};

kernel void softmax_causal_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant CausalSoftmaxParams& params [[buffer(2)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= params.batch_seq) return;

    uint offset = row * params.dim;
    uint valid_len = (params.mask_pos < 0) ? params.dim : min(uint(params.mask_pos), params.dim);

    // Find max for numerical stability (only valid positions)
    float max_val = -INFINITY;
    for (uint i = 0; i < valid_len; i++) {
        max_val = max(max_val, input[offset + i]);
    }

    // Compute exp and sum (only valid positions)
    float sum_exp = 0.0f;
    for (uint i = 0; i < valid_len; i++) {
        float exp_val = exp(input[offset + i] - max_val);
        output[offset + i] = exp_val;
        sum_exp += exp_val;
    }

    // Normalize valid positions
    float inv_sum = 1.0f / sum_exp;
    for (uint i = 0; i < valid_len; i++) {
        output[offset + i] *= inv_sum;
    }

    // Zero out masked positions
    for (uint i = valid_len; i < params.dim; i++) {
        output[offset + i] = 0.0f;
    }
}
