#include <metal_stdlib>
using namespace metal;

// Softmax backward pass
// Forward: y_i = exp(x_i - max) / sum(exp(x - max)) = softmax(x)_i
//
// Jacobian: dy_i/dx_j = y_i * (delta_ij - y_j)
// where delta_ij = 1 if i==j else 0
//
// grad_x_i = sum_j(grad_y_j * dy_j/dx_i)
//          = sum_j(grad_y_j * y_j * (delta_ji - y_i))
//          = grad_y_i * y_i - y_i * sum_j(grad_y_j * y_j)
//          = y_i * (grad_y_i - sum_j(grad_y_j * y_j))
//          = y_i * (grad_y_i - dot(grad_y, y))

struct SoftmaxParams {
    uint batch_seq;
    uint dim;
};

// Softmax backward: grad_x = y * (grad_y - dot(grad_y, y))
kernel void softmax_backward_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* output [[buffer(1)]],  // softmax output (y)
    device float* grad_input [[buffer(2)]],
    constant SoftmaxParams& params [[buffer(3)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= params.batch_seq) return;

    uint offset = row * params.dim;

    // Compute dot(grad_y, y)
    float dot_sum = 0.0f;
    for (uint i = 0; i < params.dim; i++) {
        dot_sum += grad_output[offset + i] * output[offset + i];
    }

    // grad_x_i = y_i * (grad_y_i - dot_sum)
    for (uint i = 0; i < params.dim; i++) {
        float yi = output[offset + i];
        float goi = grad_output[offset + i];
        grad_input[offset + i] = yi * (goi - dot_sum);
    }
}

// Fast softmax backward with threadgroup reduction
#define SOFTMAX_THREADS 256

kernel void softmax_backward_fast_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant SoftmaxParams& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    if (row >= params.batch_seq) return;

    threadgroup float shared_sum[SOFTMAX_THREADS];

    uint offset = row * params.dim;

    // Compute local dot product
    float local_sum = 0.0f;
    for (uint i = tid; i < params.dim; i += SOFTMAX_THREADS) {
        local_sum += grad_output[offset + i] * output[offset + i];
    }
    shared_sum[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = SOFTMAX_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float dot_sum = shared_sum[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute gradients
    for (uint i = tid; i < params.dim; i += SOFTMAX_THREADS) {
        float yi = output[offset + i];
        float goi = grad_output[offset + i];
        grad_input[offset + i] = yi * (goi - dot_sum);
    }
}
