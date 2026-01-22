#include <metal_stdlib>
using namespace metal;

// Threadgroup size for reduction operations
#define REDUCTION_THREADGROUP_SIZE 256

// Sum of squares reduction kernel
// Each threadgroup computes partial sum, writes to output[threadgroup_id]
// Final sum is computed on CPU (small number of partial sums)
kernel void sum_squares_f32(
    device const float* input [[buffer(0)]],
    device float* partial_sums [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Shared memory for threadgroup reduction
    threadgroup float shared_data[REDUCTION_THREADGROUP_SIZE];

    // Each thread accumulates sum of squares for its assigned elements
    float local_sum = 0.0f;

    // Grid-stride loop to handle arrays larger than grid size
    uint grid_size = tg_size * ((count + tg_size - 1) / tg_size);
    for (uint i = gid; i < count; i += grid_size) {
        float val = input[i];
        local_sum += val * val;
    }

    // Store to shared memory
    shared_data[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction in shared memory
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < tg_size) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes the partial sum for this threadgroup
    if (tid == 0) {
        partial_sums[tgid] = shared_data[0];
    }
}

// Simple sum reduction kernel (for summing partial results)
// Used when we have a small number of values to sum
kernel void sum_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared_data[REDUCTION_THREADGROUP_SIZE];

    // Each thread loads one element (or 0 if out of bounds)
    float val = (gid < count) ? input[gid] : 0.0f;
    shared_data[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < tg_size) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes result
    if (tid == 0) {
        output[0] = shared_data[0];
    }
}
