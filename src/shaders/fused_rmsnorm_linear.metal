#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Fused RMSNorm + Linear (MatMul with transpose)
// ============================================================================
//
// This kernel fuses RMSNorm normalization with the subsequent linear projection,
// eliminating the intermediate normalized tensor and reducing memory bandwidth.
//
// Computes: output = rmsnorm(input, gamma, eps) @ weight^T
//
// Where:
//   - input: [batch_seq, hidden_dim]
//   - gamma: [hidden_dim]
//   - weight: [out_features, hidden_dim] (transposed during matmul)
//   - output: [batch_seq, out_features]
//
// The normalization is computed as:
//   normalized = input / sqrt(mean(input^2) + eps) * gamma
//
// Then the linear projection:
//   output = normalized @ weight^T
//
// This can be rewritten as:
//   output[i, k] = (1/rms[i]) * sum_j(input[i,j] * gamma[j] * weight[k,j])
//
// So we compute the weighted sum first, then scale by 1/rms.

struct FusedRMSNormLinearParams {
    uint batch_seq;      // Number of rows (batch * seq_len)
    uint hidden_dim;     // Input hidden dimension (K)
    uint out_features;   // Output features (N)
    float eps;           // Epsilon for RMSNorm
};

#define FUSED_THREADS 256
#define TILE_K 32

// Fused RMSNorm + Linear kernel
// Each threadgroup processes one input row and produces out_features outputs
kernel void fused_rmsnorm_linear_f32(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* weight [[buffer(2)]],   // [out_features, hidden_dim]
    device float* output [[buffer(3)]],
    constant FusedRMSNormLinearParams& params [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (row >= params.batch_seq) return;

    uint hidden_dim = params.hidden_dim;
    uint out_features = params.out_features;
    uint input_offset = row * hidden_dim;
    uint output_offset = row * out_features;

    // Shared memory for:
    // 1. Partial sums for RMS computation
    // 2. Cached input * gamma values
    threadgroup float shared_sum[FUSED_THREADS];
    threadgroup float shared_input_gamma[1024];  // Cache for input * gamma (up to 1024 hidden_dim)

    // Step 1: Compute sum of squares and cache input * gamma
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < hidden_dim; i += tg_size) {
        float val = input[input_offset + i];
        local_sum_sq += val * val;

        // Cache input * gamma for reuse in matmul
        if (i < 1024) {
            shared_input_gamma[i] = val * gamma[i];
        }
    }
    shared_sum[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction for sum of squares
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < tg_size) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute inv_rms (all threads read this)
    float rms = sqrt(shared_sum[0] / float(hidden_dim) + params.eps);
    float inv_rms = 1.0f / rms;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Compute matmul output = (input * gamma * inv_rms) @ weight^T
    // Each thread computes multiple output elements
    for (uint out_idx = tid; out_idx < out_features; out_idx += tg_size) {
        float sum = 0.0f;

        // Dot product: sum_j(input[j] * gamma[j] * weight[out_idx, j])
        if (hidden_dim <= 1024) {
            // Use cached values
            for (uint j = 0; j < hidden_dim; j++) {
                sum += shared_input_gamma[j] * weight[out_idx * hidden_dim + j];
            }
        } else {
            // Fallback for large hidden_dim: read from global memory
            for (uint j = 0; j < hidden_dim; j++) {
                float inp_gamma = input[input_offset + j] * gamma[j];
                sum += inp_gamma * weight[out_idx * hidden_dim + j];
            }
        }

        // Scale by inv_rms and write output
        output[output_offset + out_idx] = sum * inv_rms;
    }
}

// Tiled version for better performance with large hidden dimensions
// Uses shared memory tiling for weight matrix
kernel void fused_rmsnorm_linear_tiled_f32(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* weight [[buffer(2)]],   // [out_features, hidden_dim]
    device float* output [[buffer(3)]],
    constant FusedRMSNormLinearParams& params [[buffer(4)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]])
{
    uint row = gid.y;
    uint out_tile = gid.x;
    uint tid = tid3.x;
    uint tg_size = FUSED_THREADS;

    if (row >= params.batch_seq) return;

    uint hidden_dim = params.hidden_dim;
    uint out_features = params.out_features;
    uint input_offset = row * hidden_dim;
    uint output_offset = row * out_features;

    // Tile of outputs this threadgroup is responsible for
    uint out_start = out_tile * FUSED_THREADS;
    uint out_end = min(out_start + FUSED_THREADS, out_features);

    // Shared memory
    threadgroup float shared_sum[FUSED_THREADS];

    // Step 1: Compute sum of squares (each threadgroup does this for its row)
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < hidden_dim; i += tg_size) {
        float val = input[input_offset + i];
        local_sum_sq += val * val;
    }
    shared_sum[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < tg_size) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = sqrt(shared_sum[0] / float(hidden_dim) + params.eps);
    float inv_rms = 1.0f / rms;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Each thread computes one output element
    uint out_idx = out_start + tid;
    if (out_idx < out_end && out_idx < out_features) {
        float sum = 0.0f;

        for (uint j = 0; j < hidden_dim; j++) {
            float inp_gamma = input[input_offset + j] * gamma[j];
            sum += inp_gamma * weight[out_idx * hidden_dim + j];
        }

        output[output_offset + out_idx] = sum * inv_rms;
    }
}
