#include <metal_stdlib>
using namespace metal;

// ============================================================================
// BF16 Operations for Mixed Precision Training
// ============================================================================
//
// Strategy: Read BF16, compute in FP32, write BF16
// This provides memory savings while maintaining numerical stability.
//
// Metal's bfloat type is available on Apple Silicon (M1+)

// ============================================================================
// GEMM (Matrix Multiplication) - BF16
// ============================================================================

struct GemmParams {
    uint M;
    uint N;
    uint K;
};

// Simple BF16 GEMM: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
kernel void gemm_bf16(
    device const bfloat* A [[buffer(0)]],
    device const bfloat* B [[buffer(1)]],
    device bfloat* C [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;

    if (row >= params.M || col >= params.N) return;

    // Accumulate in FP32 for precision
    float sum = 0.0f;
    for (uint k = 0; k < params.K; k++) {
        sum += float(A[row * params.K + k]) * float(B[k * params.N + col]);
    }

    C[row * params.N + col] = bfloat(sum);
}

// Tiled BF16 GEMM for better memory access patterns
#define TILE_SIZE 16

kernel void gemm_bf16_tiled(
    device const bfloat* A [[buffer(0)]],
    device const bfloat* B [[buffer(1)]],
    device bfloat* C [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    // Shared memory tiles (use float for accumulation precision)
    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;

    float sum = 0.0f;
    uint numTiles = (params.K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TILE_SIZE + tid.x;
        if (row < params.M && aCol < params.K) {
            As[tid.y][tid.x] = float(A[row * params.K + aCol]);
        } else {
            As[tid.y][tid.x] = 0.0f;
        }

        uint bRow = t * TILE_SIZE + tid.y;
        if (bRow < params.K && col < params.N) {
            Bs[tid.y][tid.x] = float(B[bRow * params.N + col]);
        } else {
            Bs[tid.y][tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[tid.y][k] * Bs[k][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < params.M && col < params.N) {
        C[row * params.N + col] = bfloat(sum);
    }
}

// Batched BF16 GEMM
kernel void gemm_bf16_batched(
    device const bfloat* A [[buffer(0)]],
    device const bfloat* B [[buffer(1)]],
    device bfloat* C [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z;
    uint row = gid.y;
    uint col = gid.x;

    if (batch >= batch_size || row >= params.M || col >= params.N) return;

    uint a_offset = batch * params.M * params.K;
    uint b_offset = batch * params.K * params.N;
    uint c_offset = batch * params.M * params.N;

    float sum = 0.0f;
    for (uint k = 0; k < params.K; k++) {
        sum += float(A[a_offset + row * params.K + k]) * float(B[b_offset + k * params.N + col]);
    }

    C[c_offset + row * params.N + col] = bfloat(sum);
}

// ============================================================================
// Element-wise Operations - BF16
// ============================================================================

kernel void add_bf16(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numel) return;
    out[gid] = bfloat(float(a[gid]) + float(b[gid]));
}

kernel void mul_bf16(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numel) return;
    out[gid] = bfloat(float(a[gid]) * float(b[gid]));
}

kernel void scale_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numel) return;
    output[gid] = bfloat(float(input[gid]) * scalar);
}

kernel void silu_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numel) return;
    float x = float(input[gid]);
    float sigmoid = 1.0f / (1.0f + exp(-x));
    output[gid] = bfloat(x * sigmoid);
}

kernel void gelu_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numel) return;
    float x = float(input[gid]);
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3); // sqrt(2/pi) approx
    output[gid] = bfloat(0.5f * x * (1.0f + tanh(inner)));
}

kernel void relu_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numel) return;
    float x = float(input[gid]);
    output[gid] = bfloat(max(0.0f, x));
}

kernel void swiglu_bf16(
    device const bfloat* gate [[buffer(0)]],
    device const bfloat* up [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& numel [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numel) return;
    float g = float(gate[gid]);
    float u = float(up[gid]);
    float sigmoid = 1.0f / (1.0f + exp(-g));
    out[gid] = bfloat(g * sigmoid * u);
}

// ============================================================================
// RMSNorm - BF16
// ============================================================================

struct RMSNormParams {
    uint batch_size;
    uint hidden_dim;
    float eps;
};

kernel void rmsnorm_bf16(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* gamma [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant RMSNormParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint batch_idx = gid.y;
    uint dim_idx = gid.x;

    if (batch_idx >= params.batch_size || dim_idx >= params.hidden_dim) return;

    // Compute mean of squares for this row (in FP32)
    float sum_sq = 0.0f;
    uint row_offset = batch_idx * params.hidden_dim;
    for (uint i = 0; i < params.hidden_dim; i++) {
        float val = float(input[row_offset + i]);
        sum_sq += val * val;
    }

    float rms = sqrt(sum_sq / float(params.hidden_dim) + params.eps);
    float inv_rms = 1.0f / rms;

    // Normalize and scale
    float x = float(input[row_offset + dim_idx]);
    float g = float(gamma[dim_idx]);
    output[row_offset + dim_idx] = bfloat(x * inv_rms * g);
}

// Fast RMSNorm using threadgroup reduction
// Uses 1D grid: one threadgroup per batch element
kernel void rmsnorm_bf16_fast(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* gamma [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant RMSNormParams& params [[buffer(3)]],
    constant uint& tg_size [[buffer(4)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (batch_idx >= params.batch_size) return;

    threadgroup float partial_sum[256];
    uint row_offset = batch_idx * params.hidden_dim;

    // Each thread accumulates part of the sum of squares
    float local_sum = 0.0f;
    for (uint i = tid; i < params.hidden_dim; i += tg_size) {
        float val = float(input[row_offset + i]);
        local_sum += val * val;
    }
    partial_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce within threadgroup
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = sqrt(partial_sum[0] / float(params.hidden_dim) + params.eps);
    float inv_rms = 1.0f / rms;

    // Normalize and scale
    for (uint i = tid; i < params.hidden_dim; i += tg_size) {
        float x = float(input[row_offset + i]);
        float g = float(gamma[i]);
        output[row_offset + i] = bfloat(x * inv_rms * g);
    }
}

// ============================================================================
// Softmax - BF16
// ============================================================================

struct SoftmaxParams {
    uint batch_size;
    uint seq_len;
};

kernel void softmax_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.batch_size) return;

    uint offset = gid * params.seq_len;

    // Find max (in FP32 for stability)
    float max_val = float(input[offset]);
    for (uint i = 1; i < params.seq_len; i++) {
        max_val = max(max_val, float(input[offset + i]));
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < params.seq_len; i++) {
        float val = exp(float(input[offset + i]) - max_val);
        output[offset + i] = bfloat(val); // Temp store
        sum += val;
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (uint i = 0; i < params.seq_len; i++) {
        output[offset + i] = bfloat(float(output[offset + i]) * inv_sum);
    }
}

// ============================================================================
// Precision Conversion Kernels
// ============================================================================

kernel void f32_to_bf16(
    device const float* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numel) return;
    output[gid] = bfloat(input[gid]);
}

kernel void bf16_to_f32(
    device const bfloat* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numel) return;
    output[gid] = float(input[gid]);
}

// ============================================================================
// Lion Optimizer - BF16 Weights with FP32 Momentum
// ============================================================================

struct LionParams {
    float lr;
    float beta1;
    float beta2;
    float weight_decay;
    float lr_scale;
};

// Lion step for BF16 weights, FP32 momentum (mixed precision)
kernel void lion_step_bf16(
    device bfloat* weights [[buffer(0)]],
    device const float* gradients [[buffer(1)]],  // Keep gradients in FP32
    device float* momentum [[buffer(2)]],          // Keep momentum in FP32
    constant LionParams& params [[buffer(3)]],
    constant uint& numel [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= numel) return;

    float w = float(weights[gid]);
    float g = gradients[gid];
    float m = momentum[gid];

    float lr = params.lr * params.lr_scale;

    // Update direction: sign(beta1 * m + (1 - beta1) * g)
    float update = params.beta1 * m + (1.0f - params.beta1) * g;
    float sign_update = (update > 0.0f) ? 1.0f : ((update < 0.0f) ? -1.0f : 0.0f);

    // Weight update with decoupled weight decay
    w = w * (1.0f - lr * params.weight_decay) - lr * sign_update;

    // Update momentum
    m = params.beta2 * m + (1.0f - params.beta2) * g;

    weights[gid] = bfloat(w);
    momentum[gid] = m;
}

// ============================================================================
// Embedding - BF16
// ============================================================================

kernel void embedding_bf16(
    device const bfloat* weights [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant uint& num_indices [[buffer(3)]],
    constant uint& embed_dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint idx = gid.x;
    uint dim = gid.y;

    if (idx >= num_indices || dim >= embed_dim) return;

    uint token_id = indices[idx];
    output[idx * embed_dim + dim] = weights[token_id * embed_dim + dim];
}
