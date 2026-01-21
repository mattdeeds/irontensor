#include <metal_stdlib>
using namespace metal;

// Backward pass for GEMM (matrix multiplication)
// Forward: C = A @ B where A: [M, K], B: [K, N], C: [M, N]
// Backward: grad_A = grad_C @ B^T, grad_B = A^T @ grad_C

// Note: We reuse the forward GEMM kernels by transposing appropriately
// This file provides specialized kernels if needed for efficiency

struct GemmParams {
    uint M;
    uint N;
    uint K;
};

// For grad_A = grad_C @ B^T
// grad_C: [M, N], B^T: [N, K] -> grad_A: [M, K]
// This is equivalent to grad_C @ transpose(B)
kernel void gemm_grad_a_f32(
    device const float* grad_C [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* grad_A [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;  // M dimension
    uint col = gid.x;  // K dimension

    if (row >= params.M || col >= params.K) return;

    // grad_A[row, col] = sum over n: grad_C[row, n] * B[col, n] (B transposed)
    // Since B is [K, N], B^T is [N, K], so B^T[n, col] = B[col, n]
    float sum = 0.0f;
    for (uint n = 0; n < params.N; n++) {
        sum += grad_C[row * params.N + n] * B[col * params.N + n];
    }

    grad_A[row * params.K + col] = sum;
}

// For grad_B = A^T @ grad_C
// A^T: [K, M], grad_C: [M, N] -> grad_B: [K, N]
kernel void gemm_grad_b_f32(
    device const float* A [[buffer(0)]],
    device const float* grad_C [[buffer(1)]],
    device float* grad_B [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;  // K dimension
    uint col = gid.x;  // N dimension

    if (row >= params.K || col >= params.N) return;

    // grad_B[row, col] = sum over m: A^T[row, m] * grad_C[m, col]
    // A^T[row, m] = A[m, row]
    float sum = 0.0f;
    for (uint m = 0; m < params.M; m++) {
        sum += A[m * params.K + row] * grad_C[m * params.N + col];
    }

    grad_B[row * params.N + col] = sum;
}

// Batched backward for grad_A
// Forward: C[b] = A[b] @ B[b]
// grad_A[b] = grad_C[b] @ B[b]^T
kernel void gemm_batched_grad_a_f32(
    device const float* grad_C [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* grad_A [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z;
    uint row = gid.y;
    uint col = gid.x;

    if (batch >= batch_size || row >= params.M || col >= params.K) return;

    uint grad_c_offset = batch * params.M * params.N;
    uint b_offset = batch * params.K * params.N;
    uint grad_a_offset = batch * params.M * params.K;

    float sum = 0.0f;
    for (uint n = 0; n < params.N; n++) {
        sum += grad_C[grad_c_offset + row * params.N + n] * B[b_offset + col * params.N + n];
    }

    grad_A[grad_a_offset + row * params.K + col] = sum;
}

// Batched backward for grad_B
kernel void gemm_batched_grad_b_f32(
    device const float* A [[buffer(0)]],
    device const float* grad_C [[buffer(1)]],
    device float* grad_B [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.z;
    uint row = gid.y;
    uint col = gid.x;

    if (batch >= batch_size || row >= params.K || col >= params.N) return;

    uint a_offset = batch * params.M * params.K;
    uint grad_c_offset = batch * params.M * params.N;
    uint grad_b_offset = batch * params.K * params.N;

    float sum = 0.0f;
    for (uint m = 0; m < params.M; m++) {
        sum += A[a_offset + m * params.K + row] * grad_C[grad_c_offset + m * params.N + col];
    }

    grad_B[grad_b_offset + row * params.N + col] = sum;
}
