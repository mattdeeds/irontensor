#include <metal_stdlib>
using namespace metal;

// GEMM: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
// Row-major layout

struct GemmParams {
    uint M;
    uint N;
    uint K;
};

// Simple GEMM kernel - each thread computes one element of C
kernel void gemm_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;

    if (row >= params.M || col >= params.N) return;

    float sum = 0.0f;
    for (uint k = 0; k < params.K; k++) {
        sum += A[row * params.K + k] * B[k * params.N + col];
    }

    C[row * params.N + col] = sum;
}

// Tiled GEMM for better memory access patterns
// TILE_SIZE should match threadgroup size
#define TILE_SIZE 16

kernel void gemm_f32_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant GemmParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    // Shared memory tiles
    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;

    float sum = 0.0f;

    // Number of tiles needed to cover K dimension
    uint numTiles = (params.K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        // Load tile of A into shared memory
        uint aCol = t * TILE_SIZE + tid.x;
        if (row < params.M && aCol < params.K) {
            As[tid.y][tid.x] = A[row * params.K + aCol];
        } else {
            As[tid.y][tid.x] = 0.0f;
        }

        // Load tile of B into shared memory
        uint bRow = t * TILE_SIZE + tid.y;
        if (bRow < params.K && col < params.N) {
            Bs[tid.y][tid.x] = B[bRow * params.N + col];
        } else {
            Bs[tid.y][tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product for this tile
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[tid.y][k] * Bs[k][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (row < params.M && col < params.N) {
        C[row * params.N + col] = sum;
    }
}

// Batched GEMM: C[b] = A[b] @ B[b]
// A: [batch, M, K], B: [batch, K, N], C: [batch, M, N]
kernel void gemm_f32_batched(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
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
        sum += A[a_offset + row * params.K + k] * B[b_offset + k * params.N + col];
    }

    C[c_offset + row * params.N + col] = sum;
}
