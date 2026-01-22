#include <metal_stdlib>
using namespace metal;

// Parameters for 2D transpose
struct TransposeParams {
    uint rows;      // Input rows (output cols)
    uint cols;      // Input cols (output rows)
};

// 2D Transpose: [rows, cols] -> [cols, rows]
// Each thread handles one element
kernel void transpose_2d_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant TransposeParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= params.rows || col >= params.cols) {
        return;
    }

    // input[row, col] -> output[col, row]
    uint src_idx = row * params.cols + col;
    uint dst_idx = col * params.rows + row;

    output[dst_idx] = input[src_idx];
}

// Tiled 2D Transpose for better memory coalescing
// Uses shared memory to improve performance on larger matrices
#define TILE_SIZE 16

kernel void transpose_2d_tiled_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant TransposeParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    threadgroup float* tile [[threadgroup(0)]]
) {
    uint rows = params.rows;
    uint cols = params.cols;

    // Load tile into shared memory (coalesced read)
    uint src_row = tgid.y * TILE_SIZE + tid.y;
    uint src_col = tgid.x * TILE_SIZE + tid.x;

    float val = 0.0f;
    if (src_row < rows && src_col < cols) {
        val = input[src_row * cols + src_col];
    }

    // Store in transposed position in shared memory
    tile[tid.x * TILE_SIZE + tid.y] = val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write from shared memory (coalesced write)
    // Output position is transposed
    uint dst_row = tgid.x * TILE_SIZE + tid.y;
    uint dst_col = tgid.y * TILE_SIZE + tid.x;

    if (dst_row < cols && dst_col < rows) {
        output[dst_row * rows + dst_col] = tile[tid.y * TILE_SIZE + tid.x];
    }
}

// Parameters for batched transpose (for attention tensors)
struct BatchedTransposeParams {
    uint batch;     // Batch size
    uint dim1;      // First dimension
    uint dim2;      // Second dimension
    uint dim3;      // Third dimension (innermost)
};

// Transpose dims 1 and 2: [batch, dim1, dim2, dim3] -> [batch, dim2, dim1, dim3]
// Used for attention: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
kernel void transpose_0213_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant BatchedTransposeParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint b = gid.z;
    uint d1 = gid.y;
    uint d2_d3 = gid.x;

    uint d2 = d2_d3 / params.dim3;
    uint d3 = d2_d3 % params.dim3;

    if (b >= params.batch || d1 >= params.dim1 || d2 >= params.dim2) {
        return;
    }

    // input[b, d1, d2, d3] -> output[b, d2, d1, d3]
    uint src_idx = b * (params.dim1 * params.dim2 * params.dim3) +
                   d1 * (params.dim2 * params.dim3) +
                   d2 * params.dim3 +
                   d3;

    uint dst_idx = b * (params.dim2 * params.dim1 * params.dim3) +
                   d2 * (params.dim1 * params.dim3) +
                   d1 * params.dim3 +
                   d3;

    output[dst_idx] = input[src_idx];
}

// Transpose back: [batch, dim2, dim1, dim3] -> [batch, dim1, dim2, dim3]
// This is the inverse of transpose_0213
kernel void transpose_0213_inverse_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant BatchedTransposeParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint b = gid.z;
    uint d2 = gid.y;  // Note: input has dim2 first
    uint d1_d3 = gid.x;

    uint d1 = d1_d3 / params.dim3;
    uint d3 = d1_d3 % params.dim3;

    if (b >= params.batch || d1 >= params.dim1 || d2 >= params.dim2) {
        return;
    }

    // input[b, d2, d1, d3] -> output[b, d1, d2, d3]
    uint src_idx = b * (params.dim2 * params.dim1 * params.dim3) +
                   d2 * (params.dim1 * params.dim3) +
                   d1 * params.dim3 +
                   d3;

    uint dst_idx = b * (params.dim1 * params.dim2 * params.dim3) +
                   d1 * (params.dim2 * params.dim3) +
                   d2 * params.dim3 +
                   d3;

    output[dst_idx] = input[src_idx];
}
