#include <metal_stdlib>
using namespace metal;

// Parameters for repeat_kv operations
struct RepeatKVParams {
    uint batch;         // Batch size
    uint seq_len;       // Sequence length
    uint num_heads;     // Number of query heads
    uint num_kv_heads;  // Number of KV heads
    uint head_dim;      // Head dimension
    uint repeats;       // num_heads / num_kv_heads
};

// Forward: Repeat KV heads for GQA (Grouped Query Attention)
// Input:  [batch, seq_len, num_kv_heads, head_dim]
// Output: [batch, seq_len, num_heads, head_dim]
//
// Each thread handles one output element
kernel void repeat_kv_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant RepeatKVParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint b_s = gid.z;           // batch * seq_len combined
    uint h = gid.y;             // output head index
    uint d = gid.x;             // head_dim index

    uint batch_seq = params.batch * params.seq_len;

    if (b_s >= batch_seq || h >= params.num_heads || d >= params.head_dim) {
        return;
    }

    // Map output head index to KV head index
    uint kv_h = h / params.repeats;

    // Source index: [b, s, kv_h, d]
    uint src_idx = b_s * params.num_kv_heads * params.head_dim +
                   kv_h * params.head_dim +
                   d;

    // Destination index: [b, s, h, d]
    uint dst_idx = b_s * params.num_heads * params.head_dim +
                   h * params.head_dim +
                   d;

    output[dst_idx] = input[src_idx];
}

// Backward: Sum gradients from expanded heads back to KV heads
// Input (grad_expanded):  [batch, seq_len, num_heads, head_dim]
// Output (grad_kv):       [batch, seq_len, num_kv_heads, head_dim]
//
// Each thread handles one output element (sums over repeated heads)
kernel void repeat_kv_backward_f32(
    device const float* grad_expanded [[buffer(0)]],
    device float* grad_kv [[buffer(1)]],
    constant RepeatKVParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint b_s = gid.z;           // batch * seq_len combined
    uint kv_h = gid.y;          // kv head index
    uint d = gid.x;             // head_dim index

    uint batch_seq = params.batch * params.seq_len;

    if (b_s >= batch_seq || kv_h >= params.num_kv_heads || d >= params.head_dim) {
        return;
    }

    // Sum gradients from all repeated heads
    float sum = 0.0f;
    for (uint r = 0; r < params.repeats; r++) {
        uint h = kv_h * params.repeats + r;
        uint src_idx = b_s * params.num_heads * params.head_dim +
                       h * params.head_dim +
                       d;
        sum += grad_expanded[src_idx];
    }

    // Destination index: [b, s, kv_h, d]
    uint dst_idx = b_s * params.num_kv_heads * params.head_dim +
                   kv_h * params.head_dim +
                   d;

    grad_kv[dst_idx] = sum;
}
