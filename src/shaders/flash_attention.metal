#include <metal_stdlib>
using namespace metal;

// FlashAttention - Memory-efficient attention using tiled computation
// Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
//
// Key optimizations:
// 1. Compute attention in tiles to fit in fast SRAM (threadgroup memory)
// 2. Use online softmax to avoid materializing the full N×N attention matrix
// 3. Fuse the scaling, masking, softmax, and value multiplication

struct FlashAttentionParams {
    uint batch_size;
    uint num_heads;
    uint seq_len;
    uint head_dim;
    float scale;        // 1/sqrt(head_dim)
    uint causal;        // 1 for causal masking, 0 otherwise
};

// Block sizes for tiling - tuned for Apple Silicon
constant uint BLOCK_M = 32;  // Query block size
constant uint BLOCK_N = 32;  // Key/Value block size

// Online softmax helper: given old (max, sum) and new values, compute updated (max, sum)
// and the correction factor for previously accumulated values
inline void online_softmax_update(
    float old_max,
    float old_sum,
    float new_max,
    float new_sum,
    thread float& out_max,
    thread float& out_sum,
    thread float& correction)
{
    out_max = max(old_max, new_max);
    float old_correction = exp(old_max - out_max);
    float new_correction = exp(new_max - out_max);
    out_sum = old_sum * old_correction + new_sum * new_correction;
    correction = old_correction;
}

// FlashAttention forward pass kernel
// Processes one query block per threadgroup
kernel void flash_attention_forward_f32(
    device const float* Q [[buffer(0)]],           // [batch, heads, seq, head_dim]
    device const float* K [[buffer(1)]],           // [batch, heads, seq, head_dim]
    device const float* V [[buffer(2)]],           // [batch, heads, seq, head_dim]
    device float* O [[buffer(3)]],                 // [batch, heads, seq, head_dim]
    constant FlashAttentionParams& params [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tg_size [[threads_per_threadgroup]])
{
    // Determine which batch, head, and query block we're processing
    uint batch_idx = tgid.z / params.num_heads;
    uint head_idx = tgid.z % params.num_heads;
    uint q_block_idx = tgid.x;

    uint q_start = q_block_idx * BLOCK_M;
    if (q_start >= params.seq_len) return;

    uint q_end = min(q_start + BLOCK_M, params.seq_len);
    uint q_len = q_end - q_start;

    // Thread index within the block
    uint thread_idx = tid.x;

    // Base offset for this batch and head
    uint base_offset = (batch_idx * params.num_heads + head_idx) * params.seq_len * params.head_dim;

    // Each thread handles one query position within the block
    if (thread_idx >= q_len) return;

    uint q_pos = q_start + thread_idx;

    // Load query vector for this position
    float q_vec[64];  // Assume head_dim <= 64
    for (uint d = 0; d < params.head_dim; d++) {
        q_vec[d] = Q[base_offset + q_pos * params.head_dim + d];
    }

    // Initialize online softmax state
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    // Initialize output accumulator
    float o_vec[64];
    for (uint d = 0; d < params.head_dim; d++) {
        o_vec[d] = 0.0f;
    }

    // Determine the range of keys to attend to
    uint k_end = params.causal ? (q_pos + 1) : params.seq_len;

    // Process key/value blocks
    for (uint k_start = 0; k_start < k_end; k_start += BLOCK_N) {
        uint k_block_end = min(k_start + BLOCK_N, k_end);

        // Compute attention scores for this block
        float block_max = -INFINITY;
        float scores[32];  // Assume BLOCK_N <= 32
        uint num_keys = k_block_end - k_start;

        for (uint k_idx = 0; k_idx < num_keys; k_idx++) {
            uint k_pos = k_start + k_idx;

            // Compute dot product Q[q_pos] · K[k_pos]
            float score = 0.0f;
            for (uint d = 0; d < params.head_dim; d++) {
                float k_val = K[base_offset + k_pos * params.head_dim + d];
                score += q_vec[d] * k_val;
            }
            score *= params.scale;

            // Apply causal mask
            if (params.causal && k_pos > q_pos) {
                score = -INFINITY;
            }

            scores[k_idx] = score;
            block_max = max(block_max, score);
        }

        // Compute softmax for this block and update running statistics
        float block_sum = 0.0f;
        for (uint k_idx = 0; k_idx < num_keys; k_idx++) {
            float exp_score = exp(scores[k_idx] - block_max);
            scores[k_idx] = exp_score;
            block_sum += exp_score;
        }

        // Update online softmax state
        float new_max, new_sum, correction;
        online_softmax_update(row_max, row_sum, block_max, block_sum, new_max, new_sum, correction);

        // Correct previous output accumulator
        for (uint d = 0; d < params.head_dim; d++) {
            o_vec[d] *= correction;
        }

        // Add contribution from this block
        float block_correction = exp(block_max - new_max);
        for (uint k_idx = 0; k_idx < num_keys; k_idx++) {
            uint k_pos = k_start + k_idx;
            float attn_weight = scores[k_idx] * block_correction;

            for (uint d = 0; d < params.head_dim; d++) {
                float v_val = V[base_offset + k_pos * params.head_dim + d];
                o_vec[d] += attn_weight * v_val;
            }
        }

        row_max = new_max;
        row_sum = new_sum;
    }

    // Normalize output by sum and write to global memory
    float inv_sum = 1.0f / row_sum;
    for (uint d = 0; d < params.head_dim; d++) {
        O[base_offset + q_pos * params.head_dim + d] = o_vec[d] * inv_sum;
    }
}

// Simpler version for small sequences that fits entirely in registers
// More efficient for seq_len <= 256
kernel void flash_attention_small_f32(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant FlashAttentionParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch_idx = gid.z / params.num_heads;
    uint head_idx = gid.z % params.num_heads;
    uint q_pos = gid.x;

    if (q_pos >= params.seq_len) return;

    uint base_offset = (batch_idx * params.num_heads + head_idx) * params.seq_len * params.head_dim;

    // Load query
    float q_vec[128];  // Support up to 128 head_dim
    for (uint d = 0; d < params.head_dim; d++) {
        q_vec[d] = Q[base_offset + q_pos * params.head_dim + d];
    }

    // Compute attention scores
    uint k_end = params.causal ? (q_pos + 1) : params.seq_len;

    float max_score = -INFINITY;
    float scores[512];  // Support up to 512 seq_len

    for (uint k_pos = 0; k_pos < k_end; k_pos++) {
        float score = 0.0f;
        for (uint d = 0; d < params.head_dim; d++) {
            score += q_vec[d] * K[base_offset + k_pos * params.head_dim + d];
        }
        score *= params.scale;
        scores[k_pos] = score;
        max_score = max(max_score, score);
    }

    // Softmax
    float sum_exp = 0.0f;
    for (uint k_pos = 0; k_pos < k_end; k_pos++) {
        float exp_score = exp(scores[k_pos] - max_score);
        scores[k_pos] = exp_score;
        sum_exp += exp_score;
    }

    float inv_sum = 1.0f / sum_exp;

    // Compute output
    for (uint d = 0; d < params.head_dim; d++) {
        float o_val = 0.0f;
        for (uint k_pos = 0; k_pos < k_end; k_pos++) {
            o_val += scores[k_pos] * V[base_offset + k_pos * params.head_dim + d];
        }
        O[base_offset + q_pos * params.head_dim + d] = o_val * inv_sum;
    }
}
