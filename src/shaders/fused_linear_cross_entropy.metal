#include <metal_stdlib>
using namespace metal;

// FusedLinearCrossEntropy - Memory-efficient output layer + loss computation
//
// Key insight: We don't need to materialize the full [batch*seq, vocab_size] logits tensor.
// Instead, we compute logits on-the-fly and only keep what's needed:
// 1. The logit for the target token
// 2. The log-sum-exp for normalization
//
// This reduces memory from O(batch * seq * vocab) to O(batch * seq)

struct FusedLinearCEParams {
    uint batch_seq;       // batch_size * seq_len (total tokens)
    uint hidden_dim;      // Hidden dimension
    uint vocab_size;      // Vocabulary size
    float ignore_index;   // Target value to ignore (typically -100)
    float grad_scale;     // Gradient scaling factor (typically 1/batch_seq for mean reduction)
};

// Kernel 1: Compute loss and gradient for hidden states
// For each token position, compute:
// 1. logits = hidden @ weight.T (streaming, never materialize)
// 2. loss = -log(softmax(logits)[target])
// 3. grad_hidden = (softmax(logits) - one_hot(target)) @ weight
//
// This kernel processes one token position per thread
kernel void fused_linear_cross_entropy_forward_f32(
    device const float* hidden [[buffer(0)]],      // [batch*seq, hidden_dim]
    device const float* weight [[buffer(1)]],      // [vocab_size, hidden_dim]
    device const int* targets [[buffer(2)]],       // [batch*seq]
    device float* loss [[buffer(3)]],              // [1] total loss
    device float* grad_hidden [[buffer(4)]],       // [batch*seq, hidden_dim]
    device atomic_float* loss_accumulator [[buffer(5)]], // For atomic addition
    constant FusedLinearCEParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.batch_seq) return;

    int target = targets[gid];

    // Skip if target is ignore_index
    if (float(target) == params.ignore_index || target < 0 || uint(target) >= params.vocab_size) {
        // Zero gradient for this position
        for (uint d = 0; d < params.hidden_dim; d++) {
            grad_hidden[gid * params.hidden_dim + d] = 0.0f;
        }
        return;
    }

    // Get hidden state for this position
    device const float* h = hidden + gid * params.hidden_dim;

    // First pass: compute max logit for numerical stability
    float max_logit = -INFINITY;
    for (uint v = 0; v < params.vocab_size; v++) {
        device const float* w = weight + v * params.hidden_dim;
        float logit = 0.0f;
        for (uint d = 0; d < params.hidden_dim; d++) {
            logit += h[d] * w[d];
        }
        max_logit = max(max_logit, logit);
    }

    // Second pass: compute sum of exp(logit - max) and target logit
    float sum_exp = 0.0f;
    float target_logit = 0.0f;

    for (uint v = 0; v < params.vocab_size; v++) {
        device const float* w = weight + v * params.hidden_dim;
        float logit = 0.0f;
        for (uint d = 0; d < params.hidden_dim; d++) {
            logit += h[d] * w[d];
        }
        sum_exp += exp(logit - max_logit);
        if (v == uint(target)) {
            target_logit = logit;
        }
    }

    // Compute loss: -log(softmax[target]) = -(target_logit - max - log(sum_exp))
    float log_sum_exp = max_logit + log(sum_exp);
    float token_loss = log_sum_exp - target_logit;

    // Atomically add to total loss
    atomic_fetch_add_explicit(loss_accumulator, token_loss, memory_order_relaxed);

    // Compute gradient for hidden state
    // grad_h = sum_v(softmax[v] * weight[v]) - weight[target]
    // = sum_v(exp(logit[v] - log_sum_exp) * weight[v]) - weight[target]

    device float* grad_h = grad_hidden + gid * params.hidden_dim;
    device const float* w_target = weight + uint(target) * params.hidden_dim;

    // Initialize gradient to -weight[target]
    for (uint d = 0; d < params.hidden_dim; d++) {
        grad_h[d] = -w_target[d];
    }

    // Add softmax[v] * weight[v] for all v
    for (uint v = 0; v < params.vocab_size; v++) {
        device const float* w = weight + v * params.hidden_dim;
        float logit = 0.0f;
        for (uint d = 0; d < params.hidden_dim; d++) {
            logit += h[d] * w[d];
        }
        float prob = exp(logit - log_sum_exp);
        for (uint d = 0; d < params.hidden_dim; d++) {
            grad_h[d] += prob * w[d];
        }
    }
}

// Kernel 2: Compute gradient for weight matrix
// grad_weight[v] = sum over tokens where target=v of (grad_logit * hidden)
// where grad_logit = softmax - one_hot(target)
//
// This kernel is called after the forward pass
kernel void fused_linear_cross_entropy_weight_grad_f32(
    device const float* hidden [[buffer(0)]],      // [batch*seq, hidden_dim]
    device const float* weight [[buffer(1)]],      // [vocab_size, hidden_dim]
    device const int* targets [[buffer(2)]],       // [batch*seq]
    device float* grad_weight [[buffer(3)]],       // [vocab_size, hidden_dim]
    constant FusedLinearCEParams& params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])         // (vocab_idx, hidden_idx)
{
    uint v = gid.x;
    uint d = gid.y;

    if (v >= params.vocab_size || d >= params.hidden_dim) return;

    float grad_sum = 0.0f;

    for (uint t = 0; t < params.batch_seq; t++) {
        int target = targets[t];

        // Skip ignored tokens
        if (float(target) == params.ignore_index || target < 0 || uint(target) >= params.vocab_size) {
            continue;
        }

        device const float* h = hidden + t * params.hidden_dim;

        // Compute logits for this token (need to recompute for log_sum_exp)
        float max_logit = -INFINITY;
        for (uint vv = 0; vv < params.vocab_size; vv++) {
            device const float* w = weight + vv * params.hidden_dim;
            float logit = 0.0f;
            for (uint dd = 0; dd < params.hidden_dim; dd++) {
                logit += h[dd] * w[dd];
            }
            max_logit = max(max_logit, logit);
        }

        float sum_exp = 0.0f;
        float this_logit = 0.0f;
        for (uint vv = 0; vv < params.vocab_size; vv++) {
            device const float* w = weight + vv * params.hidden_dim;
            float logit = 0.0f;
            for (uint dd = 0; dd < params.hidden_dim; dd++) {
                logit += h[dd] * w[dd];
            }
            sum_exp += exp(logit - max_logit);
            if (vv == v) {
                this_logit = logit;
            }
        }

        float log_sum_exp = max_logit + log(sum_exp);
        float prob = exp(this_logit - log_sum_exp);

        // grad_logit = prob - (1 if v == target else 0)
        float grad_logit = prob;
        if (uint(target) == v) {
            grad_logit -= 1.0f;
        }

        // grad_weight[v][d] += grad_logit * h[d]
        grad_sum += grad_logit * h[d];
    }

    grad_weight[v * params.hidden_dim + d] = grad_sum;
}

// Optimized version: Chunked computation for large vocabularies
// Process vocabulary in chunks to reuse hidden state loads
constant uint VOCAB_CHUNK_SIZE = 256;

kernel void fused_linear_cross_entropy_chunked_f32(
    device const float* hidden [[buffer(0)]],      // [batch*seq, hidden_dim]
    device const float* weight [[buffer(1)]],      // [vocab_size, hidden_dim]
    device const int* targets [[buffer(2)]],       // [batch*seq]
    device float* losses [[buffer(3)]],            // [batch*seq] per-token loss
    device float* grad_hidden [[buffer(4)]],       // [batch*seq, hidden_dim]
    constant FusedLinearCEParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    if (gid >= params.batch_seq) return;

    int target = targets[gid];

    // Skip ignored tokens
    if (float(target) == params.ignore_index || target < 0 || uint(target) >= params.vocab_size) {
        losses[gid] = 0.0f;
        for (uint d = 0; d < params.hidden_dim; d++) {
            grad_hidden[gid * params.hidden_dim + d] = 0.0f;
        }
        return;
    }

    device const float* h = hidden + gid * params.hidden_dim;

    // Load hidden state into registers
    float h_local[256];  // Assume hidden_dim <= 256
    for (uint d = 0; d < params.hidden_dim; d++) {
        h_local[d] = h[d];
    }

    // Process vocabulary in chunks for better memory access
    float max_logit = -INFINITY;

    // First pass: find max logit
    for (uint chunk_start = 0; chunk_start < params.vocab_size; chunk_start += VOCAB_CHUNK_SIZE) {
        uint chunk_end = min(chunk_start + VOCAB_CHUNK_SIZE, params.vocab_size);

        for (uint v = chunk_start; v < chunk_end; v++) {
            device const float* w = weight + v * params.hidden_dim;
            float logit = 0.0f;
            for (uint d = 0; d < params.hidden_dim; d++) {
                logit += h_local[d] * w[d];
            }
            max_logit = max(max_logit, logit);
        }
    }

    // Second pass: compute sum_exp and target_logit
    float sum_exp = 0.0f;
    float target_logit = 0.0f;

    for (uint v = 0; v < params.vocab_size; v++) {
        device const float* w = weight + v * params.hidden_dim;
        float logit = 0.0f;
        for (uint d = 0; d < params.hidden_dim; d++) {
            logit += h_local[d] * w[d];
        }
        sum_exp += exp(logit - max_logit);
        if (v == uint(target)) {
            target_logit = logit;
        }
    }

    float log_sum_exp = max_logit + log(sum_exp);
    losses[gid] = log_sum_exp - target_logit;

    // Compute gradient
    device float* grad_h = grad_hidden + gid * params.hidden_dim;
    device const float* w_target = weight + uint(target) * params.hidden_dim;

    // Initialize gradient to -weight[target]
    for (uint d = 0; d < params.hidden_dim; d++) {
        grad_h[d] = -w_target[d];
    }

    // Add softmax[v] * weight[v] for all v
    for (uint v = 0; v < params.vocab_size; v++) {
        device const float* w = weight + v * params.hidden_dim;
        float logit = 0.0f;
        for (uint d = 0; d < params.hidden_dim; d++) {
            logit += h_local[d] * w[d];
        }
        float prob = exp(logit - log_sum_exp);
        for (uint d = 0; d < params.hidden_dim; d++) {
            grad_h[d] += prob * w[d];
        }
    }

    // Apply gradient scaling (1/batch_size for mean reduction)
    for (uint d = 0; d < params.hidden_dim; d++) {
        grad_h[d] *= params.grad_scale;
    }
}

// Simple kernel to copy atomic accumulator to output
kernel void copy_loss_f32(
    device atomic_float* accumulator [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        output[0] = atomic_load_explicit(accumulator, memory_order_relaxed) / float(count);
    }
}
