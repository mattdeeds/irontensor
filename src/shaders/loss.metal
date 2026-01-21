#include <metal_stdlib>
using namespace metal;

// Cross-Entropy Loss
// Loss = -sum(target * log(softmax(logits)))
// For classification with hard labels: Loss = -log(softmax(logits)[target])
//
// Combined softmax + cross-entropy is more numerically stable:
// Loss = -logits[target] + log(sum(exp(logits)))
//      = -logits[target] + max + log(sum(exp(logits - max)))

struct CrossEntropyParams {
    uint batch_size;
    uint vocab_size;
};

// Cross-entropy loss forward
// logits: [batch, vocab_size]
// targets: [batch] - indices of correct class
// loss: [batch] - per-sample loss
kernel void cross_entropy_forward_f32(
    device const float* logits [[buffer(0)]],
    device const uint* targets [[buffer(1)]],
    device float* loss [[buffer(2)]],
    constant CrossEntropyParams& params [[buffer(3)]],
    uint batch [[thread_position_in_grid]])
{
    if (batch >= params.batch_size) return;

    uint offset = batch * params.vocab_size;
    uint target = targets[batch];

    // Find max for numerical stability
    float max_val = logits[offset];
    for (uint i = 1; i < params.vocab_size; i++) {
        max_val = max(max_val, logits[offset + i]);
    }

    // Compute log(sum(exp(logits - max)))
    float sum_exp = 0.0f;
    for (uint i = 0; i < params.vocab_size; i++) {
        sum_exp += exp(logits[offset + i] - max_val);
    }
    float log_sum_exp = max_val + log(sum_exp);

    // Loss = -logits[target] + log_sum_exp
    loss[batch] = -logits[offset + target] + log_sum_exp;
}

// Cross-entropy loss backward (combined with softmax)
// grad_logits = softmax(logits) - one_hot(target)
// This is the gradient of cross_entropy(softmax(logits), target) w.r.t. logits
kernel void cross_entropy_backward_f32(
    device const float* logits [[buffer(0)]],
    device const uint* targets [[buffer(1)]],
    device float* grad_logits [[buffer(2)]],
    constant CrossEntropyParams& params [[buffer(3)]],
    constant float& grad_scale [[buffer(4)]],  // Usually 1/batch_size for mean loss
    uint batch [[thread_position_in_grid]])
{
    if (batch >= params.batch_size) return;

    uint offset = batch * params.vocab_size;
    uint target = targets[batch];

    // Find max for numerical stability
    float max_val = logits[offset];
    for (uint i = 1; i < params.vocab_size; i++) {
        max_val = max(max_val, logits[offset + i]);
    }

    // Compute softmax
    float sum_exp = 0.0f;
    for (uint i = 0; i < params.vocab_size; i++) {
        float exp_val = exp(logits[offset + i] - max_val);
        grad_logits[offset + i] = exp_val;  // Store exp values temporarily
        sum_exp += exp_val;
    }

    // Normalize to get softmax, then subtract one-hot
    float inv_sum = 1.0f / sum_exp;
    for (uint i = 0; i < params.vocab_size; i++) {
        float softmax_val = grad_logits[offset + i] * inv_sum;
        // grad = softmax - one_hot(target)
        float target_val = (i == target) ? 1.0f : 0.0f;
        grad_logits[offset + i] = (softmax_val - target_val) * grad_scale;
    }
}

// Fused cross-entropy: computes both loss and gradient in one pass
kernel void cross_entropy_fused_f32(
    device const float* logits [[buffer(0)]],
    device const uint* targets [[buffer(1)]],
    device float* loss [[buffer(2)]],
    device float* grad_logits [[buffer(3)]],
    constant CrossEntropyParams& params [[buffer(4)]],
    constant float& grad_scale [[buffer(5)]],
    uint batch [[thread_position_in_grid]])
{
    if (batch >= params.batch_size) return;

    uint offset = batch * params.vocab_size;
    uint target = targets[batch];

    // Find max for numerical stability
    float max_val = logits[offset];
    for (uint i = 1; i < params.vocab_size; i++) {
        max_val = max(max_val, logits[offset + i]);
    }

    // Compute exp values and sum
    float sum_exp = 0.0f;
    for (uint i = 0; i < params.vocab_size; i++) {
        float exp_val = exp(logits[offset + i] - max_val);
        grad_logits[offset + i] = exp_val;
        sum_exp += exp_val;
    }

    // Compute loss
    float log_sum_exp = max_val + log(sum_exp);
    loss[batch] = -logits[offset + target] + log_sum_exp;

    // Compute gradients: softmax - one_hot
    float inv_sum = 1.0f / sum_exp;
    for (uint i = 0; i < params.vocab_size; i++) {
        float softmax_val = grad_logits[offset + i] * inv_sum;
        float target_val = (i == target) ? 1.0f : 0.0f;
        grad_logits[offset + i] = (softmax_val - target_val) * grad_scale;
    }
}

// Reduce losses to scalar (mean)
kernel void reduce_mean_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    // Simple serial reduction (can be parallelized if needed)
    if (gid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < count; i++) {
            sum += input[i];
        }
        output[0] = sum / float(count);
    }
}
