#include <metal_stdlib>
using namespace metal;

// Lion optimizer parameters
struct LionParams {
    uint count;          // Number of parameters
    float lr;            // Learning rate
    float beta1;         // Momentum decay for update (typically 0.9)
    float beta2;         // Momentum decay for momentum (typically 0.99)
    float weight_decay;  // Weight decay coefficient
};

// Lion optimizer update kernel
// Update rule:
//   update = sign(beta2 * m + (1 - beta2) * g)
//   m_new = beta1 * m + (1 - beta1) * g
//   w_new = w - lr * update - lr * weight_decay * w
//
// This is more memory efficient than Adam (only 1 state tensor vs 2)
// and often converges faster for language models.
kernel void lion_step_f32(
    device float* weights [[buffer(0)]],       // Parameters (in-place update)
    device float* momentum [[buffer(1)]],      // Momentum state (in-place update)
    device const float* gradients [[buffer(2)]], // Gradients
    constant LionParams& params [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= params.count) return;

    float w = weights[idx];
    float m = momentum[idx];
    float g = gradients[idx];

    // Compute update direction using interpolated momentum and gradient
    float update_input = params.beta2 * m + (1.0f - params.beta2) * g;
    float update = sign(update_input);

    // Update momentum for next iteration
    float m_new = params.beta1 * m + (1.0f - params.beta1) * g;
    momentum[idx] = m_new;

    // Update weights: gradient step + weight decay
    float w_new = w - params.lr * update - params.lr * params.weight_decay * w;
    weights[idx] = w_new;
}

// Fused Lion step for multiple parameter groups with different learning rates
// Useful when you want different LR for embeddings vs other layers
kernel void lion_step_scaled_f32(
    device float* weights [[buffer(0)]],
    device float* momentum [[buffer(1)]],
    device const float* gradients [[buffer(2)]],
    device const float* lr_scale [[buffer(3)]],  // Per-parameter LR multiplier
    constant LionParams& params [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= params.count) return;

    float w = weights[idx];
    float m = momentum[idx];
    float g = gradients[idx];
    float scale = lr_scale[idx];
    float lr = params.lr * scale;

    float update_input = params.beta2 * m + (1.0f - params.beta2) * g;
    float update = sign(update_input);

    float m_new = params.beta1 * m + (1.0f - params.beta1) * g;
    momentum[idx] = m_new;

    float w_new = w - lr * update - lr * params.weight_decay * w;
    weights[idx] = w_new;
}

// Zero gradients kernel (useful for gradient accumulation)
kernel void zero_gradients_f32(
    device float* gradients [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= count) return;
    gradients[idx] = 0.0f;
}

// Gradient clipping by global norm
// First pass: compute sum of squared gradients
kernel void grad_norm_squared_f32(
    device const float* gradients [[buffer(0)]],
    device atomic_float* sum_sq [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= count) return;
    float g = gradients[idx];
    atomic_fetch_add_explicit(sum_sq, g * g, memory_order_relaxed);
}

// Second pass: clip gradients if norm exceeds threshold
kernel void grad_clip_f32(
    device float* gradients [[buffer(0)]],
    constant float& clip_scale [[buffer(1)]],  // min(1.0, max_norm / actual_norm)
    constant uint& count [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= count) return;
    gradients[idx] *= clip_scale;
}
