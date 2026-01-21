#include <metal_stdlib>
using namespace metal;

// Element-wise addition: C = A + B
kernel void add_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        C[gid] = A[gid] + B[gid];
    }
}

// Element-wise multiplication: C = A * B
kernel void mul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        C[gid] = A[gid] * B[gid];
    }
}

// Scale: B = A * scalar
kernel void scale_f32(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        B[gid] = A[gid] * scalar;
    }
}

// Add scalar: B = A + scalar
kernel void add_scalar_f32(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        B[gid] = A[gid] + scalar;
    }
}

// SiLU (Swish) activation: y = x * sigmoid(x)
kernel void silu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        float x = input[gid];
        float sigmoid_x = 1.0f / (1.0f + exp(-x));
        output[gid] = x * sigmoid_x;
    }
}

// GELU activation (approximate): y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
kernel void gelu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        float x = input[gid];
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coeff = 0.044715f;
        float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        output[gid] = 0.5f * x * (1.0f + tanh(inner));
    }
}

// ReLU activation: y = max(0, x)
kernel void relu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        output[gid] = max(0.0f, input[gid]);
    }
}

// SwiGLU: output = silu(gate) * up
// This is used in Llama-style FFN: output = silu(W_gate @ x) * (W_up @ x)
kernel void swiglu_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        float g = gate[gid];
        float sigmoid_g = 1.0f / (1.0f + exp(-g));
        float silu_g = g * sigmoid_g;
        output[gid] = silu_g * up[gid];
    }
}
