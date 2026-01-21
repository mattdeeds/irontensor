#include <metal_stdlib>
using namespace metal;

// Backward pass for element-wise operations

// Add backward: grad_a = grad_out, grad_b = grad_out
// Since add is symmetric, this is trivial - just copy grad_out to both

// Mul backward: grad_a = grad_out * b, grad_b = grad_out * a
kernel void mul_backward_f32(
    device const float* grad_out [[buffer(0)]],
    device const float* a [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device float* grad_a [[buffer(3)]],
    device float* grad_b [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        grad_a[gid] = grad_out[gid] * b[gid];
        grad_b[gid] = grad_out[gid] * a[gid];
    }
}

// Scale backward: grad_a = grad_out * scalar
kernel void scale_backward_f32(
    device const float* grad_out [[buffer(0)]],
    device float* grad_a [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        grad_a[gid] = grad_out[gid] * scalar;
    }
}

// SiLU backward: grad_x = grad_out * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
//              = grad_out * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
kernel void silu_backward_f32(
    device const float* grad_out [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* grad_x [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        float xi = x[gid];
        float sig = 1.0f / (1.0f + exp(-xi));
        // d/dx[x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        //                      = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        float grad = sig * (1.0f + xi * (1.0f - sig));
        grad_x[gid] = grad_out[gid] * grad;
    }
}

// GELU backward (approximate)
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Let c = sqrt(2/pi), k = 0.044715
// Let u = c * (x + k * x^3)
// GELU(x) = 0.5 * x * (1 + tanh(u))
// d/dx = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * c * (1 + 3*k*x^2)
kernel void gelu_backward_f32(
    device const float* grad_out [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* grad_x [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        float xi = x[gid];
        const float c = 0.7978845608028654f;  // sqrt(2/pi)
        const float k = 0.044715f;

        float u = c * (xi + k * xi * xi * xi);
        float tanh_u = tanh(u);
        float sech2_u = 1.0f - tanh_u * tanh_u;

        // d/dx GELU = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * c * (1 + 3*k*x^2)
        float grad = 0.5f * (1.0f + tanh_u) + 0.5f * xi * sech2_u * c * (1.0f + 3.0f * k * xi * xi);
        grad_x[gid] = grad_out[gid] * grad;
    }
}

// ReLU backward: grad_x = grad_out * (x > 0 ? 1 : 0)
kernel void relu_backward_f32(
    device const float* grad_out [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* grad_x [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        grad_x[gid] = (x[gid] > 0.0f) ? grad_out[gid] : 0.0f;
    }
}

// SwiGLU backward: output = silu(gate) * up
// grad_gate = grad_out * up * d_silu(gate)
// grad_up = grad_out * silu(gate)
kernel void swiglu_backward_f32(
    device const float* grad_out [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device const float* up [[buffer(2)]],
    device float* grad_gate [[buffer(3)]],
    device float* grad_up [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        float g = gate[gid];
        float u = up[gid];
        float go = grad_out[gid];

        float sig = 1.0f / (1.0f + exp(-g));
        float silu_g = g * sig;

        // grad_up = grad_out * silu(gate)
        grad_up[gid] = go * silu_g;

        // grad_gate = grad_out * up * d_silu(gate)
        // d_silu(g) = sig * (1 + g * (1 - sig))
        float d_silu = sig * (1.0f + g * (1.0f - sig));
        grad_gate[gid] = go * u * d_silu;
    }
}
