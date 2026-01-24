#include <metal_stdlib>
using namespace metal;

// Philox 4x32-10 RNG - must match forward shader exactly

// Philox constants
constant uint PHILOX_M0 = 0xD2511F53u;
constant uint PHILOX_M1 = 0xCD9E8D57u;
constant uint PHILOX_W0 = 0x9E3779B9u;
constant uint PHILOX_W1 = 0xBB67AE85u;

// Single Philox round
inline void philox_round(thread uint4& ctr, uint2 key) {
    uint hi0 = mulhi(PHILOX_M0, ctr.x);
    uint lo0 = PHILOX_M0 * ctr.x;
    uint hi1 = mulhi(PHILOX_M1, ctr.z);
    uint lo1 = PHILOX_M1 * ctr.z;

    ctr = uint4(
        hi1 ^ ctr.y ^ key.x,
        lo1,
        hi0 ^ ctr.w ^ key.y,
        lo0
    );
}

// Bump the key between rounds
inline uint2 philox_bumpkey(uint2 key) {
    return uint2(key.x + PHILOX_W0, key.y + PHILOX_W1);
}

// Philox 4x32-10: 10 rounds for high-quality randomness
inline uint4 philox4x32_10(uint4 ctr, uint2 key) {
    philox_round(ctr, key); key = philox_bumpkey(key);
    philox_round(ctr, key); key = philox_bumpkey(key);
    philox_round(ctr, key); key = philox_bumpkey(key);
    philox_round(ctr, key); key = philox_bumpkey(key);
    philox_round(ctr, key); key = philox_bumpkey(key);
    philox_round(ctr, key); key = philox_bumpkey(key);
    philox_round(ctr, key); key = philox_bumpkey(key);
    philox_round(ctr, key); key = philox_bumpkey(key);
    philox_round(ctr, key); key = philox_bumpkey(key);
    philox_round(ctr, key);
    return ctr;
}

// Convert uint to uniform float in [0, 1)
inline float uint_to_float(uint x) {
    return float(x) * (1.0f / 4294967296.0f);
}

// Dropout backward kernel
// Regenerates the same mask using the same seed
// grad_input = grad_output * mask * scale (where mask is 1 if kept, 0 if dropped)
kernel void dropout_backward_f32(
    device const float* grad_output [[buffer(0)]],
    device float* grad_input [[buffer(1)]],
    constant float& dropout_prob [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    constant uint& seed_lo [[buffer(4)]],
    constant uint& seed_hi [[buffer(5)]],
    constant uint& count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    // Regenerate the same random number as forward pass
    uint4 ctr = uint4(gid, 0, 0, 0);
    uint2 key = uint2(seed_lo, seed_hi);
    uint4 rand = philox4x32_10(ctr, key);

    float u = uint_to_float(rand.x);

    // Apply same mask: if element was kept in forward, propagate gradient
    if (u >= dropout_prob) {
        grad_input[gid] = grad_output[gid] * scale;
    } else {
        grad_input[gid] = 0.0f;
    }
}
