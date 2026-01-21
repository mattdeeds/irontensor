#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    FP32,
    FP16,
    BF16,
}

impl Precision {
    pub fn byte_size(&self) -> usize {
        match self {
            Precision::FP32 => 4,
            Precision::FP16 => 2,
            Precision::BF16 => 2,
        }
    }
}

/// BF16 (Brain Floating Point 16) utilities
///
/// BF16 has the same exponent range as FP32 (8 bits) but only 7 mantissa bits.
/// This makes conversion to/from FP32 trivial - just truncate/extend the lower 16 bits.

/// Convert f32 to bf16 (represented as u16)
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    // Round-to-nearest-even: add 0x7FFF + ((bits >> 16) & 1) before truncating
    let bits = x.to_bits();
    let round = ((bits >> 16) & 1) + 0x7FFF;
    ((bits.wrapping_add(round)) >> 16) as u16
}

/// Convert bf16 (represented as u16) to f32
#[inline]
pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

/// Convert a slice of f32 to bf16
pub fn f32_slice_to_bf16(src: &[f32]) -> Vec<u16> {
    src.iter().map(|&x| f32_to_bf16(x)).collect()
}

/// Convert a slice of bf16 to f32
pub fn bf16_slice_to_f32(src: &[u16]) -> Vec<f32> {
    src.iter().map(|&x| bf16_to_f32(x)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_conversion_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0, 1e-5, 1e5];
        for &val in &values {
            let bf16 = f32_to_bf16(val);
            let back = bf16_to_f32(bf16);
            // BF16 has ~3 decimal digits of precision
            let rel_err = if val.abs() > 1e-10 {
                ((back - val) / val).abs()
            } else {
                (back - val).abs()
            };
            assert!(rel_err < 0.01, "BF16 roundtrip error for {}: got {}, err={}", val, back, rel_err);
        }
    }

    #[test]
    fn test_bf16_special_values() {
        // Zero
        assert_eq!(bf16_to_f32(f32_to_bf16(0.0)), 0.0);

        // Infinity
        let inf_bf16 = f32_to_bf16(f32::INFINITY);
        assert!(bf16_to_f32(inf_bf16).is_infinite());

        // Negative infinity
        let neg_inf_bf16 = f32_to_bf16(f32::NEG_INFINITY);
        assert!(bf16_to_f32(neg_inf_bf16).is_infinite());
        assert!(bf16_to_f32(neg_inf_bf16) < 0.0);

        // NaN
        let nan_bf16 = f32_to_bf16(f32::NAN);
        assert!(bf16_to_f32(nan_bf16).is_nan());
    }

    #[test]
    fn test_bf16_slice_conversion() {
        let f32_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let bf16_data = f32_slice_to_bf16(&f32_data);
        let back = bf16_slice_to_f32(&bf16_data);

        for (i, (&orig, &conv)) in f32_data.iter().zip(back.iter()).enumerate() {
            assert!((orig - conv).abs() < 0.01, "Mismatch at {}: {} vs {}", i, orig, conv);
        }
    }
}
