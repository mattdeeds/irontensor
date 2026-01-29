use rand::Rng;

/// Compute softmax probabilities from logits.
///
/// Uses the numerically stable version: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exp_logits.iter().sum();
    exp_logits.iter().map(|&x| x / sum).collect()
}

/// Sample a token index from logits using softmax probabilities.
///
/// Converts logits to probabilities via softmax, then samples from the
/// resulting categorical distribution using the thread-local RNG.
pub fn sample_from_logits(logits: &[f32]) -> u32 {
    let probs = softmax(logits);

    // Sample using thread-local RNG
    let r: f32 = rand::rng().random();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u32;
        }
    }

    (probs.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_ordering() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        // Higher logit should have higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_sample_returns_valid_index() {
        let logits = vec![0.0, 1.0, 2.0, 3.0];
        for _ in 0..100 {
            let idx = sample_from_logits(&logits);
            assert!(idx < 4);
        }
    }
}
