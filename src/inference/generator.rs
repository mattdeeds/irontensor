use crate::logging::InferenceTimer;
use crate::nn::GPTModel;
use tokenizers::Tokenizer;

use super::sampling::sample_from_logits;

/// Configuration for text generation.
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Temperature for sampling. 0.0 = greedy (argmax), >0 = softmax sampling.
    pub temperature: f32,
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Token ID to stop generation at. Default is Some(0) for end-of-text.
    pub stop_token: Option<u32>,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            max_tokens: 100,
            stop_token: Some(0),
        }
    }
}

/// Text generator for autoregressive language models.
///
/// Handles the generation loop including:
/// - Context window truncation
/// - Temperature scaling
/// - Softmax sampling or greedy decoding
/// - Performance logging via InferenceTimer
pub struct TextGenerator {
    config: GeneratorConfig,
}

impl TextGenerator {
    /// Create a new text generator with the given configuration.
    pub fn new(config: GeneratorConfig) -> Self {
        Self { config }
    }

    /// Generate text from a prompt.
    ///
    /// The model is temporarily set to eval mode during generation (disables dropout),
    /// then restored to its previous training state.
    ///
    /// # Arguments
    /// * `model` - The GPT model to use for generation
    /// * `tokenizer` - Tokenizer for encoding/decoding text
    /// * `prompt` - The input prompt to continue from
    ///
    /// # Returns
    /// The generated text including the original prompt.
    pub fn generate(
        &self,
        model: &mut GPTModel,
        tokenizer: &Tokenizer,
        prompt: &str,
    ) -> String {
        // Save training state and set to eval mode
        let was_training = model.is_training();
        model.set_training(false);

        let result = self.generate_internal(model, tokenizer, prompt);

        // Restore training state
        model.set_training(was_training);

        result
    }

    fn generate_internal(
        &self,
        model: &GPTModel,
        tokenizer: &Tokenizer,
        prompt: &str,
    ) -> String {
        // Encode prompt
        let encoding = tokenizer
            .encode(prompt, false)
            .expect("Failed to encode prompt");
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_len = tokens.len();

        let max_len = model.config.max_seq_len;

        // Start inference timing
        let mut timer = InferenceTimer::new(prompt_len, self.config.temperature);

        // Generate tokens one at a time
        for i in 0..self.config.max_tokens {
            // Truncate if necessary to fit context window
            let context_tokens = if tokens.len() > max_len {
                &tokens[tokens.len() - max_len..]
            } else {
                &tokens[..]
            };

            // Forward pass
            let logits = model.forward(context_tokens, 1, context_tokens.len(), 0);

            // Get logits for last position
            let vocab_size = model.config.vocab_size;
            let last_pos = context_tokens.len() - 1;
            let logits_slice = logits.as_f32_slice();
            let last_logits = &logits_slice[last_pos * vocab_size..(last_pos + 1) * vocab_size];

            // Apply temperature and sample
            let next_token = if self.config.temperature > 0.0 {
                let scaled_logits: Vec<f32> = last_logits
                    .iter()
                    .map(|&x| x / self.config.temperature)
                    .collect();
                sample_from_logits(&scaled_logits)
            } else {
                // Greedy: argmax
                last_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };

            tokens.push(next_token);

            // Track inference timing
            if i == 0 {
                timer.mark_first_token();
            } else {
                timer.token_generated();
            }

            // Stop at end token if configured
            if self.config.stop_token == Some(next_token) {
                break;
            }
        }

        // Finish timing and log (automatically logs if Logger is enabled)
        let _ = timer.finish();

        // Decode
        tokenizer.decode(&tokens, true).expect("Failed to decode")
    }
}
