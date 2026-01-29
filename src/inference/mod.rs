mod generator;
mod sampling;

pub use generator::{GeneratorConfig, TextGenerator};
pub use sampling::{sample_from_logits, softmax};
