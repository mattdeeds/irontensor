use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::nn::{GPTModel, ModelConfig};
use crate::precision::Precision;
use crate::tensor::Tensor;

/// Magic number for checkpoint file format
const CHECKPOINT_MAGIC: u32 = 0x49524F4E; // "IRON" in hex

/// Checkpoint file version
const CHECKPOINT_VERSION: u32 = 1;

/// Training checkpoint containing model weights and optimizer state
#[derive(Debug)]
pub struct Checkpoint {
    /// Model configuration
    pub config: ModelConfig,
    /// Current training step
    pub step: usize,
    /// Current epoch
    pub epoch: usize,
    /// Best validation loss seen so far
    pub best_val_loss: f32,
    /// Learning rate at checkpoint
    pub learning_rate: f32,
}

impl Checkpoint {
    /// Create a new checkpoint
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            step: 0,
            epoch: 0,
            best_val_loss: f32::INFINITY,
            learning_rate: 0.0,
        }
    }
}

/// Save a tensor to a writer
fn save_tensor<W: Write>(tensor: &Tensor, writer: &mut W) -> std::io::Result<()> {
    let shape = tensor.shape();
    let ndim = shape.len() as u32;
    writer.write_all(&ndim.to_le_bytes())?;

    for &dim in shape {
        writer.write_all(&(dim as u64).to_le_bytes())?;
    }

    let data = tensor.as_f32_slice();
    for &val in data {
        writer.write_all(&val.to_le_bytes())?;
    }

    Ok(())
}

/// Load a tensor from a reader
fn load_tensor<R: Read>(reader: &mut R) -> std::io::Result<Tensor> {
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    reader.read_exact(&mut buf4)?;
    let ndim = u32::from_le_bytes(buf4) as usize;

    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        reader.read_exact(&mut buf8)?;
        shape.push(u64::from_le_bytes(buf8) as usize);
    }

    let numel: usize = shape.iter().product();
    let mut data = Vec::with_capacity(numel);

    for _ in 0..numel {
        reader.read_exact(&mut buf4)?;
        data.push(f32::from_le_bytes(buf4));
    }

    Ok(Tensor::from_f32_slice(&data, &shape))
}

/// Save model weights to a file
pub fn save_model_weights<P: AsRef<Path>>(
    path: P,
    model: &GPTModel,
    checkpoint: &Checkpoint,
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write header
    writer.write_all(&CHECKPOINT_MAGIC.to_le_bytes())?;
    writer.write_all(&CHECKPOINT_VERSION.to_le_bytes())?;

    // Write checkpoint metadata
    writer.write_all(&(checkpoint.step as u64).to_le_bytes())?;
    writer.write_all(&(checkpoint.epoch as u64).to_le_bytes())?;
    writer.write_all(&checkpoint.best_val_loss.to_le_bytes())?;
    writer.write_all(&checkpoint.learning_rate.to_le_bytes())?;

    // Write model config
    write_model_config(&mut writer, &checkpoint.config)?;

    // Write model weights
    // Embedding
    save_tensor(&model.embed_tokens, &mut writer)?;

    // Transformer layers
    for layer in &model.layers {
        // Attention weights (wq, wk, wv, wo are Linear layers)
        save_tensor(&layer.attention.wq.weight, &mut writer)?;
        save_tensor(&layer.attention.wk.weight, &mut writer)?;
        save_tensor(&layer.attention.wv.weight, &mut writer)?;
        save_tensor(&layer.attention.wo.weight, &mut writer)?;

        // FFN weights (w_gate, w_up, w_down are Linear layers)
        save_tensor(&layer.ffn.w_gate.weight, &mut writer)?;
        save_tensor(&layer.ffn.w_up.weight, &mut writer)?;
        save_tensor(&layer.ffn.w_down.weight, &mut writer)?;

        // Layer norms
        save_tensor(&layer.attn_norm, &mut writer)?;
        save_tensor(&layer.ffn_norm, &mut writer)?;
    }

    // Final norm
    save_tensor(&model.final_norm, &mut writer)?;

    // Note: output_weight is private and may be tied to embed_tokens
    // We save a flag indicating if weights are tied
    let tie_weights = checkpoint.config.tie_weights;
    writer.write_all(&(if tie_weights { 1u8 } else { 0u8 }).to_le_bytes())?;

    writer.flush()?;
    Ok(())
}

/// Load model weights from a file
pub fn load_model_weights<P: AsRef<Path>>(path: P) -> std::io::Result<(GPTModel, Checkpoint)> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    // Read and verify header
    reader.read_exact(&mut buf4)?;
    let magic = u32::from_le_bytes(buf4);
    if magic != CHECKPOINT_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid checkpoint file (bad magic number)",
        ));
    }

    reader.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version != CHECKPOINT_VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unsupported checkpoint version: {}", version),
        ));
    }

    // Read checkpoint metadata
    reader.read_exact(&mut buf8)?;
    let step = u64::from_le_bytes(buf8) as usize;

    reader.read_exact(&mut buf8)?;
    let epoch = u64::from_le_bytes(buf8) as usize;

    reader.read_exact(&mut buf4)?;
    let best_val_loss = f32::from_le_bytes(buf4);

    reader.read_exact(&mut buf4)?;
    let learning_rate = f32::from_le_bytes(buf4);

    // Read model config
    let config = read_model_config(&mut reader)?;

    // Create model with the loaded config
    let mut model = GPTModel::new(config.clone());

    // Load model weights
    // Embedding
    model.embed_tokens = load_tensor(&mut reader)?;

    // Transformer layers
    for layer in &mut model.layers {
        layer.attention.wq.weight = load_tensor(&mut reader)?;
        layer.attention.wk.weight = load_tensor(&mut reader)?;
        layer.attention.wv.weight = load_tensor(&mut reader)?;
        layer.attention.wo.weight = load_tensor(&mut reader)?;

        layer.ffn.w_gate.weight = load_tensor(&mut reader)?;
        layer.ffn.w_up.weight = load_tensor(&mut reader)?;
        layer.ffn.w_down.weight = load_tensor(&mut reader)?;

        layer.attn_norm = load_tensor(&mut reader)?;
        layer.ffn_norm = load_tensor(&mut reader)?;
    }

    // Final norm
    model.final_norm = load_tensor(&mut reader)?;

    // Read tie_weights flag (we already used this in config)
    let mut tie_flag = [0u8; 1];
    reader.read_exact(&mut tie_flag)?;

    let checkpoint = Checkpoint {
        config,
        step,
        epoch,
        best_val_loss,
        learning_rate,
    };

    Ok((model, checkpoint))
}

fn write_model_config<W: Write>(writer: &mut W, config: &ModelConfig) -> std::io::Result<()> {
    writer.write_all(&(config.vocab_size as u64).to_le_bytes())?;
    writer.write_all(&(config.hidden_dim as u64).to_le_bytes())?;
    writer.write_all(&(config.num_layers as u64).to_le_bytes())?;
    writer.write_all(&(config.num_heads as u64).to_le_bytes())?;
    writer.write_all(&(config.num_kv_heads as u64).to_le_bytes())?;
    writer.write_all(&(config.intermediate_dim as u64).to_le_bytes())?;
    writer.write_all(&config.norm_eps.to_le_bytes())?;
    writer.write_all(&config.rope_base.to_le_bytes())?;
    writer.write_all(&(config.max_seq_len as u64).to_le_bytes())?;
    writer.write_all(&(if config.tie_weights { 1u8 } else { 0u8 }).to_le_bytes())?;
    // Precision: 0 = FP32, 1 = FP16, 2 = BF16
    let precision_byte = match config.precision {
        Precision::FP32 => 0u8,
        Precision::FP16 => 1u8,
        Precision::BF16 => 2u8,
    };
    writer.write_all(&[precision_byte])?;
    Ok(())
}

fn read_model_config<R: Read>(reader: &mut R) -> std::io::Result<ModelConfig> {
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];
    let mut buf1 = [0u8; 1];

    reader.read_exact(&mut buf8)?;
    let vocab_size = u64::from_le_bytes(buf8) as usize;

    reader.read_exact(&mut buf8)?;
    let hidden_dim = u64::from_le_bytes(buf8) as usize;

    reader.read_exact(&mut buf8)?;
    let num_layers = u64::from_le_bytes(buf8) as usize;

    reader.read_exact(&mut buf8)?;
    let num_heads = u64::from_le_bytes(buf8) as usize;

    reader.read_exact(&mut buf8)?;
    let num_kv_heads = u64::from_le_bytes(buf8) as usize;

    reader.read_exact(&mut buf8)?;
    let intermediate_dim = u64::from_le_bytes(buf8) as usize;

    reader.read_exact(&mut buf4)?;
    let norm_eps = f32::from_le_bytes(buf4);

    reader.read_exact(&mut buf4)?;
    let rope_base = f32::from_le_bytes(buf4);

    reader.read_exact(&mut buf8)?;
    let max_seq_len = u64::from_le_bytes(buf8) as usize;

    reader.read_exact(&mut buf1)?;
    let tie_weights = buf1[0] == 1;

    // Read precision (default to FP32 for backward compatibility)
    let precision = match reader.read_exact(&mut buf1) {
        Ok(()) => match buf1[0] {
            1 => Precision::FP16,
            2 => Precision::BF16,
            _ => Precision::FP32,
        },
        Err(_) => Precision::FP32, // Old format without precision
    };

    Ok(ModelConfig {
        vocab_size,
        hidden_dim,
        num_layers,
        num_heads,
        num_kv_heads,
        intermediate_dim,
        norm_eps,
        rope_base,
        max_seq_len,
        tie_weights,
        precision,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn test_save_load_tensor() {
        let tensor = Tensor::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

        let mut buffer = Vec::new();
        save_tensor(&tensor, &mut buffer).unwrap();

        let mut cursor = std::io::Cursor::new(buffer);
        let loaded = load_tensor(&mut cursor).unwrap();

        assert_eq!(tensor.shape(), loaded.shape());
        assert_eq!(tensor.as_f32_slice(), loaded.as_f32_slice());
    }

    #[test]
    fn test_save_load_model() {
        let config = ModelConfig {
            vocab_size: 100,
            hidden_dim: 32,
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 2,
            intermediate_dim: 64,
            norm_eps: 1e-5,
            rope_base: 10000.0,
            max_seq_len: 128,
            tie_weights: true,
            precision: Precision::FP32,
        };

        let model = GPTModel::new(config.clone());
        let checkpoint = Checkpoint {
            config: config.clone(),
            step: 100,
            epoch: 5,
            best_val_loss: 2.5,
            learning_rate: 1e-4,
        };

        let path = temp_dir().join("test_model.bin");
        save_model_weights(&path, &model, &checkpoint).unwrap();

        let (loaded_model, loaded_checkpoint) = load_model_weights(&path).unwrap();

        // Verify checkpoint metadata
        assert_eq!(loaded_checkpoint.step, 100);
        assert_eq!(loaded_checkpoint.epoch, 5);
        assert!((loaded_checkpoint.best_val_loss - 2.5).abs() < 1e-5);

        // Verify config
        assert_eq!(loaded_checkpoint.config.vocab_size, config.vocab_size);
        assert_eq!(loaded_checkpoint.config.hidden_dim, config.hidden_dim);
        assert_eq!(loaded_checkpoint.config.num_layers, config.num_layers);

        // Verify some weights match
        assert_eq!(
            model.embed_tokens.as_f32_slice(),
            loaded_model.embed_tokens.as_f32_slice()
        );

        // Cleanup
        std::fs::remove_file(path).ok();
    }
}
