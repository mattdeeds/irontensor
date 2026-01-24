use std::fs::File;
use std::io::Write;
use std::path::Path;

use memmap2::Mmap;

use crate::profile::{timed, OpCategory};

/// Memory-mapped tokenized dataset
///
/// Efficiently loads pre-tokenized data using mmap for near-zero overhead.
/// The file format is simple:
/// - First 8 bytes: number of tokens (u64, little-endian)
/// - Remaining bytes: tokens as u32 (little-endian)
///
/// This format allows direct memory mapping with proper alignment.
pub struct TokenDataset {
    /// Memory-mapped data
    mmap: Mmap,
    /// Number of tokens in the dataset
    num_tokens: usize,
    /// Sequence length for batching
    seq_len: usize,
}

impl TokenDataset {
    /// Open an existing tokenized dataset file
    pub fn open<P: AsRef<Path>>(path: P, seq_len: usize) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "File too small to contain header",
            ));
        }

        // Read number of tokens from header
        let num_tokens = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;

        // Verify file size
        let expected_size = 8 + num_tokens * 4;
        if mmap.len() < expected_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "File size mismatch: expected {} bytes, got {}",
                    expected_size,
                    mmap.len()
                ),
            ));
        }

        Ok(Self {
            mmap,
            num_tokens,
            seq_len,
        })
    }

    /// Create a new tokenized dataset file from token IDs
    pub fn create<P: AsRef<Path>>(path: P, tokens: &[u32]) -> std::io::Result<()> {
        let mut file = File::create(path)?;

        // Write header: number of tokens
        let num_tokens = tokens.len() as u64;
        file.write_all(&num_tokens.to_le_bytes())?;

        // Write tokens
        for &token in tokens {
            file.write_all(&token.to_le_bytes())?;
        }

        file.sync_all()?;
        Ok(())
    }

    /// Get the number of tokens in the dataset
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Get the number of complete sequences that can be formed
    pub fn num_sequences(&self) -> usize {
        if self.num_tokens == 0 {
            return 0;
        }
        // We need seq_len + 1 tokens for input and target
        (self.num_tokens - 1) / self.seq_len
    }

    /// Get the sequence length
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Get a token at a specific index
    pub fn get_token(&self, idx: usize) -> u32 {
        assert!(idx < self.num_tokens, "Token index out of bounds");
        let offset = 8 + idx * 4;
        u32::from_le_bytes(self.mmap[offset..offset + 4].try_into().unwrap())
    }

    /// Get a slice of tokens (copies into a Vec)
    pub fn get_tokens(&self, start: usize, len: usize) -> Vec<u32> {
        assert!(
            start + len <= self.num_tokens,
            "Token range out of bounds"
        );
        (start..start + len).map(|i| self.get_token(i)).collect()
    }

    /// Get a training batch (input_ids, target_ids)
    ///
    /// - `seq_idx`: Sequence index (0 to num_sequences - 1)
    ///
    /// Returns (input_ids, target_ids) where:
    /// - input_ids: tokens[seq_idx * seq_len : seq_idx * seq_len + seq_len]
    /// - target_ids: tokens[seq_idx * seq_len + 1 : seq_idx * seq_len + seq_len + 1]
    pub fn get_batch(&self, seq_idx: usize) -> (Vec<u32>, Vec<u32>) {
        let start = seq_idx * self.seq_len;
        assert!(
            start + self.seq_len + 1 <= self.num_tokens,
            "Sequence index out of bounds"
        );

        let input_ids = self.get_tokens(start, self.seq_len);
        let target_ids = self.get_tokens(start + 1, self.seq_len);

        (input_ids, target_ids)
    }

    /// Get multiple batches for mini-batch training
    ///
    /// Returns (input_ids, target_ids) where each has shape [batch_size * seq_len]
    pub fn get_batches(&self, seq_indices: &[usize]) -> (Vec<u32>, Vec<u32>) {
        let batch_size = seq_indices.len();
        let total_tokens = batch_size * self.seq_len * 2; // input + target tokens
        let _timer = timed(OpCategory::DataLoading, total_tokens);

        let mut input_ids = Vec::with_capacity(batch_size * self.seq_len);
        let mut target_ids = Vec::with_capacity(batch_size * self.seq_len);

        for &idx in seq_indices {
            let (inp, tgt) = self.get_batch(idx);
            input_ids.extend(inp);
            target_ids.extend(tgt);
        }

        (input_ids, target_ids)
    }
}

/// Simple iterator over dataset batches
pub struct DatasetIterator<'a> {
    dataset: &'a TokenDataset,
    current_idx: usize,
    indices: Vec<usize>,
    batch_size: usize,
}

impl<'a> DatasetIterator<'a> {
    /// Create a new iterator over the dataset
    pub fn new(dataset: &'a TokenDataset, batch_size: usize, shuffle: bool) -> Self {
        let num_sequences = dataset.num_sequences();
        let mut indices: Vec<usize> = (0..num_sequences).collect();

        if shuffle {
            // Simple shuffle using a deterministic pseudo-random permutation
            for i in (1..indices.len()).rev() {
                let j = ((i as f32 * 0.618033988749895) as usize) % (i + 1);
                indices.swap(i, j);
            }
        }

        Self {
            dataset,
            current_idx: 0,
            indices,
            batch_size,
        }
    }

    /// Reset the iterator
    pub fn reset(&mut self) {
        self.current_idx = 0;
    }

    /// Shuffle indices for next epoch
    pub fn shuffle(&mut self, seed: u64) {
        for i in (1..self.indices.len()).rev() {
            // Simple LCG-based shuffle
            let hash = seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64);
            let j = (hash as usize) % (i + 1);
            self.indices.swap(i, j);
        }
        self.current_idx = 0;
    }
}

impl<'a> Iterator for DatasetIterator<'a> {
    type Item = (Vec<u32>, Vec<u32>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current_idx..end_idx];
        self.current_idx = end_idx;

        Some(self.dataset.get_batches(batch_indices))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn test_dataset_create_and_open() {
        let path = temp_dir().join("test_dataset.bin");

        // Create dataset
        let tokens: Vec<u32> = (0..1000).collect();
        TokenDataset::create(&path, &tokens).unwrap();

        // Open and verify
        let dataset = TokenDataset::open(&path, 32).unwrap();
        assert_eq!(dataset.num_tokens(), 1000);

        // Verify first few tokens
        for i in 0..10 {
            assert_eq!(dataset.get_token(i), i as u32);
        }

        // Cleanup
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_dataset_get_batch() {
        let path = temp_dir().join("test_batch.bin");

        let tokens: Vec<u32> = (0..100).collect();
        TokenDataset::create(&path, &tokens).unwrap();

        let dataset = TokenDataset::open(&path, 10).unwrap();

        let (input, target) = dataset.get_batch(0);
        assert_eq!(input.len(), 10);
        assert_eq!(target.len(), 10);

        // Input should be [0, 1, 2, ..., 9]
        // Target should be [1, 2, 3, ..., 10]
        for i in 0..10 {
            assert_eq!(input[i], i as u32);
            assert_eq!(target[i], (i + 1) as u32);
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_dataset_num_sequences() {
        let path = temp_dir().join("test_seq.bin");

        // 101 tokens with seq_len=10 means we can get 10 complete sequences
        // (each needs 11 consecutive tokens: 10 for input, 1 for last target)
        let tokens: Vec<u32> = (0..101).collect();
        TokenDataset::create(&path, &tokens).unwrap();

        let dataset = TokenDataset::open(&path, 10).unwrap();
        assert_eq!(dataset.num_sequences(), 10);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_dataset_iterator() {
        let path = temp_dir().join("test_iter.bin");

        let tokens: Vec<u32> = (0..100).collect();
        TokenDataset::create(&path, &tokens).unwrap();

        let dataset = TokenDataset::open(&path, 10).unwrap();
        let mut iter = DatasetIterator::new(&dataset, 2, false);

        let mut count = 0;
        for (input, target) in &mut iter {
            assert!(input.len() <= 20); // batch_size * seq_len
            assert!(target.len() <= 20);
            count += 1;
        }

        // 9 sequences with batch_size=2 means 5 batches (last one has 1 sequence)
        assert_eq!(count, 5);

        std::fs::remove_file(path).ok();
    }
}
