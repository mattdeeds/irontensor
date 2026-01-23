//! JSON Lines writer with buffering.

use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

/// A buffered writer for JSON Lines format.
///
/// Each record is serialized as a single JSON object on its own line.
/// The writer buffers writes and flushes periodically for efficiency.
pub struct JsonlWriter {
    writer: BufWriter<File>,
    records_since_flush: usize,
    flush_interval: usize,
}

impl JsonlWriter {
    /// Create a new JSONL writer.
    ///
    /// If the file exists, new records will be appended.
    /// If the file doesn't exist, it will be created.
    ///
    /// # Arguments
    /// * `path` - Path to the JSONL file
    /// * `flush_interval` - Flush to disk after this many records (0 = flush every record)
    pub fn new<P: AsRef<Path>>(path: P, flush_interval: usize) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;

        Ok(Self {
            writer: BufWriter::new(file),
            records_since_flush: 0,
            flush_interval: if flush_interval == 0 { 1 } else { flush_interval },
        })
    }

    /// Write a single record to the file.
    ///
    /// The record is serialized to JSON and written on a single line.
    /// Automatically flushes based on the configured flush interval.
    pub fn write<T: Serialize>(&mut self, record: &T) -> std::io::Result<()> {
        let json = serde_json::to_string(record)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        writeln!(self.writer, "{}", json)?;

        self.records_since_flush += 1;
        if self.records_since_flush >= self.flush_interval {
            self.flush()?;
        }

        Ok(())
    }

    /// Flush buffered data to disk.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()?;
        self.records_since_flush = 0;
        Ok(())
    }
}

impl Drop for JsonlWriter {
    fn drop(&mut self) {
        // Best-effort flush on drop
        let _ = self.flush();
    }
}

/// Write a single JSON object to a file (for summary files).
pub fn write_json_file<P: AsRef<Path>, T: Serialize>(path: P, data: &T) -> std::io::Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, data)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::io::BufRead;

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct TestRecord {
        id: u32,
        value: String,
    }

    #[test]
    fn test_jsonl_writer_roundtrip() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_jsonl.jsonl");

        // Write records
        {
            let mut writer = JsonlWriter::new(&path, 1).unwrap();
            writer.write(&TestRecord { id: 1, value: "first".into() }).unwrap();
            writer.write(&TestRecord { id: 2, value: "second".into() }).unwrap();
        }

        // Read and verify
        let file = File::open(&path).unwrap();
        let reader = std::io::BufReader::new(file);
        let records: Vec<TestRecord> = reader
            .lines()
            .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
            .collect();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0], TestRecord { id: 1, value: "first".into() });
        assert_eq!(records[1], TestRecord { id: 2, value: "second".into() });

        // Cleanup
        std::fs::remove_file(&path).ok();
    }
}
