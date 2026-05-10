use anyhow::{Context, Result};
use serde::{de::DeserializeOwned, Serialize};
use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

/// Write a single record as pretty JSON, creating parent directories when needed.
pub fn write_json<T: Serialize>(path: impl AsRef<Path>, value: &T) -> Result<()> {
    let path = path.as_ref();
    ensure_parent(path)?;
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    serde_json::to_writer_pretty(BufWriter::new(file), value)
        .with_context(|| format!("writing JSON to {}", path.display()))
}

/// Read a single JSON record.
pub fn read_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("reading JSON from {}", path.display()))
}

/// Write newline-delimited JSON records, creating parent directories when needed.
pub fn write_jsonl<T: Serialize>(path: impl AsRef<Path>, records: &[T]) -> Result<()> {
    let path = path.as_ref();
    ensure_parent(path)?;
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = BufWriter::new(file);

    for record in records {
        serde_json::to_writer(&mut writer, record)
            .with_context(|| format!("serializing JSONL record for {}", path.display()))?;
        writer
            .write_all(b"\n")
            .with_context(|| format!("writing newline to {}", path.display()))?;
    }

    writer
        .flush()
        .with_context(|| format!("flushing {}", path.display()))
}

/// Read newline-delimited JSON records. Empty lines are ignored.
pub fn read_jsonl<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<Vec<T>> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();

    for (index, line) in reader.lines().enumerate() {
        let line =
            line.with_context(|| format!("reading line {} from {}", index + 1, path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        records.push(
            serde_json::from_str(&line)
                .with_context(|| format!("parsing line {} from {}", index + 1, path.display()))?,
        );
    }

    Ok(records)
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Document;
    use std::collections::BTreeMap;

    #[test]
    fn json_and_jsonl_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let doc = Document {
            id: "doc-1".to_string(),
            url: "https://example.com".to_string(),
            title: Some("Example".to_string()),
            body: "body".to_string(),
            fetched_at: None,
            metadata: BTreeMap::new(),
        };

        let json = dir.path().join("nested/doc.json");
        write_json(&json, &doc).unwrap();
        let loaded: Document = read_json(&json).unwrap();
        assert_eq!(loaded, doc);

        let jsonl = dir.path().join("docs.jsonl");
        write_jsonl(&jsonl, std::slice::from_ref(&doc)).unwrap();
        let loaded: Vec<Document> = read_jsonl(&jsonl).unwrap();
        assert_eq!(loaded, vec![doc]);
    }
}
