use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Source document after crawl/ingest and before chunking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub url: String,
    pub title: Option<String>,
    pub body: String,
    pub fetched_at: Option<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

impl Document {
    pub fn new(id: impl Into<String>, url: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            url: url.into(),
            title: None,
            body: body.into(),
            fetched_at: None,
            metadata: BTreeMap::new(),
        }
    }
}

/// Searchable passage tied back to a source document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub document_id: String,
    pub text: String,
    pub start_byte: usize,
    pub end_byte: usize,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

impl Chunk {
    pub fn new(
        id: impl Into<String>,
        document_id: impl Into<String>,
        text: impl Into<String>,
        start_byte: usize,
        end_byte: usize,
    ) -> Self {
        Self {
            id: id.into(),
            document_id: document_id.into(),
            text: text.into(),
            start_byte,
            end_byte,
            metadata: BTreeMap::new(),
        }
    }
}

/// Agent-facing result returned by retrieval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk_id: String,
    pub document_id: String,
    pub title: Option<String>,
    pub url: String,
    pub snippet: String,
    pub score: f32,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

impl SearchResult {
    pub fn new(
        chunk_id: impl Into<String>,
        document_id: impl Into<String>,
        url: impl Into<String>,
        snippet: impl Into<String>,
        score: f32,
    ) -> Self {
        Self {
            chunk_id: chunk_id.into(),
            document_id: document_id.into(),
            title: None,
            url: url.into(),
            snippet: snippet.into(),
            score,
            metadata: BTreeMap::new(),
        }
    }
}
