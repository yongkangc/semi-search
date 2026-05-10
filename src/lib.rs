pub mod storage;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, TantivyDocument, Value, STORED, TEXT};
use tantivy::{doc, Index, ReloadPolicy};

const FIELD_CHUNK_ID: &str = "chunk_id";
const FIELD_TITLE: &str = "title";
const FIELD_URL: &str = "url";
const FIELD_SOURCE: &str = "source";
const FIELD_TEXT: &str = "text";

/// Source document after crawl/ingest and before chunking.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Document {
    pub id: String,
    pub url: String,
    pub title: Option<String>,
    pub body: String,
    pub fetched_at: Option<String>,
    #[serde(default)]
    pub metadata: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Chunk {
    pub chunk_id: String,
    pub title: String,
    pub url: String,
    pub source: String,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub score: f32,
    pub source: String,
}

#[derive(Clone, Copy)]
struct Fields {
    chunk_id: Field,
    title: Field,
    url: Field,
    source: Field,
    text: Field,
}

fn build_schema() -> (Schema, Fields) {
    let mut builder = Schema::builder();
    let chunk_id = builder.add_text_field(FIELD_CHUNK_ID, STORED);
    let title = builder.add_text_field(FIELD_TITLE, TEXT | STORED);
    let url = builder.add_text_field(FIELD_URL, STORED);
    let source = builder.add_text_field(FIELD_SOURCE, TEXT | STORED);
    let text = builder.add_text_field(FIELD_TEXT, TEXT | STORED);
    let schema = builder.build();
    (
        schema,
        Fields {
            chunk_id,
            title,
            url,
            source,
            text,
        },
    )
}

fn fields_from_schema(schema: &Schema) -> Result<Fields> {
    Ok(Fields {
        chunk_id: schema.get_field(FIELD_CHUNK_ID)?,
        title: schema.get_field(FIELD_TITLE)?,
        url: schema.get_field(FIELD_URL)?,
        source: schema.get_field(FIELD_SOURCE)?,
        text: schema.get_field(FIELD_TEXT)?,
    })
}

pub fn index_chunks(chunks_path: impl AsRef<Path>, index_dir: impl AsRef<Path>) -> Result<usize> {
    let chunks_path = chunks_path.as_ref();
    let index_dir = index_dir.as_ref();

    if index_dir.exists() {
        fs::remove_dir_all(index_dir)
            .with_context(|| format!("clearing {}", index_dir.display()))?;
    }
    fs::create_dir_all(index_dir).with_context(|| format!("creating {}", index_dir.display()))?;

    let (schema, fields) = build_schema();
    let index = Index::create_in_dir(index_dir, schema)
        .with_context(|| format!("creating Tantivy index at {}", index_dir.display()))?;
    let mut writer = index.writer(50_000_000)?;

    let file =
        File::open(chunks_path).with_context(|| format!("opening {}", chunks_path.display()))?;
    let reader = BufReader::new(file);
    let mut count = 0usize;

    for (line_no, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("reading line {}", line_no + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let chunk: Chunk = serde_json::from_str(&line)
            .with_context(|| format!("parsing JSONL chunk at line {}", line_no + 1))?;
        writer.add_document(doc!(
            fields.chunk_id => chunk.chunk_id,
            fields.title => chunk.title,
            fields.url => chunk.url,
            fields.source => chunk.source,
            fields.text => chunk.text,
        ))?;
        count += 1;
    }

    writer.commit()?;
    writer.wait_merging_threads()?;
    Ok(count)
}

pub fn search_index(
    index_dir: impl AsRef<Path>,
    query_text: &str,
    limit: usize,
) -> Result<Vec<SearchResult>> {
    let index_dir = index_dir.as_ref();
    let index = Index::open_in_dir(index_dir)
        .with_context(|| format!("opening Tantivy index at {}", index_dir.display()))?;
    let schema = index.schema();
    let fields = fields_from_schema(&schema)?;
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()?;
    let searcher = reader.searcher();
    let query_parser =
        QueryParser::for_index(&index, vec![fields.title, fields.text, fields.source]);
    let query = query_parser
        .parse_query(query_text)
        .with_context(|| format!("parsing query {query_text:?}"))?;

    let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;
    let mut results = Vec::with_capacity(top_docs.len());
    for (score, address) in top_docs {
        let doc: TantivyDocument = searcher.doc(address)?;
        let title = stored_text(&doc, fields.title);
        let url = stored_text(&doc, fields.url);
        let source = stored_text(&doc, fields.source);
        let text = stored_text(&doc, fields.text);
        results.push(SearchResult {
            title,
            url,
            snippet: make_snippet(&text, query_text, 220),
            score,
            source,
        });
    }
    Ok(results)
}

fn stored_text(doc: &TantivyDocument, field: Field) -> String {
    doc.get_first(field)
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_string()
}

fn make_snippet(text: &str, query: &str, max_chars: usize) -> String {
    let lower_text = text.to_lowercase();
    let terms: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|term| term.len() > 2)
        .map(str::to_lowercase)
        .collect();
    let hit = terms.iter().find_map(|term| lower_text.find(term));
    let start = hit
        .map(|pos| pos.saturating_sub(max_chars / 3))
        .unwrap_or(0);
    let mut snippet: String = text.chars().skip(start).take(max_chars).collect();
    if start > 0 {
        snippet = format!("…{snippet}");
    }
    if text.chars().count() > start + snippet.chars().count() {
        snippet.push('…');
    }
    snippet
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snippet_centers_on_query_term() {
        let snippet = make_snippet(
            "NVIDIA Blackwell uses NVLink and a high-bandwidth memory subsystem for AI training.",
            "Blackwell training",
            120,
        );
        assert!(snippet.contains("Blackwell"));
        assert!(snippet.contains("training"));
    }
}
