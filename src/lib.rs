pub mod storage;

use anyhow::{anyhow, Context, Result};
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, TantivyDocument, Value, STORED, TEXT};
use tantivy::{doc, Index, ReloadPolicy};
use url::Url;

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

/// Chunk record accepted by the indexer. Supports both retrieval fixtures
/// (`chunk_id`) and crawler output (`id`).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Chunk {
    #[serde(alias = "id")]
    pub chunk_id: String,
    #[serde(default)]
    pub doc_id: Option<String>,
    pub title: String,
    pub url: String,
    pub source: String,
    pub text: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub score: f32,
    pub source: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CrawlConfig {
    #[serde(default = "default_output_jsonl")]
    pub output_jsonl: PathBuf,
    #[serde(default = "default_chunk_tokens")]
    pub chunk_tokens: usize,
    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,
    #[serde(default)]
    pub seeds: Vec<Seed>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Seed {
    pub url: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub fixture_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CrawledDocument {
    pub doc_id: String,
    pub url: String,
    pub title: String,
    pub source: String,
    pub content_hash: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChunkRecord {
    #[serde(rename = "id")]
    pub chunk_id: String,
    pub doc_id: String,
    pub chunk_index: usize,
    pub url: String,
    pub title: String,
    pub source: String,
    pub text: String,
    pub metadata: ChunkMetadata,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChunkMetadata {
    pub token_start: usize,
    pub token_end: usize,
    pub content_hash: String,
}

#[derive(Clone, Copy)]
struct Fields {
    chunk_id: Field,
    title: Field,
    url: Field,
    source: Field,
    text: Field,
}

pub fn load_crawl_config(path: impl AsRef<Path>) -> Result<CrawlConfig> {
    let path = path.as_ref();
    let raw =
        fs::read_to_string(path).with_context(|| format!("read config {}", path.display()))?;
    let mut config: CrawlConfig = toml::from_str(&raw).context("parse TOML config")?;

    if config.chunk_tokens == 0 {
        return Err(anyhow!("chunk_tokens must be > 0"));
    }
    if config.chunk_overlap >= config.chunk_tokens {
        return Err(anyhow!("chunk_overlap must be smaller than chunk_tokens"));
    }

    absolutize_config_paths(path.parent().unwrap_or(Path::new(".")), &mut config);
    Ok(config)
}

/// Quick v0 crawler/parser/chunker. This intentionally favors deterministic
/// fixture/local runs over web-scale crawl behavior.
pub fn crawl_to_chunks(config: &CrawlConfig) -> Result<Vec<ChunkRecord>> {
    let mut chunks = Vec::new();
    for seed in &config.seeds {
        let raw = read_seed(seed)?;
        let document = parse_crawled_document(seed, &raw)?;
        chunks.extend(chunk_document(
            &document,
            config.chunk_tokens,
            config.chunk_overlap,
        ));
    }
    storage::write_jsonl(&config.output_jsonl, &chunks)?;
    Ok(chunks)
}

pub fn read_seed(seed: &Seed) -> Result<String> {
    if let Some(path) = &seed.fixture_path {
        return fs::read_to_string(path)
            .with_context(|| format!("read fixture {}", path.display()));
    }

    if let Ok(url) = Url::parse(&seed.url) {
        match url.scheme() {
            "file" => {
                let path = url
                    .to_file_path()
                    .map_err(|_| anyhow!("invalid file URL: {}", seed.url))?;
                fs::read_to_string(&path).with_context(|| format!("read file URL {}", seed.url))
            }
            "http" | "https" => {
                let response = reqwest::blocking::get(&seed.url)
                    .with_context(|| format!("fetch {}", seed.url))?;
                response
                    .error_for_status()
                    .with_context(|| format!("HTTP error for {}", seed.url))?
                    .text()
                    .with_context(|| format!("read HTTP body for {}", seed.url))
            }
            other => Err(anyhow!("unsupported URL scheme: {other}")),
        }
    } else {
        fs::read_to_string(&seed.url).with_context(|| format!("read path {}", seed.url))
    }
}

pub fn parse_crawled_document(seed: &Seed, raw: &str) -> Result<CrawledDocument> {
    let looks_html = raw.contains("<html") || raw.contains("<body") || raw.contains("</p>");
    let (detected_title, body_text) = if looks_html {
        extract_html(raw)
    } else {
        (None, raw.to_string())
    };

    let text = clean_text(&body_text);
    if text.is_empty() {
        return Err(anyhow!("empty document after cleaning: {}", seed.url));
    }

    let title = seed
        .title
        .clone()
        .or(detected_title)
        .unwrap_or_else(|| seed.url.clone());
    let source = seed
        .source
        .clone()
        .unwrap_or_else(|| infer_source(&seed.url));
    let content_hash = short_hash(&text);
    let doc_id = short_hash(&format!("{}\n{}", seed.url, content_hash));

    Ok(CrawledDocument {
        doc_id,
        url: seed.url.clone(),
        title,
        source,
        content_hash,
        text,
    })
}

pub fn extract_html(raw: &str) -> (Option<String>, String) {
    let document = Html::parse_document(raw);
    let title = Selector::parse("title")
        .ok()
        .and_then(|selector| document.select(&selector).next())
        .map(|element| clean_text(&element.text().collect::<Vec<_>>().join(" ")))
        .filter(|title| !title.is_empty());

    let mut text_parts = Vec::new();
    for selector in ["article", "main", "body"] {
        if let Ok(parsed) = Selector::parse(selector) {
            if let Some(element) = document.select(&parsed).next() {
                text_parts.extend(element.text().map(str::to_string));
                break;
            }
        }
    }

    (title, text_parts.join(" "))
}

pub fn clean_text(input: &str) -> String {
    input
        .replace('\u{00a0}', " ")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn chunk_document(
    document: &CrawledDocument,
    chunk_tokens: usize,
    overlap: usize,
) -> Vec<ChunkRecord> {
    let words: Vec<&str> = document.text.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }

    let mut records = Vec::new();
    let step = chunk_tokens.saturating_sub(overlap).max(1);
    let mut start = 0;
    let mut chunk_index = 0;

    while start < words.len() {
        let end = (start + chunk_tokens).min(words.len());
        let text = words[start..end].join(" ");
        records.push(ChunkRecord {
            chunk_id: format!("{}-{:04}", document.doc_id, chunk_index),
            doc_id: document.doc_id.clone(),
            chunk_index,
            url: document.url.clone(),
            title: document.title.clone(),
            source: document.source.clone(),
            text,
            metadata: ChunkMetadata {
                token_start: start,
                token_end: end,
                content_hash: document.content_hash.clone(),
            },
        });
        if end == words.len() {
            break;
        }
        start += step;
        chunk_index += 1;
    }

    records
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

fn absolutize_config_paths(base: &Path, config: &mut CrawlConfig) {
    if config.output_jsonl.is_relative() {
        config.output_jsonl = base.join(&config.output_jsonl);
    }
    for seed in &mut config.seeds {
        if let Some(path) = &seed.fixture_path {
            if path.is_relative() {
                seed.fixture_path = Some(base.join(path));
            }
        }
    }
}

fn infer_source(url: &str) -> String {
    Url::parse(url)
        .ok()
        .and_then(|parsed| parsed.host_str().map(str::to_string))
        .unwrap_or_else(|| "local".to_string())
}

fn short_hash(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    format!("{digest:x}")[..16].to_string()
}

fn default_output_jsonl() -> PathBuf {
    PathBuf::from("data/chunks.jsonl")
}

fn default_chunk_tokens() -> usize {
    220
}

fn default_chunk_overlap() -> usize {
    40
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn seed() -> Seed {
        Seed {
            url: "https://example.com/nvidia-blackwell".to_string(),
            title: None,
            source: Some("Example Semi".to_string()),
            fixture_path: None,
        }
    }

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

    #[test]
    fn cleans_whitespace() {
        assert_eq!(clean_text(" A\n\n B\t C\u{00a0}D "), "A B C D");
    }

    #[test]
    fn parses_html_title_and_body() {
        let raw = r#"<html><head><title>Blackwell Notes</title></head><body><nav>ignore</nav><main><h1>NVIDIA Blackwell</h1><p>GB200 uses NVLink.</p></main></body></html>"#;
        let doc = parse_crawled_document(&seed(), raw).unwrap();
        assert_eq!(doc.title, "Blackwell Notes");
        assert!(doc.text.contains("NVIDIA Blackwell"));
        assert!(doc.text.contains("GB200 uses NVLink."));
    }

    #[test]
    fn chunks_with_overlap_and_metadata() {
        let document = CrawledDocument {
            doc_id: "doc123".to_string(),
            url: "https://example.com".to_string(),
            title: "Example".to_string(),
            source: "Example".to_string(),
            content_hash: "hash".to_string(),
            text: "one two three four five six seven".to_string(),
        };
        let chunks = chunk_document(&document, 3, 1);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "one two three");
        assert_eq!(chunks[1].text, "three four five");
        assert_eq!(chunks[2].metadata.token_start, 4);
        assert_eq!(chunks[2].metadata.token_end, 7);
    }

    #[test]
    fn fixture_pipeline_writes_jsonl() {
        let dir = tempdir().unwrap();
        let fixture = dir.path().join("amd-mi300.html");
        fs::write(
            &fixture,
            "<html><head><title>AMD MI300</title></head><body><article>MI300 combines CPU GPU and HBM chiplets for AI workloads.</article></body></html>",
        )
        .unwrap();
        let output = dir.path().join("chunks.jsonl");
        let config = CrawlConfig {
            output_jsonl: output.clone(),
            chunk_tokens: 6,
            chunk_overlap: 2,
            seeds: vec![Seed {
                url: "https://example.com/amd-mi300".to_string(),
                title: None,
                source: Some("Fixture".to_string()),
                fixture_path: Some(fixture),
            }],
        };

        let chunks = crawl_to_chunks(&config).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].title, "AMD MI300");
        assert!(chunks[0].text.contains("MI300"));
        assert_eq!(
            storage::read_jsonl::<Chunk>(&output).unwrap().len(),
            chunks.len()
        );
    }
}
