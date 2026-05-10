pub mod storage;

use anyhow::{anyhow, Context, Result};
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter};
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
const FIELD_COMPANIES: &str = "companies";
const FIELD_SOURCE_TYPE: &str = "source_type";
const FIELD_DOMAIN: &str = "domain";
const FIELD_PUBLISHED_AT: &str = "published_at";
const FIELD_TOPICS: &str = "topics";
const VECTOR_STORE_FILE: &str = "chunks.vector.json";
pub const EMBEDDING_DIMS: usize = 128;
pub const LOCAL_HASH_EMBEDDING_MODEL: &str = "local-hash-bow-v1";

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
    pub companies: Vec<String>,
    #[serde(default)]
    pub source_type: Option<String>,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub published_at: Option<String>,
    #[serde(default)]
    pub topics: Vec<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub chunk_id: String,
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub score: f32,
    pub source: String,
    pub companies: Vec<String>,
    pub source_type: Option<String>,
    pub domain: Option<String>,
    pub published_at: Option<String>,
    pub topics: Vec<String>,
    pub score_components: ScoreComponents,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ScoreComponents {
    pub bm25: f32,
    pub bm25_normalized: f32,
    pub vector: f32,
    pub vector_normalized: f32,
    pub final_score: f32,
}

#[derive(Debug, Clone, Default)]
pub struct SearchFilters {
    pub company: Option<String>,
    pub source_type: Option<String>,
    pub domain: Option<String>,
    pub after: Option<String>,
    pub topic: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VectorRecord {
    chunk_id: String,
    title: String,
    url: String,
    source: String,
    text: String,
    companies: Vec<String>,
    source_type: Option<String>,
    domain: Option<String>,
    published_at: Option<String>,
    topics: Vec<String>,
    embedding: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct EmbeddingModelMetadata {
    pub model: String,
    pub version: String,
    pub dimensions: usize,
    pub method: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbeddedChunk {
    #[serde(flatten)]
    pub chunk: Chunk,
    pub embedding: Vec<f32>,
    pub embedding_model: EmbeddingModelMetadata,
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
    pub companies: Vec<String>,
    pub source_type: Option<String>,
    pub domain: Option<String>,
    pub published_at: Option<String>,
    pub topics: Vec<String>,
    pub metadata: ChunkMetadata,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChunkMetadata {
    pub token_start: usize,
    pub token_end: usize,
    pub content_hash: String,
    pub companies: Vec<String>,
    pub source_type: Option<String>,
    pub domain: Option<String>,
    pub published_at: Option<String>,
    pub topics: Vec<String>,
}

#[derive(Clone, Copy)]
struct Fields {
    chunk_id: Field,
    title: Field,
    url: Field,
    source: Field,
    text: Field,
    companies: Field,
    source_type: Field,
    domain: Field,
    published_at: Field,
    topics: Field,
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
        let inferred = infer_metadata(&document.title, &document.url, &document.source, &text);
        records.push(ChunkRecord {
            chunk_id: format!("{}-{:04}", document.doc_id, chunk_index),
            doc_id: document.doc_id.clone(),
            chunk_index,
            url: document.url.clone(),
            title: document.title.clone(),
            source: document.source.clone(),
            text,
            companies: inferred.companies.clone(),
            source_type: inferred.source_type.clone(),
            domain: inferred.domain.clone(),
            published_at: inferred.published_at.clone(),
            topics: inferred.topics.clone(),
            metadata: ChunkMetadata {
                token_start: start,
                token_end: end,
                content_hash: document.content_hash.clone(),
                companies: inferred.companies,
                source_type: inferred.source_type,
                domain: inferred.domain,
                published_at: inferred.published_at,
                topics: inferred.topics,
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

pub fn embed_chunks_file(
    chunks_path: impl AsRef<Path>,
    out_path: impl AsRef<Path>,
    dimensions: usize,
) -> Result<usize> {
    if dimensions == 0 {
        return Err(anyhow!("embedding dimensions must be > 0"));
    }
    let chunks: Vec<Chunk> = storage::read_jsonl(chunks_path)?;
    let metadata = EmbeddingModelMetadata {
        model: LOCAL_HASH_EMBEDDING_MODEL.to_string(),
        version: "2026-05-10".to_string(),
        dimensions,
        method: "deterministic local semantic hashing; L2 normalized".to_string(),
    };
    let embedded: Vec<EmbeddedChunk> = chunks
        .into_iter()
        .map(|chunk| {
            let embedding_text = format!(
                "{} {} {} {} {}",
                chunk.title,
                chunk.source,
                chunk.companies.join(" "),
                chunk.topics.join(" "),
                chunk.text
            );
            EmbeddedChunk {
                embedding: embed_text_with_dimensions(&embedding_text, dimensions),
                chunk,
                embedding_model: metadata.clone(),
            }
        })
        .collect();
    storage::write_jsonl(out_path, &embedded)?;
    Ok(embedded.len())
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
    let mut vectors = Vec::new();

    for (line_no, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("reading line {}", line_no + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let chunk: Chunk = serde_json::from_str(&line)
            .with_context(|| format!("parsing JSONL chunk at line {}", line_no + 1))?;
        let indexed = IndexedChunk::from_chunk(chunk);
        let embedding_text = format!(
            "{} {} {} {} {}",
            indexed.title,
            indexed.source,
            indexed.companies.join(" "),
            indexed.topics.join(" "),
            indexed.text
        );
        vectors.push(VectorRecord {
            chunk_id: indexed.chunk_id.clone(),
            title: indexed.title.clone(),
            url: indexed.url.clone(),
            source: indexed.source.clone(),
            text: indexed.text.clone(),
            companies: indexed.companies.clone(),
            source_type: indexed.source_type.clone(),
            domain: indexed.domain.clone(),
            published_at: indexed.published_at.clone(),
            topics: indexed.topics.clone(),
            embedding: embed_text(&embedding_text),
        });
        writer.add_document(doc!(
            fields.chunk_id => indexed.chunk_id,
            fields.title => indexed.title,
            fields.url => indexed.url,
            fields.source => indexed.source,
            fields.text => indexed.text,
            fields.companies => indexed.companies.join(";"),
            fields.source_type => indexed.source_type.unwrap_or_default(),
            fields.domain => indexed.domain.unwrap_or_default(),
            fields.published_at => indexed.published_at.unwrap_or_default(),
            fields.topics => indexed.topics.join(";"),
        ))?;
        count += 1;
    }

    writer.commit()?;
    writer.wait_merging_threads()?;
    write_vector_store(index_dir, &vectors)?;
    Ok(count)
}

pub fn search_index(
    index_dir: impl AsRef<Path>,
    query_text: &str,
    limit: usize,
) -> Result<Vec<SearchResult>> {
    search_index_with_filters(index_dir, query_text, limit, &SearchFilters::default())
}

pub fn search_index_with_filters(
    index_dir: impl AsRef<Path>,
    query_text: &str,
    limit: usize,
    filters: &SearchFilters,
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
    let query_parser = QueryParser::for_index(
        &index,
        vec![
            fields.title,
            fields.text,
            fields.source,
            fields.companies,
            fields.source_type,
            fields.domain,
            fields.topics,
        ],
    );
    let query = query_parser
        .parse_query(query_text)
        .with_context(|| format!("parsing query {query_text:?}"))?;

    let bm25_limit = limit.saturating_mul(4).max(limit).max(20);
    let top_docs = searcher.search(&query, &TopDocs::with_limit(bm25_limit))?;
    let mut candidates: BTreeMap<String, Candidate> = BTreeMap::new();

    for (score, address) in top_docs {
        let doc: TantivyDocument = searcher.doc(address)?;
        if !filters.matches_doc(&doc, fields) {
            continue;
        }
        let candidate = Candidate {
            chunk_id: stored_text(&doc, fields.chunk_id),
            title: stored_text(&doc, fields.title),
            url: stored_text(&doc, fields.url),
            source: stored_text(&doc, fields.source),
            text: stored_text(&doc, fields.text),
            companies: split_terms(&stored_text(&doc, fields.companies)),
            source_type: optional_stored_text(&doc, fields.source_type),
            domain: optional_stored_text(&doc, fields.domain),
            published_at: optional_stored_text(&doc, fields.published_at),
            topics: split_terms(&stored_text(&doc, fields.topics)),
            bm25: score,
            vector: 0.0,
        };
        candidates.insert(candidate.chunk_id.clone(), candidate);
    }

    for hit in vector_search(index_dir, query_text, bm25_limit, filters)? {
        candidates
            .entry(hit.record.chunk_id.clone())
            .and_modify(|candidate| candidate.vector = candidate.vector.max(hit.score))
            .or_insert_with(|| Candidate {
                chunk_id: hit.record.chunk_id,
                title: hit.record.title,
                url: hit.record.url,
                source: hit.record.source,
                text: hit.record.text,
                companies: hit.record.companies,
                source_type: hit.record.source_type,
                domain: hit.record.domain,
                published_at: hit.record.published_at,
                topics: hit.record.topics,
                bm25: 0.0,
                vector: hit.score,
            });
    }

    let max_bm25 = candidates
        .values()
        .map(|candidate| candidate.bm25)
        .fold(0.0_f32, f32::max);
    let max_vector = candidates
        .values()
        .map(|candidate| candidate.vector)
        .fold(0.0_f32, f32::max);

    let mut results: Vec<SearchResult> = candidates
        .into_values()
        .map(|candidate| {
            let bm25_normalized = normalize(candidate.bm25, max_bm25);
            let vector_normalized = normalize(candidate.vector, max_vector);
            let final_score = (0.65 * bm25_normalized) + (0.35 * vector_normalized);
            SearchResult {
                chunk_id: candidate.chunk_id,
                title: candidate.title,
                url: candidate.url,
                snippet: make_snippet(&candidate.text, query_text, 220),
                score: final_score,
                source: candidate.source,
                companies: candidate.companies,
                source_type: candidate.source_type,
                domain: candidate.domain,
                published_at: candidate.published_at,
                topics: candidate.topics,
                score_components: ScoreComponents {
                    bm25: candidate.bm25,
                    bm25_normalized,
                    vector: candidate.vector,
                    vector_normalized,
                    final_score,
                },
            }
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.chunk_id.cmp(&b.chunk_id))
    });
    results.truncate(limit);
    Ok(results)
}

#[derive(Debug)]
struct Candidate {
    chunk_id: String,
    title: String,
    url: String,
    source: String,
    text: String,
    companies: Vec<String>,
    source_type: Option<String>,
    domain: Option<String>,
    published_at: Option<String>,
    topics: Vec<String>,
    bm25: f32,
    vector: f32,
}

#[derive(Debug)]
struct VectorHit {
    record: VectorRecord,
    score: f32,
}

fn write_vector_store(index_dir: &Path, vectors: &[VectorRecord]) -> Result<()> {
    let path = index_dir.join(VECTOR_STORE_FILE);
    let file = File::create(&path).with_context(|| format!("creating {}", path.display()))?;
    serde_json::to_writer(BufWriter::new(file), vectors)
        .with_context(|| format!("writing {}", path.display()))
}

fn vector_search(
    index_dir: &Path,
    query_text: &str,
    limit: usize,
    filters: &SearchFilters,
) -> Result<Vec<VectorHit>> {
    let path = index_dir.join(VECTOR_STORE_FILE);
    if !path.exists() {
        return Ok(Vec::new());
    }

    let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
    let records: Vec<VectorRecord> = serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("reading {}", path.display()))?;
    let query_embedding = embed_text(query_text);
    let mut hits: Vec<VectorHit> = records
        .into_iter()
        .filter(|record| filters.matches_record(record))
        .filter_map(|record| {
            let score = dot(&query_embedding, &record.embedding);
            (score > 0.0).then_some(VectorHit { record, score })
        })
        .collect();
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.record.chunk_id.cmp(&b.record.chunk_id))
    });
    hits.truncate(limit);
    Ok(hits)
}

fn normalize(score: f32, max_score: f32) -> f32 {
    if max_score <= f32::EPSILON {
        0.0
    } else {
        (score / max_score).clamp(0.0, 1.0)
    }
}

fn embed_text(text: &str) -> Vec<f32> {
    embed_text_with_dimensions(text, EMBEDDING_DIMS)
}

fn embed_text_with_dimensions(text: &str, dimensions: usize) -> Vec<f32> {
    let mut vector = vec![0.0; dimensions];
    let mut seen = HashSet::new();
    for token in semantic_tokens(text) {
        if token.len() <= 2 || !seen.insert(token.clone()) {
            continue;
        }
        let idx = token_hash(&token) % dimensions;
        vector[idx] += 1.0;
    }
    let norm = dot(&vector, &vector).sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

fn semantic_tokens(text: &str) -> Vec<String> {
    let aliases = semantic_aliases();
    tokenize(text)
        .into_iter()
        .flat_map(|token| {
            let mut expanded = vec![token.clone()];
            if let Some(extra) = aliases.get(token.as_str()) {
                expanded.extend(extra.iter().map(|term| (*term).to_string()));
            }
            expanded
        })
        .collect()
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|term| term.len() > 2)
        .map(str::to_lowercase)
        .collect()
}

fn semantic_aliases() -> HashMap<&'static str, Vec<&'static str>> {
    HashMap::from([
        ("accelerator", vec!["gpu", "training", "inference"]),
        ("accelerators", vec!["gpu", "training", "inference"]),
        ("gpu", vec!["accelerator", "training", "inference"]),
        ("gpus", vec!["accelerator", "training", "inference"]),
        ("hbm", vec!["memory", "bandwidth"]),
        ("memory", vec!["hbm", "bandwidth"]),
        ("bandwidth", vec!["hbm", "memory"]),
        ("nvlink", vec!["networking", "interconnect", "fabric"]),
        ("networking", vec!["nvlink", "interconnect", "fabric"]),
        ("interconnect", vec!["nvlink", "networking", "fabric"]),
        ("blackwell", vec!["gb200", "nvidia", "accelerator"]),
        ("gb200", vec!["blackwell", "nvidia", "accelerator"]),
        ("mi300", vec!["amd", "accelerator", "hbm"]),
        ("chiplet", vec!["package", "advanced", "packaging"]),
        ("chiplets", vec!["package", "advanced", "packaging"]),
        ("ai", vec!["training", "inference", "accelerator"]),
        ("training", vec!["ai", "accelerator", "workload"]),
        ("inference", vec!["ai", "accelerator", "workload"]),
        ("economics", vec!["cost", "tco"]),
        ("cost", vec!["economics", "tco"]),
    ])
}

fn token_hash(token: &str) -> usize {
    let digest = Sha256::digest(token.as_bytes());
    usize::from_le_bytes(digest[..std::mem::size_of::<usize>()].try_into().unwrap())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(left, right)| left * right)
        .sum()
}

fn build_schema() -> (Schema, Fields) {
    let mut builder = Schema::builder();
    let chunk_id = builder.add_text_field(FIELD_CHUNK_ID, STORED);
    let title = builder.add_text_field(FIELD_TITLE, TEXT | STORED);
    let url = builder.add_text_field(FIELD_URL, STORED);
    let source = builder.add_text_field(FIELD_SOURCE, TEXT | STORED);
    let text = builder.add_text_field(FIELD_TEXT, TEXT | STORED);
    let companies = builder.add_text_field(FIELD_COMPANIES, TEXT | STORED);
    let source_type = builder.add_text_field(FIELD_SOURCE_TYPE, TEXT | STORED);
    let domain = builder.add_text_field(FIELD_DOMAIN, TEXT | STORED);
    let published_at = builder.add_text_field(FIELD_PUBLISHED_AT, STORED);
    let topics = builder.add_text_field(FIELD_TOPICS, TEXT | STORED);
    let schema = builder.build();
    (
        schema,
        Fields {
            chunk_id,
            title,
            url,
            source,
            text,
            companies,
            source_type,
            domain,
            published_at,
            topics,
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
        companies: schema.get_field(FIELD_COMPANIES)?,
        source_type: schema.get_field(FIELD_SOURCE_TYPE)?,
        domain: schema.get_field(FIELD_DOMAIN)?,
        published_at: schema.get_field(FIELD_PUBLISHED_AT)?,
        topics: schema.get_field(FIELD_TOPICS)?,
    })
}

fn stored_text(doc: &TantivyDocument, field: Field) -> String {
    doc.get_first(field)
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_string()
}

fn optional_stored_text(doc: &TantivyDocument, field: Field) -> Option<String> {
    let value = stored_text(doc, field);
    (!value.is_empty()).then_some(value)
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

#[derive(Debug, Clone, Default)]
struct InferredMetadata {
    companies: Vec<String>,
    source_type: Option<String>,
    domain: Option<String>,
    published_at: Option<String>,
    topics: Vec<String>,
}

#[derive(Debug, Clone)]
struct IndexedChunk {
    chunk_id: String,
    title: String,
    url: String,
    source: String,
    text: String,
    companies: Vec<String>,
    source_type: Option<String>,
    domain: Option<String>,
    published_at: Option<String>,
    topics: Vec<String>,
}

impl IndexedChunk {
    fn from_chunk(chunk: Chunk) -> Self {
        let inferred = infer_metadata(&chunk.title, &chunk.url, &chunk.source, &chunk.text);
        let companies = metadata_strings(&chunk.metadata, "companies")
            .or_else(|| non_empty_vec(chunk.companies.clone()))
            .unwrap_or(inferred.companies);
        let source_type = metadata_string(&chunk.metadata, "source_type")
            .or(chunk.source_type)
            .or(inferred.source_type);
        let domain = metadata_string(&chunk.metadata, "domain")
            .or(chunk.domain)
            .or(inferred.domain);
        let published_at = metadata_string(&chunk.metadata, "published_at")
            .or(chunk.published_at)
            .or(inferred.published_at);
        let topics = metadata_strings(&chunk.metadata, "topics")
            .or_else(|| non_empty_vec(chunk.topics.clone()))
            .unwrap_or(inferred.topics);

        Self {
            chunk_id: chunk.chunk_id,
            title: chunk.title,
            url: chunk.url,
            source: chunk.source,
            text: chunk.text,
            companies,
            source_type,
            domain,
            published_at,
            topics,
        }
    }
}

impl SearchFilters {
    fn matches_doc(&self, doc: &TantivyDocument, fields: Fields) -> bool {
        if let Some(company) = &self.company {
            if !contains_term(&stored_text(doc, fields.companies), company) {
                return false;
            }
        }
        if let Some(source_type) = &self.source_type {
            if !eq_ci(&stored_text(doc, fields.source_type), source_type) {
                return false;
            }
        }
        if let Some(domain) = &self.domain {
            if !eq_ci(&stored_text(doc, fields.domain), domain) {
                return false;
            }
        }
        if let Some(after) = &self.after {
            let published_at = stored_text(doc, fields.published_at);
            if published_at.is_empty() || published_at.as_str() < after.as_str() {
                return false;
            }
        }
        if let Some(topic) = &self.topic {
            if !contains_term(&stored_text(doc, fields.topics), topic) {
                return false;
            }
        }
        true
    }

    fn matches_record(&self, record: &VectorRecord) -> bool {
        if let Some(company) = &self.company {
            if !record.companies.iter().any(|value| eq_ci(value, company)) {
                return false;
            }
        }
        if let Some(source_type) = &self.source_type {
            if !record
                .source_type
                .as_deref()
                .is_some_and(|value| eq_ci(value, source_type))
            {
                return false;
            }
        }
        if let Some(domain) = &self.domain {
            if !record
                .domain
                .as_deref()
                .is_some_and(|value| eq_ci(value, domain))
            {
                return false;
            }
        }
        if let Some(after) = &self.after {
            if !record
                .published_at
                .as_deref()
                .is_some_and(|value| value >= after.as_str())
            {
                return false;
            }
        }
        if let Some(topic) = &self.topic {
            if !record.topics.iter().any(|value| {
                eq_ci(value, topic) || value.to_lowercase().contains(&topic.to_lowercase())
            }) {
                return false;
            }
        }
        true
    }
}

fn infer_metadata(title: &str, url: &str, source: &str, text: &str) -> InferredMetadata {
    let haystack = format!("{title} {url} {source} {text}");
    InferredMetadata {
        companies: infer_companies(&haystack),
        source_type: Some(infer_source_type(url, source, title)),
        domain: infer_domain(url),
        published_at: infer_published_at(&haystack),
        topics: infer_topics(&haystack),
    }
}

fn infer_companies(haystack: &str) -> Vec<String> {
    let candidates = [
        (
            "NVIDIA",
            ["nvidia", "nvda", "blackwell", "gb200"].as_slice(),
        ),
        ("AMD", ["amd", "mi300", "instinct"].as_slice()),
        ("TSMC", ["tsmc", "cowos"].as_slice()),
        ("Intel", ["intel", "gaudi"].as_slice()),
        ("Broadcom", ["broadcom", "avgo"].as_slice()),
        ("Micron", ["micron", "mu"].as_slice()),
        ("SK Hynix", ["sk hynix", "hynix"].as_slice()),
        ("Samsung", ["samsung"].as_slice()),
        ("ASML", ["asml", "euv"].as_slice()),
    ];
    let lower = haystack.to_lowercase();
    candidates
        .iter()
        .filter(|(_, aliases)| {
            aliases
                .iter()
                .any(|alias| lower.contains(&alias.to_lowercase()))
        })
        .map(|(name, _)| (*name).to_string())
        .collect()
}

fn infer_source_type(url: &str, source: &str, title: &str) -> String {
    let lower = format!("{url} {source} {title}").to_lowercase();
    if lower.contains("sec.gov")
        || lower.contains("10-k")
        || lower.contains("10-q")
        || lower.contains("filing")
    {
        "filing".to_string()
    } else if lower.contains("earnings") || lower.contains("transcript") {
        "earnings".to_string()
    } else if lower.contains("substack") {
        "substack".to_string()
    } else if lower.contains("news") || lower.contains("press") {
        "news".to_string()
    } else if lower.contains("research")
        || lower.contains("note")
        || lower.contains("architecture")
        || lower.contains("analysis")
    {
        "analysis".to_string()
    } else if lower.contains("fixture") {
        "fixture".to_string()
    } else {
        "document".to_string()
    }
}

fn infer_domain(url: &str) -> Option<String> {
    Url::parse(url)
        .ok()
        .and_then(|parsed| parsed.host_str().map(str::to_string))
}

fn infer_published_at(haystack: &str) -> Option<String> {
    let bytes = haystack.as_bytes();
    for window in bytes.windows(10) {
        if window[4] == b'-'
            && window[7] == b'-'
            && window[..4].iter().all(u8::is_ascii_digit)
            && window[5..7].iter().all(u8::is_ascii_digit)
            && window[8..10].iter().all(u8::is_ascii_digit)
        {
            return String::from_utf8(window.to_vec()).ok();
        }
    }
    None
}

fn infer_topics(haystack: &str) -> Vec<String> {
    let lower = haystack.to_lowercase();
    let candidates = [
        (
            "ai-accelerators",
            [
                "accelerator",
                "gpu",
                "training",
                "inference",
                "blackwell",
                "mi300",
            ]
            .as_slice(),
        ),
        (
            "advanced-packaging",
            ["cowos", "advanced packaging", "chiplet", "package"].as_slice(),
        ),
        ("memory", ["hbm", "dram", "nand", "memory"].as_slice()),
        (
            "networking",
            [
                "nvlink",
                "networking",
                "ethernet",
                "infiniband",
                "interconnect",
            ]
            .as_slice(),
        ),
        (
            "semicap",
            ["euv", "lithography", "wafer", "asml"].as_slice(),
        ),
        (
            "pricing",
            ["pricing", "margin", "cost", "economics"].as_slice(),
        ),
    ];
    candidates
        .iter()
        .filter(|(_, needles)| needles.iter().any(|needle| lower.contains(needle)))
        .map(|(topic, _)| (*topic).to_string())
        .collect()
}

fn metadata_string(metadata: &serde_json::Value, key: &str) -> Option<String> {
    metadata
        .get(key)
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn metadata_strings(metadata: &serde_json::Value, key: &str) -> Option<Vec<String>> {
    let values = metadata.get(key)?;
    if let Some(array) = values.as_array() {
        return non_empty_vec(
            array
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::to_string)
                .collect(),
        );
    }
    values
        .as_str()
        .and_then(|value| non_empty_vec(split_terms(value)))
}

fn non_empty_vec(values: Vec<String>) -> Option<Vec<String>> {
    let unique: Vec<String> = values
        .into_iter()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    (!unique.is_empty()).then_some(unique)
}

fn split_terms(value: &str) -> Vec<String> {
    value
        .split(|c: char| c == ',' || c == ';')
        .map(str::trim)
        .filter(|term| !term.is_empty())
        .map(str::to_string)
        .collect()
}

fn contains_term(haystack: &str, needle: &str) -> bool {
    split_terms(haystack).iter().any(|term| eq_ci(term, needle))
        || haystack.to_lowercase().contains(&needle.to_lowercase())
}

fn eq_ci(left: &str, right: &str) -> bool {
    left.eq_ignore_ascii_case(right)
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
