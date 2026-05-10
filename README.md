# semi-search

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/a24f0127-f4b3-47f4-a82a-00fdcbd4f9af" />


Semi Search: agent-first search for semiconductor research.

The goal is to build an agent-first search engine, in Rust, optimized for coding agents, research agents, and semiconductor fundamental investing workflows. Instead of generic web search, this project focuses on retrieving high-signal semiconductor context: company filings, product architecture notes, technical blogs, industry analysis, news, and supply-chain commentary.

Initial target corpus:

- NVIDIA
- AMD
- TSMC
- SemiAnalysis
- SemiVision Substack
- Irrational Analysis Substack
- Semiconductor news, blogs, architecture notes, and investment research sources

## Product goal

Given a question like:

> What are the key differences between NVIDIA Blackwell and AMD MI300 for AI training economics?

The system should retrieve source-backed context with citations, including:

- relevant architecture details
- company/product references
- industry analysis
- news and blog commentary
- freshness metadata
- snippets that can be passed directly into a coding or research agent

This is not just a vector database. It is a full retrieval system:

1. crawl and ingest sources
2. clean, parse, deduplicate, and chunk documents
3. build keyword + vector indexes
4. retrieve and rerank relevant context
5. expose agent-friendly search APIs
6. evaluate search quality continuously

## Prototype scope

The first prototype should be deliberately small and useful:

- crawl or ingest a small curated set of semiconductor URLs, with autodiscovery from sitemaps, RSS feeds, and internal links
- parse HTML and Markdown documents
- chunk documents into source-backed passages
- index chunks with keyword search and vector embeddings
- expose a simple local search API
- return results with title, URL, snippet, score, and source metadata
- include a small evaluation set of semiconductor investing queries

## Proposed architecture

```text
Semiconductor sources
        │
        ▼
Crawler / connectors
        │
        ▼
Parser / cleaner / deduper
        │
        ▼
Chunker + metadata extractor
        │
        ▼
Embeddings + keyword terms
        │
        ▼
Vector index + BM25 index
        │
        ▼
Hybrid retrieval API
        │
        ▼
Reranker / source-quality scorer
        │
        ▼
Agent context builder
```

## Core systems

- Crawler and source connectors, including sitemap/RSS/link autodiscovery
- Document processing pipeline
- Chunking and metadata extraction
- Storage for raw docs, cleaned docs, chunks, and crawl state
- Vector index
- Keyword/BM25 index
- Hybrid retrieval and reranking
- Search/query API
- Agent integration layer
- Evaluation harness
- Feedback and observability loop
- Trust, provenance, and citation layer

## Rust-first direction

The project should be Rust-first for performance, reliability, and eventual scale.

Likely components:

- crawler workers
- document parser pipeline
- chunk/index builder
- search API server
- evaluation CLI

Candidate ecosystem:

- `tokio` for async runtime
- `reqwest` for fetching
- `scraper` or `html5ever` for HTML parsing
- `tantivy` for BM25/full-text search
- Qdrant, LanceDB, or an embedded vector index for dense retrieval
- `axum` for the API server
- `serde` for data models

## Non-goals for v0

- crawling the entire web
- distributed infrastructure
- perfect ranking
- production-grade freshness scheduling
- paid/private source ingestion unless explicitly configured

The first win is simple: useful semiconductor search over a curated corpus, with citations.
