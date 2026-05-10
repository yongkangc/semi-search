use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "semi-search")]
#[command(about = "Agent-first semiconductor research search", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Quick crawl seed sources into chunk JSONL.
    Crawl(CrawlCommand),
    /// Parse raw documents into cleaned documents. Reserved for later pipeline split.
    Parse(PathCommand),
    /// Chunk cleaned documents into source-backed passages. Reserved for later pipeline split.
    Chunk(PathCommand),
    /// Generate deterministic local embeddings for chunk JSONL.
    Embed(EmbedCommand),
    /// Build a local BM25/Tantivy + local vector index from chunk JSONL.
    Index(IndexCommand),
    /// Search a local hybrid BM25/vector index and emit cited JSON results.
    Search(SearchCommand),
    /// Run retrieval quality evaluations. Reserved for golden-query harness.
    Eval(PathCommand),
}

#[derive(Debug, Args)]
struct CrawlCommand {
    /// TOML seed config to run.
    #[arg(short, long, default_value = "configs/seeds.example.toml")]
    config: PathBuf,
}

#[derive(Debug, Args)]
struct PathCommand {
    /// Local data directory used by this pipeline stage.
    #[arg(long, default_value = "data")]
    data_dir: PathBuf,
}

#[derive(Debug, Args)]
struct EmbedCommand {
    /// Input chunk JSONL. Each line needs id/chunk_id, title, url, source, text.
    #[arg(long)]
    chunks: PathBuf,
    /// Output embedded chunk JSONL.
    #[arg(long)]
    out: PathBuf,
    /// Embedding vector dimensions for the local deterministic model.
    #[arg(long, default_value_t = semi_search::EMBEDDING_DIMS)]
    dimensions: usize,
}

#[derive(Debug, Args)]
struct IndexCommand {
    /// JSONL file. Each line needs id/chunk_id, title, url, source, text.
    #[arg(long)]
    chunks: PathBuf,
    /// Local index directory to create. Existing contents are replaced.
    #[arg(long, default_value = "data/index")]
    index: PathBuf,
}

#[derive(Debug, Args)]
struct SearchCommand {
    /// Local index directory created by `semi-search index`.
    #[arg(long, default_value = "data/index")]
    index: PathBuf,
    /// Query text.
    #[arg(long)]
    query: String,
    /// Maximum number of results to return.
    #[arg(long, default_value_t = 10)]
    limit: usize,
    /// Filter by inferred or provided company tag, e.g. NVIDIA, AMD, TSMC.
    #[arg(long)]
    company: Option<String>,
    /// Filter by source type, e.g. analysis, substack, news, filing.
    #[arg(long = "source-type")]
    source_type: Option<String>,
    /// Filter by URL domain, e.g. semianalysis.com.
    #[arg(long)]
    domain: Option<String>,
    /// Filter to documents published at or after YYYY-MM-DD.
    #[arg(long)]
    after: Option<String>,
    /// Filter by inferred or provided topic tag.
    #[arg(long)]
    topic: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Crawl(args) => {
            let config = semi_search::load_crawl_config(args.config)?;
            let chunks = semi_search::crawl_to_chunks(&config)?;
            println!(
                "wrote_chunks={} output={}",
                chunks.len(),
                config.output_jsonl.display()
            );
            Ok(())
        }
        Command::Parse(args) => stub_stage("parse", args.data_dir),
        Command::Chunk(args) => stub_stage("chunk", args.data_dir),
        Command::Embed(args) => {
            let count = semi_search::embed_chunks_file(args.chunks, args.out, args.dimensions)?;
            println!(
                "embedded_chunks={count} model={} dimensions={}",
                semi_search::LOCAL_HASH_EMBEDDING_MODEL,
                args.dimensions
            );
            Ok(())
        }
        Command::Index(args) => {
            let count = semi_search::index_chunks(args.chunks, args.index)?;
            println!("indexed_chunks={count}");
            Ok(())
        }
        Command::Eval(args) => stub_stage("eval", args.data_dir),
        Command::Search(args) => {
            let filters = semi_search::SearchFilters {
                company: args.company,
                source_type: args.source_type,
                domain: args.domain,
                after: args.after,
                topic: args.topic,
            };
            let results = semi_search::search_index_with_filters(
                args.index,
                &args.query,
                args.limit,
                &filters,
            )?;
            println!("{}", serde_json::to_string_pretty(&results)?);
            Ok(())
        }
    }
}

fn stub_stage(stage: &str, data_dir: PathBuf) -> Result<()> {
    println!("{stage}: stub; data_dir={}", data_dir.display());
    Ok(())
}
