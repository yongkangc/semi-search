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
    /// Crawl seed sources into raw documents.
    Crawl(PathCommand),
    /// Parse raw documents into cleaned documents.
    Parse(PathCommand),
    /// Chunk cleaned documents into source-backed passages.
    Chunk(PathCommand),
    /// Build a local BM25/Tantivy index from chunk JSONL.
    Index(IndexCommand),
    /// Search a local BM25/Tantivy index and emit cited JSON results.
    Search(SearchCommand),
    /// Run retrieval quality evaluations.
    Eval(PathCommand),
}

#[derive(Debug, Args)]
struct PathCommand {
    /// Local data directory used by this pipeline stage.
    #[arg(long, default_value = "data")]
    data_dir: PathBuf,
}

#[derive(Debug, Args)]
struct IndexCommand {
    /// JSONL file. Each line needs chunk_id, title, url, source, text.
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
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Crawl(args) => stub_stage("crawl", args.data_dir),
        Command::Parse(args) => stub_stage("parse", args.data_dir),
        Command::Chunk(args) => stub_stage("chunk", args.data_dir),
        Command::Index(args) => {
            let count = semi_search::index_chunks(args.chunks, args.index)?;
            println!("indexed_chunks={count}");
            Ok(())
        }
        Command::Eval(args) => stub_stage("eval", args.data_dir),
        Command::Search(args) => {
            let results = semi_search::search_index(args.index, &args.query, args.limit)?;
            println!("{}", serde_json::to_string_pretty(&results)?);
            Ok(())
        }
    }
}

fn stub_stage(stage: &str, data_dir: PathBuf) -> Result<()> {
    println!("{stage}: stub; data_dir={}", data_dir.display());
    Ok(())
}
