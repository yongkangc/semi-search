use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;
use std::fs;
use tempfile::TempDir;

#[test]
fn crawls_fixture_chunks_then_indexes_and_searches() {
    let temp = TempDir::new().expect("temp dir");
    let fixture = temp.path().join("blackwell.html");
    fs::write(
        &fixture,
        "<html><head><title>NVIDIA Blackwell</title></head><body><main>Blackwell accelerators use NVLink networking, HBM memory, and improve AI training economics.</main></body></html>",
    )
    .expect("fixture");
    let chunks = temp.path().join("chunks.jsonl");
    let config = temp.path().join("seeds.toml");
    fs::write(
        &config,
        format!(
            r#"output_jsonl = "{}"
chunk_tokens = 20
chunk_overlap = 4

[[seeds]]
url = "https://example.com/nvidia-blackwell"
source = "fixture"
fixture_path = "{}"
"#,
            chunks.display(),
            fixture.display()
        ),
    )
    .expect("config");

    Command::cargo_bin("semi-search")
        .expect("binary")
        .args(["crawl", "--config"])
        .arg(&config)
        .assert()
        .success()
        .stdout(predicate::str::contains("wrote_chunks="));

    let index_dir = temp.path().join("index");
    Command::cargo_bin("semi-search")
        .expect("binary")
        .args(["index", "--chunks"])
        .arg(&chunks)
        .args(["--index"])
        .arg(&index_dir)
        .assert()
        .success()
        .stdout(predicate::str::contains("indexed_chunks="));

    let output = Command::cargo_bin("semi-search")
        .expect("binary")
        .args(["search", "--index"])
        .arg(&index_dir)
        .args(["--query", "Blackwell AI training economics", "--limit", "3"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let results: Value = serde_json::from_slice(&output).expect("search emits JSON");
    let rows = results.as_array().expect("array results");
    assert!(!rows.is_empty(), "expected search hit");
    let serialized = serde_json::to_string(&results).expect("serialize");
    assert!(serialized.contains("NVIDIA Blackwell"));
    assert!(serialized.contains("https://example.com/nvidia-blackwell"));
}
