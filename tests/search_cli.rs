use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;
use tempfile::TempDir;

#[test]
fn indexes_chunks_then_search_returns_cited_semiconductor_results() {
    let temp = TempDir::new().expect("temp dir");
    let index_dir = temp.path().join("index");
    let chunks = "tests/fixtures/semiconductor_chunks.jsonl";

    Command::cargo_bin("semi-search")
        .expect("binary")
        .args(["index", "--chunks", chunks, "--index"])
        .arg(&index_dir)
        .assert()
        .success()
        .stdout(predicate::str::contains("indexed_chunks=4"));

    let output = Command::cargo_bin("semi-search")
        .expect("binary")
        .args([
            "search",
            "--index",
            index_dir.to_str().expect("utf8 path"),
            "--query",
            "Blackwell MI300 AI training economics",
            "--limit",
            "3",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let results: Value = serde_json::from_slice(&output).expect("search emits JSON results");
    let rows = results.as_array().expect("array results");
    assert!(!rows.is_empty(), "expected at least one search hit");

    let first = rows.first().expect("first result");
    for field in ["title", "url", "snippet", "score", "source"] {
        assert!(
            first.get(field).is_some(),
            "missing field {field}: {first:?}"
        );
    }

    let serialized = serde_json::to_string(&results).expect("serialize results");
    assert!(
        serialized.contains("Blackwell") || serialized.contains("MI300"),
        "expected semiconductor query terms in cited results: {serialized}"
    );
    assert!(
        serialized.contains("https://example.com/"),
        "expected source citation URL in results: {serialized}"
    );
}

#[test]
fn hybrid_search_uses_vector_semantics_when_keywords_do_not_match() {
    let temp = TempDir::new().expect("temp dir");
    let index_dir = temp.path().join("index");
    let chunks = "tests/fixtures/semiconductor_chunks.jsonl";

    Command::cargo_bin("semi-search")
        .expect("binary")
        .args(["index", "--chunks", chunks, "--index"])
        .arg(&index_dir)
        .assert()
        .success();

    let output = Command::cargo_bin("semi-search")
        .expect("binary")
        .args([
            "search",
            "--index",
            index_dir.to_str().expect("utf8 path"),
            "--query",
            "GB200 interconnect bandwidth inference",
            "--limit",
            "2",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let results: Value = serde_json::from_slice(&output).expect("search emits JSON results");
    let rows = results.as_array().expect("array results");
    assert!(!rows.is_empty(), "expected vector-backed search hit");

    let serialized = serde_json::to_string(&results).expect("serialize results");
    assert!(
        serialized.contains("NVIDIA Blackwell Architecture Notes"),
        "semantic aliases should retrieve Blackwell/NVLink/HBM even when query says GB200/interconnect/bandwidth: {serialized}"
    );

    let first = rows.first().expect("first result");
    let components = first
        .get("score_components")
        .expect("transparent score components");
    assert!(
        components
            .get("vector")
            .and_then(Value::as_f64)
            .expect("vector score")
            > 0.0,
        "expected vector score component: {components:?}"
    );
}
