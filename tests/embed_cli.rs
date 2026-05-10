use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;
use tempfile::TempDir;

#[test]
fn embed_command_writes_deterministic_embedded_jsonl() {
    let temp = TempDir::new().expect("temp dir");
    let out = temp.path().join("embedded.jsonl");

    Command::cargo_bin("semi-search")
        .expect("binary")
        .args([
            "embed",
            "--chunks",
            "tests/fixtures/semiconductor_chunks.jsonl",
            "--out",
        ])
        .arg(&out)
        .args(["--dimensions", "32"])
        .assert()
        .success()
        .stdout(predicate::str::contains("embedded_chunks=4"))
        .stdout(predicate::str::contains("model=local-hash-bow-v1"));

    let first = std::fs::read_to_string(&out)
        .expect("embedded output")
        .lines()
        .next()
        .expect("first jsonl row")
        .to_string();
    let row: Value = serde_json::from_str(&first).expect("valid json");
    assert_eq!(row["chunk_id"], "nvda-blackwell-001");
    assert_eq!(row["embedding"].as_array().expect("embedding").len(), 32);
    assert_eq!(row["embedding_model"]["model"], "local-hash-bow-v1");
    assert_eq!(row["embedding_model"]["dimensions"], 32);

    let out2 = temp.path().join("embedded-again.jsonl");
    Command::cargo_bin("semi-search")
        .expect("binary")
        .args([
            "embed",
            "--chunks",
            "tests/fixtures/semiconductor_chunks.jsonl",
            "--out",
        ])
        .arg(&out2)
        .args(["--dimensions", "32"])
        .assert()
        .success();
    assert_eq!(
        std::fs::read_to_string(&out).expect("first run"),
        std::fs::read_to_string(&out2).expect("second run"),
        "embedding output should be deterministic"
    );
}
