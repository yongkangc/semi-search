use assert_cmd::Command;
use serde_json::Value;
use tempfile::TempDir;

fn build_fixture_index() -> TempDir {
    let temp = TempDir::new().expect("temp dir");
    let index_dir = temp.path().join("index");
    Command::cargo_bin("semi-search")
        .expect("binary")
        .args([
            "index",
            "--chunks",
            "tests/fixtures/semiconductor_chunks.jsonl",
            "--index",
        ])
        .arg(&index_dir)
        .assert()
        .success();
    temp
}

#[test]
fn eval_reports_recall_like_filter_matches_and_top_titles() {
    let temp = build_fixture_index();
    let output = Command::cargo_bin("semi-search")
        .expect("binary")
        .args([
            "eval",
            "--queries",
            "tests/fixtures/eval_queries.jsonl",
            "--index",
            temp.path().join("index").to_str().expect("utf8 path"),
            "--limit",
            "3",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let report: Value = serde_json::from_slice(&output).expect("eval emits JSON report");
    assert_eq!(report["query_count"], 2);
    assert_eq!(report["passed"], 2);
    assert_eq!(report["failed"], 0);
    assert_eq!(report["recall_like_filter_match"], 1.0);
    assert!(report["results"][0]["top_titles"].as_array().unwrap().len() > 0);
}

#[test]
fn compare_prints_semi_search_and_placeholder_baseline_json() {
    let temp = build_fixture_index();
    let output = Command::cargo_bin("semi-search")
        .expect("binary")
        .args([
            "compare",
            "--index",
            temp.path().join("index").to_str().expect("utf8 path"),
            "--query",
            "Blackwell MI300 AI training economics",
            "--limit",
            "2",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let report: Value = serde_json::from_slice(&output).expect("compare emits JSON");
    assert_eq!(report["baseline"]["kind"], "web-baseline-placeholder");
    assert_eq!(report["baseline"]["result_count"], 0);
    assert!(report["semi_search"]["result_count"].as_u64().unwrap() > 0);
    assert!(report["semi_search"]["results"].as_array().unwrap()[0]
        .get("title")
        .is_some());
}
