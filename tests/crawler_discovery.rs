use std::fs;
use tempfile::TempDir;

#[test]
fn autodiscovers_same_domain_links_from_local_fixtures() {
    let temp = TempDir::new().expect("temp dir");
    let root = temp.path().join("root.html");
    let child = temp.path().join("child.html");
    let chunks = temp.path().join("chunks.jsonl");
    let config = temp.path().join("seeds.toml");

    fs::write(
        &root,
        r#"<html><head><title>Root</title></head><body><main>
        NVIDIA Blackwell root page links to an architecture note.
        <a href="/research/blackwell-architecture">architecture</a>
        <a href="https://other.example.com/research/outside">outside domain</a>
        </main></body></html>"#,
    )
    .expect("root fixture");
    fs::write(
        &child,
        r#"<html><head><title>Child</title></head><body><main>
        Child page covers NVLink, HBM, and AI training economics.
        </main></body></html>"#,
    )
    .expect("child fixture");

    fs::write(
        &config,
        format!(
            r#"output_jsonl = "{}"
chunk_tokens = 40
chunk_overlap = 5
max_pages = 3
discover_same_domain = true

[[fixture_responses]]
url = "https://example.com/root"
path = "{}"

[[fixture_responses]]
url = "https://example.com/research/blackwell-architecture"
path = "{}"

[[seeds]]
url = "https://example.com/root"
source = "fixture"
allow_paths = ["/root", "/research/"]
"#,
            chunks.display(),
            root.display(),
            child.display()
        ),
    )
    .expect("config");

    let config = semi_search::load_crawl_config(&config).expect("load config");
    let records = semi_search::crawl_to_chunks(&config).expect("crawl");
    let urls: Vec<_> = records.iter().map(|record| record.url.as_str()).collect();

    assert!(urls.contains(&"https://example.com/root"));
    assert!(urls.contains(&"https://example.com/research/blackwell-architecture"));
    assert!(!urls.iter().any(|url| url.contains("other.example.com")));
}

#[test]
fn applies_allow_and_deny_path_patterns_before_fetching_discovered_pages() {
    let temp = TempDir::new().expect("temp dir");
    let root = temp.path().join("root.html");
    let allowed = temp.path().join("allowed.html");
    let denied = temp.path().join("denied.html");
    let chunks = temp.path().join("chunks.jsonl");
    let config = temp.path().join("seeds.toml");

    fs::write(
        &root,
        r#"<html><head><title>Root</title></head><body><main>
        TSMC technology root.
        <a href="/research/cowos-capacity">CoWoS capacity</a>
        <a href="/research/archive-old-node">old node archive</a>
        <a href="/marketing/events">events</a>
        </main></body></html>"#,
    )
    .expect("root fixture");
    fs::write(
        &allowed,
        "<html><body><main>CoWoS advanced packaging capacity and N3 demand.</main></body></html>",
    )
    .expect("allowed fixture");
    fs::write(
        &denied,
        "<html><body><main>This denied archive fixture must not appear.</main></body></html>",
    )
    .expect("denied fixture");

    fs::write(
        &config,
        format!(
            r#"output_jsonl = "{}"
chunk_tokens = 40
chunk_overlap = 5
max_pages = 5
discover_same_domain = true

[[fixture_responses]]
url = "https://example.com/root"
path = "{}"

[[fixture_responses]]
url = "https://example.com/research/cowos-capacity"
path = "{}"

[[fixture_responses]]
url = "https://example.com/research/archive-old-node"
path = "{}"

[[seeds]]
url = "https://example.com/root"
source = "fixture"
allow_paths = ["/root", "/research/"]
deny_paths = ["archive"]
"#,
            chunks.display(),
            root.display(),
            allowed.display(),
            denied.display()
        ),
    )
    .expect("config");

    let config = semi_search::load_crawl_config(&config).expect("load config");
    let records = semi_search::crawl_to_chunks(&config).expect("crawl");
    let body = records
        .iter()
        .map(|record| format!("{}\n{}", record.url, record.text))
        .collect::<Vec<_>>()
        .join("\n");

    assert!(body.contains("/research/cowos-capacity"));
    assert!(body.contains("CoWoS advanced packaging"));
    assert!(!body.contains("archive-old-node"));
    assert!(!body.contains("denied archive fixture"));
    assert!(!body.contains("/marketing/events"));
}
