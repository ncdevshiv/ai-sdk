//! Wire-level edge-case battery for the discovery engine.
//!
//! Live gateways cannot produce controlled anomalies on demand (they cannot
//! return `choices: []` for one request and a perfect completion for the
//! next, on the same model). This harness runs a local HTTP server that
//! serves crafted responses, so every shape the SDK claims to handle is
//! actually exercised — including the ones no live provider will produce.
//!
//! Every test is against the real `Transport` + `DiscoveryEngine`; nothing is
//! mocked at the HTTP-client level.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use ai_discovery::probe::TransportPolicy;
use ai_discovery::{DiscoveryConfig, DiscoveryEngine, ModelRole};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

/// A canned HTTP response.
#[derive(Clone, Debug)]
struct HttpResp {
    status: u16,
    body: String,
    headers: Vec<(String, String)>,
}

impl HttpResp {
    fn new(status: u16, body: impl Into<String>) -> Self {
        Self {
            status,
            body: body.into(),
            headers: vec![("content-type".to_string(), "application/json".to_string())],
        }
    }

    fn txt(status: u16, body: impl Into<String>) -> Self {
        let mut r = Self::new(status, body);
        r.headers = vec![("content-type".to_string(), "text/plain".to_string())];
        r
    }
}

/// Handler: receives the request body, returns the response.
type Handler = Arc<dyn Fn(&str) -> HttpResp + Send + Sync>;

struct MockState {
    /// Exact path -> handler; "*" is the fallback.
    routes: Mutex<HashMap<String, Handler>>,
    /// Every request line, in arrival order, for assertions.
    log: Mutex<Vec<String>>,
}

#[derive(Clone)]
struct Mock {
    addr: SocketAddr,
    state: Arc<MockState>,
    task: Arc<tokio::task::JoinHandle<()>>,
}

impl Mock {
    async fn start(routes: Vec<(&str, Handler)>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().unwrap();
        let state = Arc::new(MockState {
            routes: Mutex::new(
                routes
                    .into_iter()
                    .map(|(p, h)| (p.to_string(), h))
                    .collect(),
            ),
            log: Mutex::new(Vec::new()),
        });
        let st = state.clone();
        let task = tokio::spawn(async move {
            loop {
                let (mut sock, _) = match listener.accept().await {
                    Ok(v) => v,
                    Err(_) => break,
                };
                let st = st.clone();
                tokio::spawn(async move {
                    let _ = serve_conn(&mut sock, st).await;
                });
            }
        });
        Self {
            addr,
            state,
            task: Arc::new(task),
        }
    }

    fn base_url(&self) -> String {
        format!("http://{}/v1", self.addr)
    }

    /// Number of requests that hit a given path (pooled for assertions).
    async fn hits(&self, path: &str) -> usize {
        self.state
            .log
            .lock()
            .await
            .iter()
            .filter(|l| l.as_str() == path)
            .count()
    }
}

impl Drop for Mock {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn serve_conn(
    sock: &mut tokio::net::TcpStream,
    state: Arc<MockState>,
) -> std::io::Result<()> {
    let mut buf = Vec::with_capacity(64 * 1024);
    // Read until end of headers.
    loop {
        let mut chunk = [0u8; 4096];
        let n = sock.read(&mut chunk).await?;
        if n == 0 {
            return Ok(());
        }
        buf.extend_from_slice(&chunk[..n]);
        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
    }
    let head_end = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|p| p + 4)
        .unwrap_or(buf.len());
    let headers_text = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let mut lines = headers_text.split("\r\n");
    let request_line = lines.next().unwrap_or("").to_string();
    let path = request_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .to_string();

    let mut content_length = 0usize;
    for l in lines {
        let ll = l.to_ascii_lowercase();
        if let Some(v) = ll.strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    while buf.len() < head_end + content_length {
        let mut chunk = [0u8; 4096];
        let n = sock.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let body = String::from_utf8_lossy(&buf[head_end..head_end + content_length]).to_string();

    state.log.lock().await.push(path.clone());
    let handler = {
        let routes = state.routes.lock().await;
        routes.get(&path).or_else(|| routes.get("*")).cloned()
    };
    let resp = match handler {
        Some(h) => h(&body),
        None => HttpResp::txt(404, "404 page not found"),
    };

    let mut out = format!(
        "HTTP/1.1 {} {}\r\n",
        resp.status,
        match resp.status {
            200 => "OK",
            400 => "Bad Request",
            401 => "Unauthorized",
            404 => "Not Found",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            _ => "Status",
        }
    );
    for (k, v) in &resp.headers {
        out.push_str(&format!("{k}: {v}\r\n"));
    }
    // This server handles exactly one request per connection and then drops
    // the socket. Advertising that with `Connection: close` stops reqwest
    // from pooling the dead socket and reusing it for the next request,
    // which made these tests fail intermittently under parallel execution.
    out.push_str("Connection: close\r\n");
    out.push_str(&format!("Content-Length: {}\r\n\r\n", resp.body.len()));
    out.push_str(&resp.body);
    sock.write_all(out.as_bytes()).await?;
    sock.flush().await?;
    let _ = sock.shutdown().await;
    Ok(())
}

fn ok_chat(content: impl Into<String>) -> HttpResp {
    HttpResp::new(
        200,
        json!({
            "id": "cmpl-1", "object": "chat.completion", "created": 1, "model": "mock",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content.into()},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
        })
        .to_string(),
    )
}

fn chat_proxy<F: Fn(&Value) -> HttpResp + Send + Sync + 'static>(f: F) -> Handler {
    Arc::new(move |body| f(&serde_json::from_str(body).unwrap_or(Value::Null)))
}

fn engine_for(mock: &Mock, timeout_ms: u64, policy: TransportPolicy) -> DiscoveryEngine {
    DiscoveryEngine::with_policy(
        "mock",
        mock.base_url(),
        "test-key",
        Duration::from_millis(timeout_ms),
        policy,
    )
    .unwrap()
}

async fn discover_one(
    mock: &Mock,
    id: &str,
    cfg: &DiscoveryConfig,
) -> ai_discovery::DiscoveredModel {
    let e = engine_for(mock, 10_000, TransportPolicy::none());
    e.discover_one(id, None, true, cfg).await
}

fn cfg() -> DiscoveryConfig {
    DiscoveryConfig {
        timeout: Duration::from_secs(10),
        transport_policy: TransportPolicy::none(),
        probe_vision: true,
        probe_tools: true,
        probe_structured_output: true,
        probe_endpoints: true,
        probe_thinking: true,
        probe_context: false,
        max_context_probe: 128_000,
        context_rounds: 6,
        limit: 0,
        extra_models: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// /v1/models envelope shapes
// ---------------------------------------------------------------------------

/// A failed listing must surface as an error: a wrong API key and an empty
/// catalog are different conditions, and both must be distinguishable.
#[tokio::test]
async fn discover_propagates_list_failure() {
    let m = Mock::start(vec![(
        "/v1/models",
        Arc::new(|_| HttpResp::txt(500, "boom")),
    )])
    .await;
    let e = engine_for(&m, 5_000, TransportPolicy::none());
    let err = e.discover(&cfg()).await.unwrap_err();
    assert!(err.to_string().contains("500"), "got: {err}");
}

#[tokio::test]
async fn list_models_bare_array() {
    let m = Mock::start(vec![(
        "/v1/models",
        Arc::new(|_| {
            HttpResp::new(
                200,
                json!([{"id": "a"}, {"id": "b", "object": "model"}]).to_string(),
            )
        }),
    )])
    .await;
    let e = engine_for(&m, 5_000, TransportPolicy::none());
    let ids = e.list_models().await.unwrap();
    assert_eq!(ids.len(), 2);
}

#[tokio::test]
async fn list_models_models_envelope() {
    let m = Mock::start(vec![(
        "/v1/models",
        Arc::new(|_| HttpResp::new(200, json!({"models": [{"id": "x"}]}).to_string())),
    )])
    .await;
    let e = engine_for(&m, 5_000, TransportPolicy::none());
    assert_eq!(e.list_models().await.unwrap().len(), 1);
}

#[tokio::test]
async fn list_models_null_data_is_empty_not_error() {
    let m = Mock::start(vec![(
        "/v1/models",
        Arc::new(|_| HttpResp::new(200, json!({"data": null, "object": "list"}).to_string())),
    )])
    .await;
    let e = engine_for(&m, 5_000, TransportPolicy::none());
    assert!(e.list_models().await.unwrap().is_empty());
}

#[tokio::test]
async fn list_models_html_garbage_is_list_failed() {
    let m = Mock::start(vec![(
        "/v1/models",
        Arc::new(|_| HttpResp::txt(200, "<html>welcome</html>")),
    )])
    .await;
    let e = engine_for(&m, 5_000, TransportPolicy::none());
    let err = e.list_models().await.unwrap_err();
    assert!(err.to_string().contains("not JSON"), "got: {err}");
}

#[tokio::test]
async fn list_models_500_reports_status() {
    let m = Mock::start(vec![(
        "/v1/models",
        Arc::new(|_| HttpResp::txt(500, "boom")),
    )])
    .await;
    let e = engine_for(&m, 5_000, TransportPolicy::none());
    let err = e.list_models().await.unwrap_err();
    assert!(err.to_string().contains("500"), "got: {err}");
}

// ---------------------------------------------------------------------------
// Chat responses with unusable shapes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_200_without_choices_is_flagged_as_anomaly() {
    let m = Mock::start(vec![(
        "/v1/chat/completions",
        Arc::new(|_| {
            HttpResp::new(
                200,
                json!({"id": "x", "object": "chat.completion"}).to_string(),
            )
        }),
    )])
    .await;
    let d = discover_one(&m, "m", &cfg()).await;
    assert!(d.reachable, "a 2xx is reachable, but must be flagged");
    assert!(
        d.anomalies.iter().any(|a| a.contains("no usable message")),
        "anomalies: {:?}",
        d.anomalies
    );
}

#[tokio::test]
async fn chat_200_empty_choices_array_is_flagged_as_anomaly() {
    let m = Mock::start(vec![(
        "/v1/chat/completions",
        Arc::new(|_| HttpResp::new(200, json!({"choices": []}).to_string())),
    )])
    .await;
    let d = discover_one(&m, "m", &cfg()).await;
    assert!(d.reachable);
    assert!(d.anomalies.iter().any(|a| a.contains("no usable message")));
}

#[tokio::test]
async fn chat_200_without_usage_is_tolerated() {
    let m = Mock::start(vec![(
        "/v1/chat/completions",
        Arc::new(|_| {
            HttpResp::new(
                200,
                json!({"choices": [{"message": {"role": "assistant", "content": "hi"}}]})
                    .to_string(),
            )
        }),
    )])
    .await;
    let d = discover_one(&m, "m", &cfg()).await;
    assert!(d.reachable);
    assert!(
        !d.anomalies.iter().any(|a| a.contains("no usable message")),
        "anomalies: {:?}",
        d.anomalies
    );
}

#[tokio::test]
async fn chat_content_parts_array_is_concatenated() {
    let m = Mock::start(vec![(
        "/v1/chat/completions",
        Arc::new(|_| {
            HttpResp::new(
                200,
                json!({"choices": [{"message": {"role": "assistant",
                    "content": [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]}}]}).to_string(),
            )
        }),
    )])
    .await;
    let d = discover_one(&m, "m", &cfg()).await;
    assert!(d.reachable);
}

#[tokio::test]
async fn chat_404_empty_body_is_model_not_found() {
    let m = Mock::start(vec![(
        "/v1/chat/completions",
        Arc::new(|_| HttpResp::new(404, "")),
    )])
    .await;
    let d = discover_one(&m, "m", &cfg()).await;
    assert!(!d.reachable);
    let blocker = d.blocker.unwrap_or_default();
    assert!(
        blocker.contains("not served by this gateway"),
        "blocker was: {blocker}"
    );
}

#[tokio::test]
async fn chat_429_empty_body_is_rate_limited() {
    let m = Mock::start(vec![(
        "/v1/chat/completions",
        Arc::new(|_| HttpResp::new(429, "")),
    )])
    .await;
    let d = discover_one(&m, "m", &cfg()).await;
    assert!(!d.reachable);
    assert!(d.blocker.unwrap_or_default().contains("throttled"));
}

#[tokio::test]
async fn auth_401_empty_body() {
    let m = Mock::start(vec![(
        "/v1/chat/completions",
        Arc::new(|_| HttpResp::new(401, "")),
    )])
    .await;
    let d = discover_one(&m, "m", &cfg()).await;
    assert!(!d.reachable);
}

// ---------------------------------------------------------------------------
// Transport behaviour
// ---------------------------------------------------------------------------

/// Retry behaviour is a transport property; assert it directly so the count
/// is unambiguous (discover_one fires many requests per model).
#[tokio::test]
async fn retry_after_header_is_honoured() {
    let attempt = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let attempt2 = attempt.clone();
    let m = Mock::start(vec![(
        "/v1/chat/completions",
        Arc::new(move |_| {
            let n = attempt2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n == 0 {
                let mut r = HttpResp::new(429, "");
                r.headers.push(("retry-after".to_string(), "1".to_string()));
                r
            } else {
                ok_chat("OK")
            }
        }),
    )])
    .await;
    let t = ai_discovery::probe::Transport::with_policy(
        m.base_url(),
        "k",
        Duration::from_secs(10),
        TransportPolicy {
            min_interval: Duration::ZERO,
            max_attempts: 2,
            max_timeout_attempts: 2,
            base_backoff: Duration::from_millis(50),
            max_backoff: Duration::from_secs(1),
        },
    )
    .unwrap();
    let raw = t
        .post("chat/completions", &json!({"model": "m", "messages": []}))
        .await;
    assert_eq!(raw.status, 200);
    assert_eq!(m.hits("/v1/chat/completions").await, 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn timeout_is_classified_as_timeout() {
    let m = Mock::start(vec![(
        "/v1/chat/completions",
        Arc::new(|_| {
            std::thread::sleep(Duration::from_secs(2));
            HttpResp::new(200, "{}")
        }),
    )])
    .await;
    let policy = TransportPolicy {
        min_interval: Duration::ZERO,
        max_attempts: 1,
        max_timeout_attempts: 1,
        base_backoff: Duration::ZERO,
        max_backoff: Duration::ZERO,
    };
    let e = engine_for(&m, 300, policy);
    let d = e.discover_one("m", None, true, &cfg()).await;
    assert!(!d.reachable);
    assert!(
        d.blocker
            .as_deref()
            .unwrap_or("")
            .contains("no response within"),
        "blocker: {:?}",
        d.blocker
    );
}

/// A timeout on a sweep should cost one attempt, not `max_attempts ×
/// timeout`: each attempt burns the full window, so repeated retries
/// multiply dead-model latency by N (measured on the real NVIDIA sweep:
/// 90 s × 4 attempts ≈ 6.5 min per dead model).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn timeout_is_not_repeated_in_sweep_mode() {
    let m = Mock::start(vec![(
        "/v1/chat/completions",
        Arc::new(|_| {
            std::thread::sleep(Duration::from_secs(2));
            HttpResp::new(200, "{}")
        }),
    )])
    .await;
    let t = ai_discovery::probe::Transport::with_policy(
        m.base_url(),
        "k",
        Duration::from_millis(300),
        TransportPolicy {
            min_interval: Duration::ZERO,
            max_attempts: 4,
            max_timeout_attempts: 1,
            base_backoff: Duration::from_millis(10),
            max_backoff: Duration::from_millis(50),
        },
    )
    .unwrap();
    let raw = t
        .post("chat/completions", &json!({"model": "m", "messages": []}))
        .await;
    assert_eq!(
        raw.error().unwrap().class,
        ai_discovery::ErrorClass::Timeout
    );
    assert_eq!(m.hits("/v1/chat/completions").await, 1);
    // The default (max_timeout_attempts = 2) retries exactly once.
    let m2 = Mock::start(vec![(
        "/v1/chat/completions",
        Arc::new(|_| {
            std::thread::sleep(Duration::from_secs(2));
            HttpResp::new(200, "{}")
        }),
    )])
    .await;
    let t2 = ai_discovery::probe::Transport::with_policy(
        m2.base_url(),
        "k",
        Duration::from_millis(300),
        TransportPolicy {
            min_interval: Duration::ZERO,
            max_attempts: 4,
            max_timeout_attempts: 2,
            base_backoff: Duration::from_millis(10),
            max_backoff: Duration::from_millis(50),
        },
    )
    .unwrap();
    let _ = t2
        .post("chat/completions", &json!({"model": "m", "messages": []}))
        .await;
    assert_eq!(m2.hits("/v1/chat/completions").await, 2);
}

#[tokio::test]
async fn connection_reset_is_network_class() {
    // Server accepts then immediately closes; client sees a connection error.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = listener.accept().await;
        // drop: EOF immediately
    });
    let e = DiscoveryEngine::with_policy(
        "mock",
        format!("http://{addr}/v1"),
        "k",
        Duration::from_secs(5),
        TransportPolicy::none(),
    )
    .unwrap();
    let d = e.discover_one("m", None, true, &cfg()).await;
    assert!(!d.reachable);
    // Not a panic is the main assertion; class must be Network or Timeout.
    let err = d.blocker.unwrap_or_default();
    assert!(
        err.contains("network") || err.contains("no response"),
        "blocker: {err}"
    );
}

// ---------------------------------------------------------------------------
// Role discovery via endpoint routing
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embedding_model_is_routed_by_endpoint() {
    let m = Mock::start(vec![
        (
            "/v1/chat/completions",
            Arc::new(|_| HttpResp::txt(404, "404 page not found")),
        ),
        (
            "/v1/embeddings",
            Arc::new(|_| {
                HttpResp::new(
                    200,
                    json!({"data": [{"embedding": [0.1, 0.2, 0.3]}]}).to_string(),
                )
            }),
        ),
    ])
    .await;
    let d = discover_one(&m, "m", &cfg()).await;
    assert_eq!(d.role, ModelRole::Embedding);
    assert!(d.accepted_endpoints.iter().any(|p| p == "embeddings"));
}

#[tokio::test]
async fn image_model_is_routed_via_400_rejection() {
    // A 400 enumerating valid sizes is positive evidence the endpoint routes
    // to the model (J-017).
    let m = Mock::start(vec![
        (
            "/v1/chat/completions",
            Arc::new(|_| HttpResp::txt(404, "404 page not found")),
        ),
        (
            "/v1/images/generations",
            Arc::new(|_| {
                HttpResp::new(
                    400,
                    r#"{"error":{"message":"invalid size, valid are [512, 1024]"}}"#,
                )
            }),
        ),
    ])
    .await;
    let d = discover_one(&m, "m", &cfg()).await;
    assert_eq!(d.role, ModelRole::ImageGeneration);
    assert!(
        d.accepted_endpoints
            .iter()
            .any(|p| p.contains("images/generations"))
    );
}

#[tokio::test]
async fn endpoint_5xx_is_anomaly_not_role() {
    let m = Mock::start(vec![
        (
            "/v1/chat/completions",
            Arc::new(|_| HttpResp::txt(404, "404 page not found")),
        ),
        (
            "/v1/images/generations",
            Arc::new(|_| HttpResp::txt(500, "Error during inference")),
        ),
    ])
    .await;
    let d = discover_one(&m, "m", &cfg()).await;
    assert_eq!(d.role, ModelRole::Unknown);
    assert!(
        d.anomalies.iter().any(|a| a.contains("images/generations")),
        "anomalies: {:?}",
        d.anomalies
    );
}

// ---------------------------------------------------------------------------
// Capability probes that must observe, not assume
// ---------------------------------------------------------------------------

#[tokio::test]
async fn thinking_toggle_detected_by_observation_only() {
    // The model always reasons, except when the (only) honest spelling
    // `thinking.type=disabled` is present. A spelling that is *accepted but
    // ignored* (HTTP 200, reasoning still present) must be recorded as a
    // no-op, and the accepted-but-ignored one must not win.
    let m = Mock::start(vec![(
        "/v1/chat/completions",
        chat_proxy(|body| {
            let thinking_disabled = body
                .get("thinking")
                .and_then(|t| t.get("type"))
                .and_then(|t| t.as_str())
                == Some("disabled");
            if thinking_disabled {
                HttpResp::new(
                    200,
                    json!({
                        "choices": [{"message": {"role": "assistant", "content": "4"},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 1,
                                  "completion_tokens_details": {"reasoning_tokens": 0}}
                    })
                    .to_string(),
                )
            } else {
                HttpResp::new(
                    200,
                    json!({
                        "choices": [{"message": {"role": "assistant", "content": "4",
                                                 "reasoning": "but 2+2 = 4 because…"},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 10,
                                  "completion_tokens_details": {"reasoning_tokens": 9}}
                    })
                    .to_string(),
                )
            }
        }),
    )])
    .await;
    let d = discover_one(&m, "m", &cfg()).await;
    let t = d.thinking.expect("thinking probe ran");
    assert!(t.emits_reasoning);
    assert_eq!(
        t.disable_spelling.as_deref(),
        Some("thinking.type=disabled")
    );
    let noops = t
        .observations
        .iter()
        .filter(|(l, ok)| *ok && l.as_str() != "thinking.type=disabled")
        .count();
    assert_eq!(
        noops, 0,
        "no other spelling may be credited: {:?}",
        t.observations
    );
}

#[tokio::test]
async fn max_output_ceiling_mined_from_rejection() {
    let m = Mock::start(vec![(
        "/v1/chat/completions",
        chat_proxy(|body| {
            let mt = body.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
            if mt > 100_000 {
                HttpResp::new(
                    400,
                    r#"{"error":{"message":"field MaxTokens invalid, should be in [1, 65536]"}}"#,
                )
            } else {
                ok_chat("hi")
            }
        }),
    )])
    .await;
    let d = discover_one(&m, "m", &cfg()).await;
    assert_eq!(d.max_output_tokens.value, 65536);
    assert_eq!(
        d.max_output_tokens.source,
        ai_discovery::Source::Inferred,
        "evidence: {}",
        d.max_output_tokens.evidence
    );
}

#[tokio::test]
async fn context_window_binary_searches_until_rejection() {
    // Mock rejects prompts whose payload exceeds ~4k tokens (4 chars/token),
    // mimicking "maximum context length is 4096 tokens".
    let m = Mock::start(vec![(
        "/v1/chat/completions",
        chat_proxy(|body| {
            let content = body
                .get("messages")
                .and_then(|ms| ms.get(0))
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
                .unwrap_or("");
            let approx_tokens = content.len() / 4;
            if approx_tokens > 4096 {
                HttpResp::new(
                    400,
                    r#"{"error":{"message":"maximum context length is 4096 tokens"}}"#,
                )
            } else {
                let used = content.len() / 4;
                HttpResp::new(
                    200,
                    json!({
                        "choices": [{"message": {"role": "assistant", "content": "OK"},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": used, "completion_tokens": 1}
                    })
                    .to_string(),
                )
            }
        }),
    )])
    .await;
    let mut c = cfg();
    c.probe_context = true;
    c.max_context_probe = 8192;
    c.context_rounds = 6;
    let d = discover_one(&m, "m", &c).await;
    let v = d.context_window.value;
    assert!((3900..=4200).contains(&v), "value was {v}");
    assert_eq!(
        d.context_window.source,
        ai_discovery::Source::Probed,
        "evidence: {}",
        d.context_window.evidence
    );
}

#[tokio::test]
async fn streaming_sse_detected() {
    let m = Mock::start(vec![(
        "/v1/chat/completions",
        Arc::new(|_| {
            HttpResp::txt(
                200,
                "data: {\"choices\":[{\"delta\":{\"content\":\"O\"}}]}\n\ndata: [DONE]\n\n",
            )
        }),
    )])
    .await;
    let d = discover_one(&m, "m", &cfg()).await;
    assert_eq!(d.streaming.as_ref().map(|f| f.value), Some(true));
}

#[tokio::test]
async fn streaming_silently_ignored_is_recorded() {
    // 200 with a plain JSON body: the model accepted `stream: true` but did
    // not stream. Must be recorded as `false` with an anomaly, not as true.
    let m = Mock::start(vec![(
        "/v1/chat/completions",
        Arc::new(|_| {
            let mut r = ok_chat("OK");
            r.body = json!({
                "id": "x", "object": "chat.completion", "created": 1, "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "the text contains data: points"},
                             "finish_reason": "stop"}]
            }).to_string();
            r
        }),
    )])
    .await;
    let d = discover_one(&m, "m", &cfg()).await;
    assert_eq!(d.streaming.as_ref().map(|f| f.value), Some(false));
    assert!(d.anomalies.iter().any(|a| a.contains("stream=true")));
}

#[tokio::test]
async fn json_object_but_no_json_schema_is_split_capability() {
    let m = Mock::start(vec![(
        "/v1/chat/completions",
        chat_proxy(|body| {
            let rf = body.get("response_format");
            let is_schema = rf
                .and_then(|r| r.get("type"))
                .and_then(|t| t.as_str())
                == Some("json_schema");
            if is_schema {
                HttpResp::new(
                    400,
                    r#"{"error":{"message":"guided_grammar has compile_grammar_error"}}"#,
                )
            } else {
                HttpResp::new(
                    200,
                    json!({"choices": [{"message": {"role": "assistant", "content": "{\"ok\":true}"},
                                        "finish_reason": "stop"}]}).to_string(),
                )
            }
        }),
    )])
    .await;
    let d = discover_one(&m, "m", &cfg()).await;
    let s = d.structured_output.expect("probe ran");
    assert!(s.value, "json_object alone yields true");
    assert!(s.evidence.contains("json_object=true"));
    assert!(s.evidence.contains("json_schema=false"));
}

/// Stochastic tool-call emission must not be judged from a single sample:
/// a model that produces a call on 2 of 3 samples is supported; 1 of 3 is
/// not (majority), and the sample counts must be visible in evidence.
#[tokio::test]
async fn tools_probe_uses_majority_vote() {
    let n = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let n2 = n.clone();
    let m = Mock::start(vec![(
        "/v1/chat/completions",
        Arc::new(move |body| {
            // Only tool probes carry a `tools` array; index those separately
            // so other probes (vision, thinking, …) never shift the samples.
            if serde_json::from_str::<Value>(body)
                .ok()
                .and_then(|b| b.get("tools").cloned())
                .is_none()
            {
                return ok_chat("fine");
            }
            let i = n2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // Tool-call present on samples 1 and 3, absent on 2.
            let (content, tool_calls) = if i == 2 {
                ("I need to look it up.", None)
            } else {
                (
                    "I will check.",
                    Some(json!([{
                        "id": "call_1", "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                    }])),
                )
            };
            HttpResp::new(
                200,
                json!({
                    "choices": [{"message": {"role": "assistant", "content": content,
                                             "tool_calls": tool_calls},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}
                })
                .to_string(),
            )
        }),
    )])
    .await;
    let d = discover_one(&m, "m", &cfg()).await;
    let t = d.tools.expect("tools probe ran");
    assert!(t.value, "2/3 majority should be supported");
    assert!(t.evidence.contains("2/3"), "evidence: {}", t.evidence);
}

#[tokio::test]
async fn tools_probe_minority_is_not_supported() {
    let n = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let n2 = n.clone();
    let m = Mock::start(vec![(
        "/v1/chat/completions",
        Arc::new(move |body| {
            if serde_json::from_str::<Value>(body)
                .ok()
                .and_then(|b| b.get("tools").cloned())
                .is_none()
            {
                return ok_chat("fine");
            }
            let i = n2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let (content, tool_calls) = if i == 1 {
                (
                    "I will check.",
                    Some(json!([{
                        "id": "call_1", "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                    }])),
                )
            } else {
                ("I need to look it up.", None)
            };
            HttpResp::new(
                200,
                json!({
                    "choices": [{"message": {"role": "assistant", "content": content,
                                             "tool_calls": tool_calls},
                                 "finish_reason": "stop"}]
                })
                .to_string(),
            )
        }),
    )])
    .await;
    let d = discover_one(&m, "m", &cfg()).await;
    let t = d.tools.expect("tools probe ran");
    assert!(!t.value, "1/3 minority should be not supported");
    assert!(t.evidence.contains("1/3"), "evidence: {}", t.evidence);
}

#[tokio::test]
async fn vision_probe_5xx_is_inconclusive_not_negative() {
    // A 500 on the image request is a server verdict, not a capability
    // verdict (J-031): a genuinely vision-capable model can 500 on image
    // parts; the SDK must not report "no vision" from that.
    let m = Mock::start(vec![(
        "/v1/chat/completions",
        chat_proxy(|body| {
            let has_image = serde_json::to_string(body)
                .map(|s| s.contains("image_url"))
                .unwrap_or(false);
            if has_image {
                HttpResp::txt(
                    500,
                    "Internal Server Error: Error while making inference request",
                )
            } else {
                ok_chat("fine")
            }
        }),
    )])
    .await;
    let d = discover_one(&m, "m", &cfg()).await;
    assert!(
        !d.input_modalities
            .value
            .iter()
            .any(|m| format!("{m:?}") == "Image"),
        "no image modality from an inconclusive probe"
    );
    assert!(
        d.anomalies.iter().any(|a| a.contains("inconclusive")),
        "anomalies: {:?}",
        d.anomalies
    );
    // The vision fact must carry low confidence, mirroring "unknown".
    let vf = d.input_modalities.confidence;
    assert!(vf < 0.5, "confidence was {vf}");
}

/// A model that rejects plain-text input is not broken, it has a different
/// input contract (J-014: `nvidia/nemotron-parse`). The reason must be
/// traceable, not collapsed into a generic bad_request.
#[tokio::test]
async fn non_text_rejection_is_a_traceable_anomaly() {
    let m = Mock::start(vec![(
        "/v1/chat/completions",
        Arc::new(|_| {
            HttpResp::new(
                400,
                r#"{"object":"error","message":"Content cannot be a plain string. The model does not support text input."}"#,
            )
        }),
    )])
    .await;
    let d = discover_one(&m, "m", &cfg()).await;
    assert!(!d.reachable);
    assert!(
        d.anomalies.iter().any(|a| a.contains("plain-text input")),
        "anomalies: {:?}",
        d.anomalies
    );
}

#[tokio::test]
async fn vision_probe_2xx_marks_image_input() {
    let m = Mock::start(vec![(
        "/v1/chat/completions",
        Arc::new(|_| ok_chat("circle")),
    )])
    .await;
    let d = discover_one(&m, "m", &cfg()).await;
    assert!(
        d.input_modalities
            .value
            .iter()
            .any(|m| format!("{m:?}") == "Image"),
        "modalities: {:?}",
        d.input_modalities.value
    );
}
