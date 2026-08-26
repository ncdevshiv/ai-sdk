//! End-to-end protocol tests: drive `Sidecar::serve` over in-memory duplexes
//! with fake providers and assert the JSON-RPC frames a host observes.

use std::sync::{Arc, OnceLock};

use ai_core::{AiClient, ChatRequest, Completion, EventStream, Model, ModelInfo, Provider};
use ai_errors::{AiError, ValidationError};
use ai_sidecar::{MAX_FRAME_BYTES, Sidecar, read_line_capped};
use ai_types::{ModelId, ProviderId, StreamEvent, ToolCall, Usage};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{Mutex, watch};

type SharedWriter = Arc<Mutex<Box<dyn tokio::io::AsyncWrite + Unpin + Send>>>;

const FAKE_INFO: fn(&str) -> ModelInfo = |id| {
    let mut info = ModelInfo::new(ProviderId::new("fake"), ModelId::new(id), 8192, 4096);
    info.name = format!("Fake {id}");
    info
};

struct FakeModel {
    info: ModelInfo,
}

#[async_trait::async_trait]
impl Model for FakeModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    async fn generate(&self, request: ChatRequest) -> Result<Completion, AiError> {
        let text = message_text(&request);
        Ok(Completion {
            provider: ProviderId::new("fake"),
            model: ModelId::new("m1"),
            text,
            tool_calls: Vec::new(),
            usage: Usage::new(3, 5),
            reasoning: None,
            raw: Value::Null,
            finish_reason: Some("stop".to_string()),
        })
    }

    async fn stream(&self, request: ChatRequest) -> Result<EventStream, AiError> {
        if message_text(&request) == "boom" {
            return Err(AiError::Validation(ValidationError::new("boom failure")));
        }
        if message_text(&request) == "hang" {
            return Ok(Box::pin(futures::stream::pending()));
        }
        let events: Vec<Result<StreamEvent, AiError>> = vec![
            Ok(StreamEvent::TextDelta {
                delta: "hel".to_string(),
            }),
            Ok(StreamEvent::TextDelta {
                delta: "lo".to_string(),
            }),
            Ok(StreamEvent::ReasoningDelta {
                delta: "thinking".to_string(),
            }),
            Ok(StreamEvent::ToolCallStarted {
                id: "call-1".to_string(),
                name: "lookup".to_string(),
            }),
            Ok(StreamEvent::ToolCallDelta {
                id: "call-1".to_string(),
                arguments_delta: "{\"q\"".to_string(),
            }),
            Ok(StreamEvent::ToolCallDelta {
                id: "call-1".to_string(),
                arguments_delta: ":42}".to_string(),
            }),
            Ok(StreamEvent::ToolCallCompleted {
                call: ToolCall {
                    id: "call-1".to_string(),
                    name: "lookup".to_string(),
                    arguments: "{\"q\":42}".to_string(),
                },
            }),
            Ok(StreamEvent::UsageUpdate {
                usage: Usage::new(3, 5),
            }),
            Ok(StreamEvent::Completed {
                finish_reason: Some("stop".to_string()),
            }),
        ];
        if message_text(&request) == "held" {
            // First event stalls until [`release_held_stream`] fires, keeping
            // the pump registered in the sidecar's streams map.
            let mut gate = held_stream_gate().subscribe();
            let held = futures::stream::once(async move {
                while !*gate.borrow_and_update() {
                    gate.changed().await.expect("gate sender stays alive");
                }
                Ok(StreamEvent::TextDelta {
                    delta: "after-release".to_string(),
                })
            });
            return Ok(Box::pin(futures::StreamExt::chain(
                held,
                futures::stream::iter(events),
            )));
        }
        Ok(Box::pin(futures::stream::iter(events)))
    }
}

/// Shared gate for `held` streams; process-local and starting closed.
fn held_stream_gate() -> &'static watch::Sender<bool> {
    static GATE: OnceLock<watch::Sender<bool>> = OnceLock::new();
    GATE.get_or_init(|| watch::channel(false).0)
}

fn release_held_stream() {
    held_stream_gate()
        .send(true)
        .expect("at least one held receiver exists");
}

struct FakeProvider;

#[async_trait::async_trait]
impl Provider for FakeProvider {
    fn id(&self) -> &str {
        "fake"
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, AiError> {
        Ok(vec![FAKE_INFO("m1")])
    }

    fn model(&self, model_id: &str) -> Result<Arc<dyn Model>, AiError> {
        Ok(Arc::new(FakeModel {
            info: FAKE_INFO(model_id),
        }))
    }
}

fn message_text(request: &ChatRequest) -> String {
    request
        .messages
        .iter()
        .map(|message| message.text_content())
        .collect::<String>()
}

/// Spawns the sidecar over duplex pipes; returns the host-side writer and
/// line reader.
async fn spawn(client: AiClient) -> (tokio::io::DuplexStream, BufReader<tokio::io::DuplexStream>) {
    let (host_to_sidecar, sidecar_input) = tokio::io::duplex(64 * 1024);
    let (sidecar_output, host_from_sidecar) = tokio::io::duplex(64 * 1024);
    let writer: SharedWriter = Arc::new(Mutex::new(Box::new(sidecar_output)));
    let sidecar = Arc::new(Sidecar::with_client(writer, client));
    tokio::spawn(sidecar.serve(sidecar_input));
    (host_to_sidecar, BufReader::new(host_from_sidecar))
}

async fn send(writer: &mut tokio::io::DuplexStream, frame: Value) {
    writer
        .write_all(format!("{}\n", serde_json::to_string(&frame).unwrap()).as_bytes())
        .await
        .unwrap();
    writer.flush().await.unwrap();
}

async fn read_frame(reader: &mut BufReader<tokio::io::DuplexStream>) -> Value {
    let mut line = String::new();
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        reader.read_line(&mut line),
    )
    .await
    .expect("frame arrives within timeout")
    .expect("stream readable");
    let frame: Value = serde_json::from_str(line.trim()).expect("valid JSON frame");
    frame
}

fn user_request(text: &str) -> Value {
    json!({
        "messages": [{ "role": "user", "parts": [{ "type": "text", "text": text }] }],
    })
}

#[tokio::test]
async fn initialize_reports_protocol_version() {
    let client = AiClient::builder()
        .provider(Arc::new(FakeProvider))
        .build()
        .unwrap();
    let (mut writer, mut reader) = spawn(client).await;
    send(
        &mut writer,
        json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {} }),
    )
    .await;
    let frame = read_frame(&mut reader).await;
    assert_eq!(frame["id"], 1);
    assert_eq!(frame["result"]["protocol"], 1);
}

#[tokio::test]
async fn stream_forwards_events_and_done() {
    let client = AiClient::builder()
        .provider(Arc::new(FakeProvider))
        .build()
        .unwrap();
    let (mut writer, mut reader) = spawn(client).await;
    send(
        &mut writer,
        json!({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "chat.stream",
            "params": { "stream_id": "s-1", "reference": "fake:m1", "request": user_request("hi") },
        }),
    )
    .await;
    let accepted = read_frame(&mut reader).await;
    assert_eq!(accepted["id"], 7);
    assert_eq!(accepted["result"]["accepted"], true);

    let mut events = Vec::new();
    loop {
        let frame = read_frame(&mut reader).await;
        assert_eq!(frame["method"], "chat/event");
        assert_eq!(frame["params"]["stream_id"], "s-1");
        events.push(frame["params"]["event"].clone());
        if events.len() == 9 {
            break;
        }
    }
    assert_eq!(events[0], json!({ "type": "text_delta", "delta": "hel" }));
    assert_eq!(
        events[2],
        json!({ "type": "reasoning_delta", "delta": "thinking" })
    );
    assert_eq!(
        events[3],
        json!({ "type": "tool_call_started", "id": "call-1", "name": "lookup" })
    );
    assert_eq!(
        events[6],
        json!({ "type": "tool_call_completed", "call": { "id": "call-1", "name": "lookup", "arguments": "{\"q\":42}" } })
    );
    assert_eq!(events[7]["type"], "usage_update");
    assert_eq!(
        events[8],
        json!({ "type": "completed", "finish_reason": "stop" })
    );

    let done = read_frame(&mut reader).await;
    assert_eq!(done["method"], "chat/done");
    assert_eq!(done["params"]["ok"], true);
}

#[tokio::test]
async fn stream_setup_failure_is_request_error() {
    let client = AiClient::builder()
        .provider(Arc::new(FakeProvider))
        .build()
        .unwrap();
    let (mut writer, mut reader) = spawn(client).await;
    send(
        &mut writer,
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "chat.stream",
            "params": { "stream_id": "s-err", "reference": "fake:m1", "request": user_request("boom") },
        }),
    )
    .await;
    // A failure before any event streams rejects the request itself.
    let frame = read_frame(&mut reader).await;
    assert_eq!(frame["id"], 2);
    assert_eq!(frame["error"]["code"], -32000);
    assert_eq!(frame["error"]["data"]["kind"], "validation");
    assert_eq!(frame["error"]["data"]["retryable"], false);
}

#[tokio::test]
async fn cancel_aborts_pending_stream() {
    let client = AiClient::builder()
        .provider(Arc::new(FakeProvider))
        .build()
        .unwrap();
    let (mut writer, mut reader) = spawn(client).await;
    send(
        &mut writer,
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "chat.stream",
            "params": { "stream_id": "s-hang", "reference": "fake:m1", "request": user_request("hang") },
        }),
    )
    .await;
    let _accepted = read_frame(&mut reader).await;

    send(
        &mut writer,
        json!({ "jsonrpc": "2.0", "id": 4, "method": "stream.cancel", "params": { "stream_id": "s-hang" } }),
    )
    .await;
    let cancelled = read_frame(&mut reader).await;
    assert_eq!(cancelled["id"], 4);
    assert_eq!(cancelled["result"]["cancelled"], true);

    send(
        &mut writer,
        json!({ "jsonrpc": "2.0", "id": 5, "method": "chat.generate", "params": { "reference": "fake:m1", "request": user_request("echo me") } }),
    )
    .await;
    let completion = read_frame(&mut reader).await;
    assert_eq!(completion["id"], 5);
    assert_eq!(completion["result"]["text"], "echo me");
    assert_eq!(completion["result"]["usage"]["input_tokens"], 3);
}

#[tokio::test]
async fn unknown_method_is_method_not_found() {
    let client = AiClient::builder()
        .provider(Arc::new(FakeProvider))
        .build()
        .unwrap();
    let (mut writer, mut reader) = spawn(client).await;
    send(
        &mut writer,
        json!({ "jsonrpc": "2.0", "id": 9, "method": "nope", "params": {} }),
    )
    .await;
    let frame = read_frame(&mut reader).await;
    assert_eq!(frame["error"]["code"], -32601);
}

#[tokio::test]
async fn model_list_returns_catalog() {
    let client = AiClient::builder()
        .provider(Arc::new(FakeProvider))
        .build()
        .unwrap();
    let (mut writer, mut reader) = spawn(client).await;
    send(
        &mut writer,
        json!({ "jsonrpc": "2.0", "id": 11, "method": "model.list", "params": { "provider": "fake" } }),
    )
    .await;
    let frame = read_frame(&mut reader).await;
    assert_eq!(frame["id"], 11);
    assert_eq!(frame["result"][0]["id"], "m1");
    assert_eq!(frame["result"][0]["name"], "Fake m1");
}

/// An unconfigured sidecar: `model.discover` must answer without any
/// `configure` generation, which is the whole point of interrogating a draft.
async fn spawn_unconfigured() -> (tokio::io::DuplexStream, BufReader<tokio::io::DuplexStream>) {
    let (host_to_sidecar, sidecar_input) = tokio::io::duplex(64 * 1024);
    let (sidecar_output, host_from_sidecar) = tokio::io::duplex(64 * 1024);
    let writer: SharedWriter = Arc::new(Mutex::new(Box::new(sidecar_output)));
    let sidecar = Arc::new(Sidecar::new(writer));
    tokio::spawn(sidecar.serve(sidecar_input));
    (host_to_sidecar, BufReader::new(host_from_sidecar))
}

#[tokio::test]
async fn model_discover_requires_base_url_for_openai_compatible_drafts() {
    let (mut writer, mut reader) = spawn_unconfigured().await;
    send(
        &mut writer,
        json!({
            "jsonrpc": "2.0",
            "id": 21,
            "method": "model.discover",
            "params": { "api_key": "sk-draft" },
        }),
    )
    .await;
    let frame = read_frame(&mut reader).await;
    assert_eq!(frame["id"], 21);
    assert_eq!(frame["error"]["code"], -32602);
    assert_eq!(frame["error"]["message"], "missing `base_url`");
}

#[tokio::test]
async fn model_discover_rejects_unknown_dialect_before_any_network_call() {
    let (mut writer, mut reader) = spawn_unconfigured().await;
    send(
        &mut writer,
        json!({
            "jsonrpc": "2.0",
            "id": 22,
            "method": "model.discover",
            "params": { "api": "irc", "base_url": "https://gateway.example/v1" },
        }),
    )
    .await;
    let frame = read_frame(&mut reader).await;
    assert_eq!(frame["id"], 22);
    assert_eq!(frame["error"]["code"], -32000);
    assert_eq!(frame["error"]["data"]["kind"], "configuration");
    assert!(
        frame["error"]["message"]
            .as_str()
            .unwrap()
            .contains("unknown wire dialect `irc`"),
        "{}",
        frame["error"]["message"]
    );
}

#[tokio::test]
async fn configure_honors_explicit_native_dialect_on_custom_route_ids() {
    // Building — not calling — an Anthropic adapter under a custom id proves
    // the dialect overrides id-based selection; the endpoint stays default.
    let (mut writer, mut reader) = spawn_unconfigured().await;
    send(
        &mut writer,
        json!({
            "jsonrpc": "2.0",
            "id": 23,
            "method": "configure",
            "params": { "providers": { "acme-relay": { "api_key": "sk-ant-test", "api": "anthropic" } } },
        }),
    )
    .await;
    let frame = read_frame(&mut reader).await;
    assert_eq!(frame["id"], 23);
    assert_eq!(frame["result"]["ok"], true);
    assert_eq!(frame["result"]["providers"], json!(["acme-relay"]));
}

#[tokio::test]
async fn configure_rejects_unknown_dialect_loudly() {
    let (mut writer, mut reader) = spawn_unconfigured().await;
    send(
        &mut writer,
        json!({
            "jsonrpc": "2.0",
            "id": 24,
            "method": "configure",
            "params": { "providers": { "acme-relay": { "api_key": "k", "api": "smtp" } } },
        }),
    )
    .await;
    let frame = read_frame(&mut reader).await;
    assert_eq!(frame["id"], 24);
    assert_eq!(frame["error"]["code"], -32000);
    assert_eq!(frame["error"]["data"]["kind"], "configuration");
}

/// A second `chat.stream` reusing an active id is rejected with
/// `-32602` while the first pump keeps streaming to completion.
#[tokio::test]
async fn duplicate_stream_id_rejected_and_first_stream_survives() {
    let client = AiClient::builder()
        .provider(Arc::new(FakeProvider))
        .build()
        .unwrap();
    let (mut writer, mut reader) = spawn(client).await;
    send(
        &mut writer,
        json!({
            "jsonrpc": "2.0",
            "id": 41,
            "method": "chat.stream",
            "params": { "stream_id": "dup", "reference": "fake:m1", "request": user_request("held") },
        }),
    )
    .await;
    let accepted = read_frame(&mut reader).await;
    assert_eq!(accepted["id"], 41);
    assert_eq!(accepted["result"]["accepted"], true);

    // The held stream stays registered, so the reuse is a hard error.
    send(
        &mut writer,
        json!({
            "jsonrpc": "2.0",
            "id": 42,
            "method": "chat.stream",
            "params": { "stream_id": "dup", "reference": "fake:m1", "request": user_request("hi") },
        }),
    )
    .await;
    let rejected = read_frame(&mut reader).await;
    assert_eq!(rejected["id"], 42);
    assert_eq!(rejected["error"]["code"], -32602);
    assert_eq!(
        rejected["error"]["message"],
        "stream_id `dup` is already active"
    );

    // The first pump was untouched: releasing it streams a full transcript.
    release_held_stream();
    for _ in 0..10 {
        let frame = read_frame(&mut reader).await;
        assert_eq!(frame["method"], "chat/event");
        assert_eq!(frame["params"]["stream_id"], "dup");
    }
    let done = read_frame(&mut reader).await;
    assert_eq!(done["method"], "chat/done");
    assert_eq!(done["params"]["ok"], true);
    assert_eq!(done["params"]["stream_id"], "dup");

    // Nothing follows the single terminal done.
    let mut stray = String::new();
    let quiet = tokio::time::timeout(
        std::time::Duration::from_millis(300),
        reader.read_line(&mut stray),
    )
    .await;
    assert!(quiet.is_err(), "unexpected extra frame: {stray}");
}

#[tokio::test]
async fn oversized_frame_is_rejected_and_loop_keeps_serving() {
    let client = AiClient::builder()
        .provider(Arc::new(FakeProvider))
        .build()
        .unwrap();
    let (mut writer, mut reader) = spawn(client).await;
    writer
        .write_all(vec![b'x'; MAX_FRAME_BYTES + 1024 * 1024].as_slice())
        .await
        .unwrap();
    writer.write_all(b"\n").await.unwrap();
    send(
        &mut writer,
        json!({ "jsonrpc": "2.0", "id": 31, "method": "initialize", "params": {} }),
    )
    .await;

    let error = read_frame(&mut reader).await;
    assert_eq!(error["error"]["code"], -32602);
    assert!(
        error["error"]["message"]
            .as_str()
            .unwrap()
            .contains("frame exceeds"),
        "{}",
        error["error"]["message"]
    );

    // The read loop survived: the next request still answers.
    let init = read_frame(&mut reader).await;
    assert_eq!(init["id"], 31);
    assert_eq!(init["result"]["protocol"], 1);
}

#[tokio::test]
async fn read_line_capped_enforces_cap_and_eof_contract() {
    let mut cursor = std::io::Cursor::new(b"hello\nworld".to_vec());
    assert_eq!(
        read_line_capped(&mut cursor, 16).await.unwrap(),
        Some(b"hello".to_vec())
    );
    // An unterminated tail at EOF still counts as a line.
    assert_eq!(
        read_line_capped(&mut cursor, 16).await.unwrap(),
        Some(b"world".to_vec())
    );
    assert_eq!(read_line_capped(&mut cursor, 16).await.unwrap(), None);

    let mut cursor = std::io::Cursor::new(vec![b'x'; 32]);
    let error = read_line_capped(&mut cursor, 8).await.unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert!(error.to_string().contains("frame exceeds"), "{error}");
}
