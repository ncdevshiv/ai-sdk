//! AI SDK stdio JSON-RPC sidecar.
//!
//! Exposes the SDK's provider registry, model catalog, and streaming
//! completions over newline-delimited JSON-RPC 2.0 so another process can
//! drive every provider without linking Rust.
//!
//! Framing follows the shared transport contract: a frame with `id` and
//! `method` is a request, `method` alone is a notification, and every request
//! receives exactly one response. Streaming completions answer `chat.stream`
//! immediately, forward [`ai_types::StreamEvent`]s as `chat/event`
//! notifications, and close with one terminal `chat/done` notification.
//!
//! @module ai_sidecar

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ai_core::{AiClient, ChatRequest as WireChatRequest};
use ai_errors::AiError;
use ai_providers::create_provider_with_api;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::RwLock;
use tokio::task::JoinHandle;

/// Wire protocol version. Bump on breaking method or payload changes.
pub const PROTOCOL_VERSION: u64 = 1;

/// Maximum accepted frame size in bytes; larger frames are rejected with a
/// `-32602` response rather than buffered unboundedly.
pub const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// JSON-RPC error codes used by sidecar responses.
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const PROVIDER_FAILURE: i64 = -32000;

/// Writer half shared between the read loop and spawned stream pumps.
pub type SharedWriter = Arc<tokio::sync::Mutex<Box<dyn AsyncWrite + Unpin + Send>>>;

/// One configured provider on the `configure` payload.
#[derive(Debug, Clone, Deserialize)]
pub struct ProviderProfile {
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub default_model: Option<String>,
    /// Explicit wire dialect (`anthropic`, `google`) for a route whose own id
    /// cannot name one; omission keeps id-based adapter selection.
    #[serde(default)]
    pub api: Option<String>,
}

/// The `configure` request parameters.
#[derive(Debug, Default, Deserialize)]
pub struct ConfigureParams {
    #[serde(default)]
    pub providers: HashMap<String, ProviderProfile>,
    #[serde(default)]
    pub default_provider: Option<String>,
}

/// The `model.discover` request parameters: one endpoint interrogation that
/// configuration has not stored yet.
#[derive(Debug, Default, Deserialize)]
pub struct DiscoverParams {
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    /// Wire dialect to interrogate with; omission means OpenAI-compatible.
    #[serde(default)]
    pub api: Option<String>,
}

/// Error raised inside a handler; becomes the JSON-RPC error response.
#[cfg_attr(test, derive(Debug))]
struct RpcError {
    code: i64,
    message: String,
    data: Option<Value>,
}

impl RpcError {
    fn invalid_params(message: String) -> Self {
        Self {
            code: INVALID_PARAMS,
            message,
            data: None,
        }
    }

    fn provider_failure(err: &AiError) -> Self {
        Self {
            code: PROVIDER_FAILURE,
            message: err.to_string(),
            data: Some(error_object(err)),
        }
    }
}

fn error_object(err: &AiError) -> Value {
    let kind = match err {
        AiError::Configuration(_) => "configuration",
        AiError::Authentication(_) => "authentication",
        AiError::Provider(_) => "provider",
        AiError::RateLimit(_) => "rate_limit",
        AiError::Timeout(_) => "timeout",
        AiError::Network(_) => "network",
        AiError::Serialization(_) => "serialization",
        AiError::Validation(_) => "validation",
        AiError::Tool(_) => "tool",
        AiError::Web(_) => "web",
        AiError::Storage(_) => "storage",
        AiError::Agent(_) => "agent",
        AiError::Workflow(_) => "workflow",
        AiError::Cancelled(_) => "cancelled",
        AiError::Internal(_) => "internal",
        _ => "unknown",
    };
    json!({ "kind": kind, "message": err.to_string(), "retryable": err.is_retryable() })
}

async fn send_line(writer: &SharedWriter, frame: &Value) {
    let mut line = serde_json::to_string(frame).expect("frame serializes");
    line.push('\n');
    let mut out = writer.lock().await;
    if let Err(e) = out.write_all(line.as_bytes()).await {
        eprintln!("ai-sidecar: failed to write frame: {e}");
        return;
    }
    if let Err(e) = out.flush().await {
        eprintln!("ai-sidecar: failed to write frame: {e}");
    }
}

/// Reads one `\n`-terminated frame from `reader` without unbounded
/// buffering.
///
/// Returns the bytes before the newline (an unterminated tail at EOF counts
/// as a line), or `Ok(None)` on clean EOF with an empty buffer. When a line
/// would exceed `cap` bytes, buffering stops and the frame is drained
/// through its newline in buffer-sized chunks; the result is then a
/// [`std::io::ErrorKind::InvalidData`] error naming the limit.
pub async fn read_line_capped<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    cap: usize,
) -> std::io::Result<Option<Vec<u8>>> {
    let mut line: Vec<u8> = Vec::new();
    let mut oversized = false;
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            if oversized {
                return Err(oversized_frame_error(cap));
            }
            return if line.is_empty() {
                Ok(None)
            } else {
                Ok(Some(line))
            };
        }
        let chunk_len = available.len();
        match available.iter().position(|byte| *byte == b'\n') {
            Some(newline) => {
                if oversized || line.len() + newline > cap {
                    reader.consume(newline + 1);
                    return Err(oversized_frame_error(cap));
                }
                line.extend_from_slice(&available[..newline]);
                reader.consume(newline + 1);
                return Ok(Some(line));
            }
            None => {
                if !oversized {
                    if line.len() + chunk_len > cap {
                        // The frame is already rejected; stop growing the
                        // buffer and just drain to the newline.
                        oversized = true;
                        line = Vec::new();
                    } else {
                        line.extend_from_slice(available);
                    }
                }
                reader.consume(chunk_len);
            }
        }
    }
}

fn oversized_frame_error(cap: usize) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("frame exceeds {cap} byte limit"),
    )
}

fn required_string<'a>(params: &'a Value, key: &str) -> Result<&'a str, RpcError> {
    params
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| RpcError::invalid_params(format!("missing `{key}`")))
}

/// Sidecar state shared across the read loop and spawned stream pumps.
pub struct Sidecar {
    client: RwLock<Option<AiClient>>,
    streams: Mutex<HashMap<String, JoinHandle<()>>>,
    /// In-flight request responders keyed by assignment order; each task
    /// retires its own slot once its response is written.
    responders: Mutex<HashMap<u64, JoinHandle<()>>>,
    /// Monotonic key source for the `responders` map.
    next_responder: AtomicU64,
    writer: SharedWriter,
}

impl Sidecar {
    /// Creates a sidecar that writes frames to `writer`. Production callers
    /// start unconfigured and send `configure`; [`Sidecar::with_client`]
    /// seeds a client directly for tests.
    pub fn new(writer: SharedWriter) -> Self {
        Self {
            client: RwLock::new(None),
            streams: Mutex::new(HashMap::new()),
            responders: Mutex::new(HashMap::new()),
            next_responder: AtomicU64::new(0),
            writer,
        }
    }

    pub fn with_client(writer: SharedWriter, client: AiClient) -> Self {
        Self {
            client: RwLock::new(Some(client)),
            ..Self::new(writer)
        }
    }

    /// Reads newline-delimited frames from `input` until EOF, then drains
    /// in-flight request responses before aborting any completion streams.
    pub async fn serve<R>(self: Arc<Self>, input: R)
    where
        R: AsyncRead + Unpin,
    {
        let mut reader = BufReader::new(input);
        loop {
            let line = match read_line_capped(&mut reader, MAX_FRAME_BYTES).await {
                Ok(Some(line)) => String::from_utf8_lossy(&line).into_owned(),
                Ok(None) => break,
                Err(error) if error.kind() == std::io::ErrorKind::InvalidData => {
                    // An oversized frame is rejected unread; its id cannot be
                    // known, so the response carries the JSON-RPC null id.
                    send_line(
                        &self.writer,
                        &json!({
                            "jsonrpc": "2.0",
                            "id": Value::Null,
                            "error": {
                                "code": INVALID_PARAMS,
                                "message": error.to_string(),
                                "data": Value::Null,
                            },
                        }),
                    )
                    .await;
                    continue;
                }
                // A read failure ends the loop the same way EOF does.
                Err(_) => break,
            };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let frame: Value = match serde_json::from_str(trimmed) {
                Ok(frame) => frame,
                // Malformed lines are ignored, matching the shared transport.
                Err(_) => continue,
            };
            let method = frame
                .get("method")
                .and_then(Value::as_str)
                .map(String::from);
            let Some(method) = method else {
                continue;
            };
            let id = frame.get("id").cloned();
            let Some(id) = id else {
                // Notifications are not part of the host→sidecar surface.
                continue;
            };
            let params = frame.get("params").cloned().unwrap_or(json!({}));
            let this = Arc::clone(&self);
            let response_id = self.next_responder.fetch_add(1, Ordering::Relaxed);
            let responder = tokio::spawn(async move {
                let result = this.handle_request(method.as_str(), params).await;
                let response = match result {
                    Ok(value) => json!({ "jsonrpc": "2.0", "id": id, "result": value }),
                    Err(error) => json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {
                            "code": error.code,
                            "message": error.message,
                            "data": error.data,
                        },
                    }),
                };
                send_line(&this.writer, &response).await;
                this.responders
                    .lock()
                    .expect("responders lock not poisoned")
                    .remove(&response_id);
            });
            self.responders
                .lock()
                .expect("responders lock not poisoned")
                .insert(response_id, responder);
        }
        // EOF: finish pending responses so no request is dropped, then stop
        // streaming pumps. Streams are cancelled without a terminal
        // `chat/done` because the host is gone. Responder tasks that already
        // wrote their response removed their own slots above.
        let responders: Vec<JoinHandle<()>> = self
            .responders
            .lock()
            .expect("responders lock not poisoned")
            .drain()
            .map(|(_, handle)| handle)
            .collect();
        for responder in responders {
            let _ = responder.await;
        }
        self.abort_all_streams();
    }

    async fn handle_request(
        self: &Arc<Self>,
        method: &str,
        params: Value,
    ) -> Result<Value, RpcError> {
        match method {
            "initialize" => Ok(json!({
                "protocol": PROTOCOL_VERSION,
                "version": env!("CARGO_PKG_VERSION"),
            })),
            "configure" => {
                let parsed: ConfigureParams = serde_json::from_value(params)
                    .map_err(|e| RpcError::invalid_params(e.to_string()))?;
                let mut builder = AiClient::builder();
                for (name, profile) in &parsed.providers {
                    // Keyless profiles are skipped rather than rejected so the
                    // host can re-configure as credentials arrive.
                    let Some(api_key) = profile.api_key.as_deref().filter(|k| !k.is_empty()) else {
                        continue;
                    };
                    let config = ai_config::ProviderConfig {
                        api_key: Some(api_key.to_string()),
                        base_url: profile.base_url.clone(),
                        default_model: profile.default_model.clone(),
                    };
                    let provider = create_provider_with_api(name, profile.api.as_deref(), &config)
                        .map_err(|e| RpcError::provider_failure(&e))?;
                    // The ROUTE name owns the registry key: references the
                    // host sends are `route:model`, and two routes may share
                    // one wire format.
                    builder = builder.provider_as(name.as_str(), provider);
                }
                if let Some(default) = &parsed.default_provider {
                    builder = builder.default_provider(default);
                }
                let client = builder
                    .build()
                    .map_err(|e| RpcError::provider_failure(&e))?;
                let providers = client.provider_ids();
                *self.client.write().await = Some(client);
                Ok(json!({
                    "ok": true,
                    "providers": providers,
                }))
            }
            "provider.list" => {
                let client = self.require_client().await?;
                Ok(json!({ "providers": client.provider_ids() }))
            }
            "model.list" => {
                let client = self.require_client().await?;
                let provider_name = required_string(&params, "provider")?;
                let provider = client
                    .provider(provider_name)
                    .map_err(|e| RpcError::provider_failure(&e))?;
                let models = provider
                    .list_models()
                    .await
                    .map_err(|e| RpcError::provider_failure(&e))?;
                Ok(serde_json::to_value(models)
                    .map_err(|e| RpcError::invalid_params(e.to_string()))?)
            }
            "model.discover" => {
                let parsed: DiscoverParams = serde_json::from_value(params)
                    .map_err(|e| RpcError::invalid_params(e.to_string()))?;
                let dialect = parsed.api.as_deref().filter(|api| !api.is_empty());
                let base_url = parsed.base_url.clone().filter(|url| !url.is_empty());
                // Only the OpenAI-compatible family lacks an SDK default
                // endpoint; interrogating a draft on that dialect without one
                // is a caller bug, not a network failure.
                if matches!(dialect, None | Some("openai-compatible")) && base_url.is_none() {
                    return Err(RpcError::invalid_params("missing `base_url`".to_string()));
                }
                let config = ai_config::ProviderConfig {
                    api_key: parsed.api_key.filter(|key| !key.is_empty()),
                    base_url,
                    default_model: None,
                };
                // A transient provider: the interrogation never joins the
                // configured generation, so probing a draft cannot disturb
                // in-flight streams or later `configure` calls.
                let provider = create_provider_with_api("discover", dialect, &config)
                    .map_err(|e| RpcError::provider_failure(&e))?;
                let models = provider
                    .list_models()
                    .await
                    .map_err(|e| RpcError::provider_failure(&e))?;
                Ok(
                    json!({ "models": serde_json::to_value(models).map_err(|e| RpcError::invalid_params(e.to_string()))? }),
                )
            }
            "model.info" => {
                let client = self.require_client().await?;
                let reference = required_string(&params, "reference")?;
                let (_provider_name, model) = client
                    .resolve_model(reference)
                    .map_err(|e| RpcError::provider_failure(&e))?;
                Ok(serde_json::to_value(model.info())
                    .map_err(|e| RpcError::invalid_params(e.to_string()))?)
            }
            "chat.generate" => {
                let client = self.require_client().await?;
                let reference = required_string(&params, "reference")?;
                let request = parse_chat_request(&params)?;
                let completion = client
                    .generate_request(reference, request)
                    .await
                    .map_err(|e| RpcError::provider_failure(&e))?;
                Ok(serde_json::to_value(completion)
                    .map_err(|e| RpcError::invalid_params(e.to_string()))?)
            }
            "chat.stream" => {
                let client = self.require_client().await?;
                let stream_id = required_string(&params, "stream_id")?.to_string();
                let reference = required_string(&params, "reference")?.to_string();
                let request = parse_chat_request(&params)?;

                // Inserting over an active id would orphan the first pump;
                // reject the newcomer without disturbing that stream.
                if self
                    .streams
                    .lock()
                    .expect("streams lock not poisoned")
                    .contains_key(stream_id.as_str())
                {
                    return Err(RpcError::invalid_params(format!(
                        "stream_id `{stream_id}` is already active"
                    )));
                }

                let mut events = client
                    .stream_request(&reference, request)
                    .await
                    .map_err(|e| RpcError::provider_failure(&e))?;

                let this = Arc::clone(self);
                let pump_stream_id = stream_id.clone();
                let pump = tokio::spawn(async move {
                    loop {
                        match events.next().await {
                            Some(Ok(event)) => {
                                send_line(
                                    &this.writer,
                                    &json!({
                                        "jsonrpc": "2.0",
                                        "method": "chat/event",
                                        "params": {
                                            "stream_id": pump_stream_id,
                                            "event": event,
                                        },
                                    }),
                                )
                                .await;
                            }
                            Some(Err(err)) => {
                                send_line(
                                    &this.writer,
                                    &json!({
                                        "jsonrpc": "2.0",
                                        "method": "chat/done",
                                        "params": {
                                            "stream_id": pump_stream_id,
                                            "ok": false,
                                            "error": error_object(&err),
                                        },
                                    }),
                                )
                                .await;
                                break;
                            }
                            None => {
                                send_line(
                                    &this.writer,
                                    &json!({
                                        "jsonrpc": "2.0",
                                        "method": "chat/done",
                                        "params": {
                                            "stream_id": pump_stream_id,
                                            "ok": true,
                                        },
                                    }),
                                )
                                .await;
                                break;
                            }
                        }
                    }
                    this.streams
                        .lock()
                        .expect("streams lock not poisoned")
                        .remove(pump_stream_id.as_str());
                });
                self.streams
                    .lock()
                    .expect("streams lock not poisoned")
                    .insert(stream_id.clone(), pump);
                Ok(json!({ "accepted": true, "stream_id": stream_id }))
            }
            "stream.cancel" => {
                let stream_id = required_string(&params, "stream_id")?.to_string();
                let handle = self
                    .streams
                    .lock()
                    .expect("streams lock not poisoned")
                    .remove(stream_id.as_str());
                match handle {
                    Some(handle) => {
                        handle.abort();
                        Ok(json!({ "cancelled": true }))
                    }
                    None => Ok(json!({ "cancelled": false })),
                }
            }
            "health" => Ok(json!({
                "ok": true,
                "protocol": PROTOCOL_VERSION,
                "version": env!("CARGO_PKG_VERSION"),
            })),
            other => Err(RpcError {
                code: METHOD_NOT_FOUND,
                message: format!("unknown method `{other}`"),
                data: None,
            }),
        }
    }

    async fn require_client(&self) -> Result<AiClient, RpcError> {
        self.client.read().await.clone().ok_or_else(|| {
            RpcError::invalid_params("no client configured; send `configure` first".to_string())
        })
    }

    /// Graceful drain: wait for pumps to finish then abort leftovers.
    /// Used by the supervisor during a quiesce before killing the old child.
    pub fn drain_streams(&self, deadline_ms: u64) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(deadline_ms);
        while std::time::Instant::now() < deadline {
            let remaining = self
                .streams
                .lock()
                .expect("streams lock not poisoned")
                .len();
            if remaining == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        self.abort_all_streams();
    }

    fn abort_all_streams(&self) {
        let handles: Vec<JoinHandle<()>> = self
            .streams
            .lock()
            .expect("streams lock not poisoned")
            .drain()
            .map(|(_, h)| h)
            .collect();
        for handle in handles {
            handle.abort();
        }
    }
}

fn parse_chat_request(params: &Value) -> Result<WireChatRequest, RpcError> {
    serde_json::from_value(params.get("request").cloned().unwrap_or(Value::Null))
        .map_err(|e| RpcError::invalid_params(format!("invalid request: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::time::Duration;
    use tokio::io::BufReader;

    fn cursor_reader(bytes: &[u8]) -> BufReader<Cursor<&[u8]>> {
        BufReader::new(Cursor::new(bytes))
    }

    /// Unwraps a handler result without requiring `Debug` on the Ok type.
    fn expect_err<T: std::fmt::Debug>(result: Result<T, RpcError>) -> RpcError {
        match result {
            Ok(value) => panic!("expected an error, got {value:?}"),
            Err(err) => err,
        }
    }

    #[tokio::test]
    async fn read_line_capped_returns_frame_and_then_clean_eof() {
        let mut reader = cursor_reader(b"{\"a\":1}\n{\"b\":2}\n");
        assert_eq!(
            read_line_capped(&mut reader, MAX_FRAME_BYTES)
                .await
                .unwrap(),
            Some(b"{\"a\":1}".to_vec())
        );
        assert_eq!(
            read_line_capped(&mut reader, MAX_FRAME_BYTES)
                .await
                .unwrap(),
            Some(b"{\"b\":2}".to_vec())
        );
        assert_eq!(
            read_line_capped(&mut reader, MAX_FRAME_BYTES)
                .await
                .unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn read_line_capped_treats_unterminated_tail_as_a_frame() {
        let mut reader = cursor_reader(b"no trailing newline");
        assert_eq!(
            read_line_capped(&mut reader, 64).await.unwrap(),
            Some(b"no trailing newline".to_vec())
        );
    }

    #[tokio::test]
    async fn read_line_capped_rejects_oversized_frames_with_invalid_data() {
        let oversized = "x".repeat(128);
        let mut reader = cursor_reader(oversized.as_bytes());
        let err = read_line_capped(&mut reader, 64).await.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("64 byte limit"), "{err}");
    }

    #[tokio::test]
    async fn read_line_capped_drains_oversized_frame_and_keeps_serving() {
        // The oversized frame is rejected, but its bytes must be fully
        // consumed so the loop keeps serving the next request.
        let payload = format!("{}\nok\n", "x".repeat(200));
        let mut reader = cursor_reader(payload.as_bytes());
        let err = read_line_capped(&mut reader, 64).await.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(
            read_line_capped(&mut reader, 64).await.unwrap(),
            Some(b"ok".to_vec())
        );
    }

    #[test]
    fn error_object_maps_kinds_and_retryability() {
        let configuration = error_object(&AiError::Configuration(
            ai_errors::ConfigurationError::new("provider", "bad config"),
        ));
        assert_eq!(configuration["kind"], "configuration");
        assert_eq!(configuration["retryable"], false);

        let rate_limit = error_object(&AiError::RateLimit(
            ai_errors::RateLimitError::new("openai", "slow down")
                .with_retry_after(Duration::from_secs(1)),
        ));
        assert_eq!(rate_limit["kind"], "rate_limit");
        assert_eq!(rate_limit["retryable"], true);

        let timeout = error_object(&AiError::Timeout(ai_errors::TimeoutError::new(
            "chat.generate",
            Duration::from_secs(5),
        )));
        assert_eq!(timeout["kind"], "timeout");

        let serialization = error_object(&AiError::Serialization(
            ai_errors::SerializationError::new("bad json"),
        ));
        assert_eq!(serialization["kind"], "serialization");
    }

    #[test]
    fn required_string_rejects_missing_empty_and_non_string() {
        let params = json!({ "good": "v", "empty": "", "number": 7 });
        assert_eq!(required_string(&params, "good").unwrap(), "v");
        for key in ["missing", "empty", "number"] {
            let err = match required_string(&params, key) {
                Ok(value) => panic!("expected error for `{key}`, got {value}"),
                Err(err) => err,
            };
            assert_eq!(err.code, INVALID_PARAMS);
            assert!(err.message.contains(key), "{}", err.message);
        }
    }

    #[test]
    fn parse_chat_request_accepts_wire_shape_and_rejects_garbage() {
        let params = json!({
            "reference": "openai/gpt-4o",
            "request": {
                "messages": [
                    { "role": "user", "parts": [ { "type": "text", "text": "hi" } ] }
                ]
            }
        });
        let request = parse_chat_request(&params).expect("wire-shaped chat request parses");
        assert_eq!(request.messages.len(), 1);

        let bad = json!({ "request": { "messages": "not-a-list" } });
        let err = expect_err(parse_chat_request(&bad));
        assert_eq!(err.code, INVALID_PARAMS);

        // Absent `request` degrades to Null and must fail validation, not panic.
        let err = expect_err(parse_chat_request(&json!({})));
        assert_eq!(err.code, INVALID_PARAMS);
    }
}
