//! OpenAI-compatible provider adapter.
//!
//! Implements the standard OpenAI Chat Completions wire protocol, which is
//! also spoken by OpenRouter, Ollama, Azure OpenAI, and many gateways
//! (including the project's `opencode.ai/zen/go/v1` gateway). Provider- and
//! model-specific extras are surfaced without breaking portability:
//!
//! - DeepSeek-style `reasoning_content` → [`StreamEvent::ReasoningDelta`] /
//!   [`Completion::reasoning`]
//! - Cache-aware usage (`prompt_cache_hit_tokens`,
//!   `completion_tokens_details.reasoning_tokens`) → [`ai_types::Usage`]
//! - Gateway `cost` field → preserved in [`Completion::raw`]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};

use ai_core::{ChatRequest, EventStream, Model, Provider, ResponseFormat};
use ai_errors::{AiError, SerializationError};
use ai_models::{ModelCapabilities, ModelInfo};
use ai_stream::{ToolCallAccumulator, sse_parse};
use ai_types::{
    Completion, ContentPart, Message, Modality, ModelId, ProviderId, Role, StreamEvent, ToolCall,
    Usage,
};

use crate::http::{
    HttpClient, map_reqwest_error, map_response_error, parse_json, retry_after_from_headers,
};

/// Well-known default base URLs per provider id.
fn default_base_url(provider: &str) -> Option<&'static str> {
    match provider {
        "openai" => Some("https://api.openai.com/v1"),
        "openrouter" => Some("https://openrouter.ai/api/v1"),
        "ollama" => Some("http://localhost:11434/v1"),
        _ => None,
    }
}

/// Configuration for an OpenAI-compatible provider.
#[derive(Debug, Clone)]
pub struct OpenAiCompatConfig {
    pub provider_id: String,
    pub api_key: String,
    pub base_url: String,
    /// Per-call timeout (defaults to 30 s when unset).
    pub timeout: Duration,
    /// Extra static headers (e.g. OpenRouter HTTP-Referer), applied to every
    /// outgoing request. Adapter-set headers (`Authorization`) win conflicts.
    pub extra_headers: Vec<(String, String)>,
}

impl OpenAiCompatConfig {
    pub fn new(
        provider_id: impl Into<String>,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            provider_id: provider_id.into(),
            api_key: api_key.into(),
            base_url: base_url.into(),
            timeout: Duration::from_secs(90),
            extra_headers: Vec::new(),
        }
    }

    /// Builds the config from provider config, applying the well-known
    /// default base URL when the provider has one and none is set.
    pub fn from_provider_config(
        provider_id: &str,
        cfg: &ai_config::ProviderConfig,
    ) -> Result<Self, AiError> {
        let api_key = cfg.require_api_key(provider_id)?.to_string();
        let base_url = cfg
            .base_url
            .clone()
            .or_else(|| default_base_url(provider_id).map(String::from))
            .ok_or_else(|| {
                AiError::Configuration(ai_errors::ConfigurationError::new(
                    "base_url",
                    format!(
                        "provider `{provider_id}` has no default base URL; \
                         configure `base_url` (e.g. AI_SDK_GATEWAY_BASE_URL)"
                    ),
                ))
            })?;
        Ok(Self::new(provider_id, api_key, base_url))
    }
}

/// An OpenAI-compatible provider.
pub struct OpenAiCompatProvider {
    config: OpenAiCompatConfig,
    http: HttpClient,
    capabilities: ModelCapabilities,
}

impl std::fmt::Debug for OpenAiCompatProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatProvider")
            .field("provider_id", &self.config.provider_id)
            .field("base_url", &self.config.base_url)
            .field("api_key", &"***redacted***")
            .finish()
    }
}

/// Reads a concept from a `/v1/models` entry through the generic scanner
/// (`ai_discovery::declared`): one synonym table shared by every crate,
/// nested-path support (`capabilities.context_window`), numeric strings, and
/// synonym-rank precedence.
fn declared_u64(entry: &Value, concept: ai_discovery::declared::Concept) -> Option<u64> {
    ai_discovery::declared::first_u64(entry, concept).map(|f| f.value)
}

/// Infers vision support from an entry's capability fields via the generic
/// concept scanner.
///
/// Checks boolean `supports_vision`/`vision`/… (any of the synonym table's
/// spellings, at any depth), feature lists for `vision`/`image` tokens, and
/// input-modality arrays for image-like values. Returns `None` when the entry
/// carries no vision signal.
fn declared_vision(entry: &Value) -> Option<bool> {
    use ai_discovery::declared::{Concept, all_strings, first_bool, has_feature};
    if let Some(f) = first_bool(entry, Concept::Vision) {
        return Some(f.value);
    }
    if let Some(f) = has_feature(entry, "vision").or_else(|| has_feature(entry, "image")) {
        if f.value {
            return Some(true);
        }
    }
    if let Some((mods, _)) = all_strings(entry, Concept::InputModalities) {
        let m = ai_discovery::modalities_from_strings(&mods);
        let has_image = m.contains(&Modality::Image);
        let has_text = m.contains(&Modality::Text);
        if has_image || has_text {
            return Some(has_image);
        }
    }
    None
}

/// Builds a [`ModelInfo`] for an OpenAI-compatible `/v1/models` entry.
///
/// Priority: curated `default_catalog` for `provider:id` → response fields
/// (`context_window`/`context_length`/… and `max_output_tokens`/…) → `0`
/// sentinel for truly unknown limits (UI shows `—` rather than a fake
/// `128k/8k`). Vision is `true` only when the catalog says so or the entry
/// explicitly advertises image modality — never blanket `true`.
fn model_info_from_entry(provider: &ProviderId, entry: &Value) -> Option<ModelInfo> {
    let id = entry.get("id").and_then(|i| i.as_str())?;
    // Name: prefer explicit display fields, fallback to id.
    let name = entry
        .get("display_name")
        .or_else(|| entry.get("name"))
        .or_else(|| entry.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or(id)
        .to_string();
    if let Some(catalog_info) = ai_models::default_catalog().get(provider.as_str(), id) {
        let mut info = catalog_info.clone();
        // Override name with the gateway's label when it differs.
        if name != id {
            info.name = name;
        }
        // If the gateway advertises fresher limits, let them win. Parsed with
        // the generic concept scanner (ai-discovery): synonym table, nested
        // paths, rank precedence — not a hand-listed key set.
        if let Some(ctx) = declared_u64(entry, ai_discovery::declared::Concept::ContextWindow) {
            info.context_window = ctx;
        }
        if let Some(max_out) = declared_u64(entry, ai_discovery::declared::Concept::MaxOutputTokens)
        {
            info.max_output_tokens = max_out;
        }
        if let Some(vision) = declared_vision(entry) {
            info.capabilities.supports_vision = vision;
            if vision
                && !info
                    .capabilities
                    .input_modalities
                    .contains(&Modality::Image)
            {
                info.capabilities.input_modalities.push(Modality::Image);
            }
            if !vision {
                info.capabilities
                    .input_modalities
                    .retain(|m| *m != Modality::Image);
                if info.capabilities.input_modalities.is_empty() {
                    info.capabilities.input_modalities.push(Modality::Text);
                }
            }
        }
        return Some(info);
    }
    // No catalog hit — parse limits from the entry or use 0 sentinel.
    let context_window =
        declared_u64(entry, ai_discovery::declared::Concept::ContextWindow).unwrap_or(0);
    let max_output_tokens =
        declared_u64(entry, ai_discovery::declared::Concept::MaxOutputTokens).unwrap_or(0);
    let supports_vision = declared_vision(entry).unwrap_or(false);
    let input_modalities = if supports_vision {
        vec![Modality::Text, Modality::Image]
    } else {
        vec![Modality::Text]
    };
    Some(
        ModelInfo::new(
            provider.clone(),
            ModelId::new(id),
            context_window,
            max_output_tokens,
        )
        .with_name(name)
        .with_capabilities(ModelCapabilities {
            input_modalities,
            output_modalities: vec![Modality::Text],
            // Zero evidence → zero claims. An unknown model must be
            // probed (ai-discovery) before any capability is asserted;
            // defaulting these to `true` reports capabilities the model
            // does not have, which is worse than reporting none.
            supports_streaming: false,
            supports_tools: false,
            supports_structured_output: false,
            supports_embeddings: false,
            supports_vision,
            supports_fine_tuning: false,
        }),
    )
}

/// Like [`model_info_from_entry`] but for on-demand handles where no
/// gateway JSON entry exists: consults the catalog only.
fn model_info_for_id(provider: &ProviderId, model_id: &str) -> ModelInfo {
    if let Some(catalog_info) = ai_models::default_catalog().get(provider.as_str(), model_id) {
        return catalog_info.clone();
    }
    // Unknown model — no fake limits or blanket vision.
    ModelInfo::new(provider.clone(), ModelId::new(model_id), 0, 0)
        .with_name(model_id)
        .with_capabilities(ModelCapabilities {
            input_modalities: vec![Modality::Text],
            output_modalities: vec![Modality::Text],
            supports_streaming: true,
            supports_tools: true,
            supports_structured_output: false,
            supports_embeddings: false,
            supports_vision: false,
            supports_fine_tuning: false,
        })
}

impl OpenAiCompatProvider {
    pub fn new(config: OpenAiCompatConfig) -> Result<Self, AiError> {
        let http = HttpClient::shared();
        Ok(Self {
            config,
            http,
            // Generic fallback for direct `model()` handles; per-model
            // `list_models` derives capabilities from catalog/entry, not
            // this blanket value. Text-only is the conservative default —
            // vision is claimed only when the catalog or entry warrants it.
            capabilities: ModelCapabilities {
                input_modalities: vec![Modality::Text],
                output_modalities: vec![Modality::Text],
                supports_streaming: true,
                supports_tools: true,
                supports_structured_output: false,
                supports_embeddings: false,
                supports_vision: false,
                supports_fine_tuning: false,
            },
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.config.base_url.trim_end_matches('/'), path)
    }

    /// Builds the headers for an outgoing request: [`OpenAiCompatConfig::
    /// extra_headers`] first, then the adapter-owned `Authorization` inserted
    /// last so it replaces any conflicting extra. Invalid names or values in
    /// `extra_headers` fail the request with a configuration error instead of
    /// being silently dropped.
    fn request_headers(&self) -> Result<HeaderMap, AiError> {
        let mut headers = HeaderMap::new();
        for (name, value) in &self.config.extra_headers {
            let name = HeaderName::from_bytes(name.as_bytes()).map_err(|_| {
                AiError::Configuration(ai_errors::ConfigurationError::new(
                    "extra_headers",
                    format!("invalid header name `{name}`"),
                ))
            })?;
            let value = HeaderValue::from_str(value).map_err(|_| {
                AiError::Configuration(ai_errors::ConfigurationError::new(
                    "extra_headers",
                    format!("invalid header value for `{name}`"),
                ))
            })?;
            headers.insert(name, value);
        }
        let auth =
            HeaderValue::from_str(&format!("Bearer {}", self.config.api_key)).map_err(|_| {
                AiError::Configuration(ai_errors::ConfigurationError::new(
                    "api_key",
                    "api key contains characters invalid in an HTTP header",
                ))
            })?;
        headers.insert(AUTHORIZATION, auth);
        Ok(headers)
    }

    /// Executes a JSON request and returns the parsed body, mapping HTTP
    /// errors to typed [`AiError`]s.
    async fn request_json(&self, path: &str, body: Value) -> Result<Value, AiError> {
        let operation = format!("{}.{}", self.config.provider_id, path);
        let url = self.url(path);
        let headers = self.request_headers()?;
        if wire_debug_enabled() {
            let body_bytes = serde_json::to_vec(&body).unwrap_or_default();
            write_wire_debug("POST", &url, &headers, &body_bytes);
        }
        let response = tokio::time::timeout(
            self.config.timeout,
            self.http
                .execute(self.http.post(url).headers(headers).json(&body)),
        )
        .await
        .map_err(|_| {
            AiError::Timeout(ai_errors::TimeoutError::new(
                &operation,
                self.config.timeout,
            ))
        })?
        .map_err(|e| map_reqwest_error(&operation, e))?;

        let status = response.status();
        let retry_after = retry_after_from_headers(response.headers());
        let bytes = response
            .bytes()
            .await
            .map_err(|e| map_reqwest_error(&operation, e))?
            .to_vec();

        if !status.is_success() {
            return Err(
                map_response_error(&self.config.provider_id, status, retry_after, &bytes).await,
            );
        }
        parse_json(&operation, &bytes)
    }

    /// Opens a streaming request and returns the raw SSE byte stream.
    async fn request_stream(&self, path: &str, body: Value) -> Result<EventStream, AiError> {
        let operation = format!("{}.{}", self.config.provider_id, path);
        let url = self.url(path);
        let headers = self.request_headers()?;
        if wire_debug_enabled() {
            let body_bytes = serde_json::to_vec(&body).unwrap_or_default();
            write_wire_debug("POST", &url, &headers, &body_bytes);
        }
        let response = tokio::time::timeout(
            self.config.timeout,
            self.http
                .execute(self.http.post(url).headers(headers).json(&body)),
        )
        .await
        .map_err(|_| {
            AiError::Timeout(ai_errors::TimeoutError::new(
                &operation,
                self.config.timeout,
            ))
        })?
        .map_err(|e| map_reqwest_error(&operation, e))?;

        let status = response.status();
        let retry_after = retry_after_from_headers(response.headers());
        if !status.is_success() {
            let bytes = response
                .bytes()
                .await
                .map_err(|e| map_reqwest_error(&operation, e))?
                .to_vec();
            return Err(
                map_response_error(&self.config.provider_id, status, retry_after, &bytes).await,
            );
        }

        // reqwest's byte stream → SSE events → unified stream events.
        let operation_stream = operation.clone();
        let byte_stream = response
            .bytes_stream()
            .map(move |item| item.map_err(|e| map_reqwest_error(&operation_stream, e)));
        let sse = sse_parse(byte_stream);
        Ok(map_sse_to_events(sse))
    }
}

/// Per-stream bookkeeping for OpenAI tool-call fragments.
///
/// The wire format announces each tool call on its first delta chunk with
/// `id` + `function.name`, then continuation chunks carry ONLY `index` +
/// `function.arguments`. This state maps a fragment's `index` back to the
/// call id (and name) announced earlier in the same stream, so argument
/// deltas stay keyed to the call that owns them. When a fragment arrives
/// with no id and no previously announced id for its index, a stable
/// placeholder (`call-{index}`) is synthesized — and reused if the name
/// shows up later — keeping [`ToolCallAccumulator`] keys consistent.
#[derive(Debug, Default)]
struct ToolCallStreamState {
    /// tool-call index → call id announced by the provider (or synthesized).
    ids: HashMap<u64, String>,
    /// tool-call index → function name seen for that slot.
    names: HashMap<u64, String>,
}

impl ToolCallStreamState {
    /// Resolves the canonical call id for a fragment at `index`.
    ///
    /// Prefers an id announced now or earlier; otherwise synthesizes
    /// `call-{index}` once and reuses it for subsequent fragments.
    fn resolve_call_id(&mut self, index: u64, announced: Option<&str>) -> String {
        if let Some(id) = announced {
            self.ids.insert(index, id.to_string());
        }
        if let Some(id) = self.ids.get(&index) {
            return id.clone();
        }
        let synthesized = format!("call-{index}");
        self.ids.insert(index, synthesized.clone());
        synthesized
    }

    /// True when `name` is the first name observed for `index` (recording it).
    fn records_first_name(&mut self, index: u64, name: &str) -> bool {
        self.names.insert(index, name.to_string()).is_none()
    }
}

/// Converts parsed SSE events into unified [`StreamEvent`]s.
///
/// Handles OpenAI streaming chunks: `delta.content` → text,
/// `delta.reasoning_content` → reasoning, `delta.tool_calls` → tool call
/// started/delta events (accumulated across chunks), and the final chunk's
/// `usage` + `finish_reason` (which also finalizes in-flight tool calls).
/// Tolerates `[DONE]` and gateway-specific trailing events (e.g.
/// `{"choices":[],"cost":"0"}`).
fn map_sse_to_events(sse: ai_stream::SseStream) -> EventStream {
    let mut accumulator = ToolCallAccumulator::new();
    let mut tool_state = ToolCallStreamState::default();
    let stream = sse.flat_map(move |sse_result| {
        let mut out: Vec<Result<StreamEvent, AiError>> = Vec::new();
        match sse_result {
            Err(e) => out.push(Err(e)),
            Ok(event) => {
                if event.data == "[DONE]" {
                    return futures::stream::iter(out);
                }
                match serde_json::from_str::<Value>(&event.data) {
                    Ok(chunk) => {
                        let mut chunk_events = Vec::new();
                        chunk_to_events(&mut tool_state, &chunk, &mut chunk_events);
                        for e in &chunk_events {
                            accumulator.push(e);
                        }
                        out.extend(chunk_events.into_iter().map(Ok));

                        // When the stream finishes, finalize any tool calls
                        // that were streamed as started/delta fragments.
                        let finished = chunk
                            .pointer("/choices/0/finish_reason")
                            .and_then(|f| f.as_str())
                            .is_some_and(|f| !f.is_empty());
                        if finished {
                            let calls = accumulator.finalize_and_drain();
                            out.extend(
                                calls
                                    .into_iter()
                                    .map(|call| Ok(StreamEvent::ToolCallCompleted { call })),
                            );
                        }
                    }
                    Err(_) => out.push(Err(AiError::Serialization(SerializationError::new(
                        format!(
                            "invalid SSE payload: {}",
                            &event.data[..event.data.len().min(200)]
                        ),
                    )))),
                }
            }
        }
        futures::stream::iter(out)
    });
    Box::pin(stream)
}

/// Converts one streaming chunk JSON into unified events, resolving
/// tool-call fragments against the per-stream index→id bookkeeping in
/// `state`.
fn chunk_to_events(state: &mut ToolCallStreamState, chunk: &Value, out: &mut Vec<StreamEvent>) {
    let choices = match chunk.get("choices").and_then(|c| c.as_array()) {
        Some(choices) => choices,
        None => return, // e.g. the gateway's trailing {"choices":[],"cost":"0"}
    };
    let Some(choice) = choices.first() else {
        return;
    };

    let delta = choice.get("delta").cloned().unwrap_or(Value::Null);

    // Content (text).
    if let Some(text) = delta.get("content").and_then(|c| c.as_str()) {
        if !text.is_empty() {
            out.push(StreamEvent::TextDelta {
                delta: text.to_string(),
            });
        }
    }

    // Reasoning content: DeepSeek-style `reasoning_content`, with the
    // OpenRouter/Nous-style `reasoning` key as fallback.
    if let Some(text) = delta
        .get("reasoning_content")
        .and_then(|c| c.as_str())
        .filter(|s| !s.is_empty())
        .or_else(|| delta.get("reasoning").and_then(|c| c.as_str()))
    {
        if !text.is_empty() {
            out.push(StreamEvent::ReasoningDelta {
                delta: text.to_string(),
            });
        }
    }

    // Tool call fragments.
    if let Some(tool_calls) = delta.get("tool_calls").and_then(|t| t.as_array()) {
        for tc in tool_calls {
            let index = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0);
            let id = tc
                .get("id")
                .and_then(|i| i.as_str())
                .filter(|s| !s.is_empty());
            let name = tc
                .pointer("/function/name")
                .and_then(|n| n.as_str())
                .filter(|s| !s.is_empty());
            let args = tc.pointer("/function/arguments").and_then(|a| a.as_str());

            // Resolve the call id for this slot: announced on this or an
            // earlier fragment, else synthesized stably from the index.
            // Continuation chunks carry only `index`, so this lookup is
            // what keeps argument deltas keyed to the right call.
            let call_id = state.resolve_call_id(index, id);

            // Emit Started exactly once per index — when the name first
            // appears — using the same resolved id as the deltas.
            if let Some(name) = name {
                if state.records_first_name(index, name) {
                    out.push(StreamEvent::ToolCallStarted {
                        id: call_id.clone(),
                        name: name.to_string(),
                    });
                }
            }
            if let Some(args) = args {
                if !args.is_empty() {
                    out.push(StreamEvent::ToolCallDelta {
                        id: call_id,
                        arguments_delta: args.to_string(),
                    });
                }
            }
        }
    }

    // Finish + usage.
    if let Some(finish) = choice.get("finish_reason").and_then(|f| f.as_str()) {
        if !finish.is_empty() && finish != "null" {
            out.push(StreamEvent::Completed {
                finish_reason: Some(finish.to_string()),
            });
        }
    }
    if let Some(usage) = chunk.get("usage") {
        if !usage.is_null() {
            out.push(StreamEvent::UsageUpdate {
                usage: parse_usage(usage),
            });
        }
    }
}

/// Parses OpenAI-style usage into [`ai_types::Usage`].
fn parse_usage(usage: &Value) -> Usage {
    Usage {
        input_tokens: usage
            .get("prompt_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        output_tokens: usage
            .get("completion_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        reasoning_tokens: usage
            .pointer("/completion_tokens_details/reasoning_tokens")
            .and_then(|v| v.as_u64()),
        cached_input_tokens: usage
            .get("prompt_cache_hit_tokens")
            .and_then(|v| v.as_u64()),
        total_tokens: usage.get("total_tokens").and_then(|v| v.as_u64()),
    }
}

/// Serializes a unified [`ChatRequest`] into the OpenAI wire body.
/// Converts an `f32` sampling parameter to a JSON number without float
/// widening artifacts: `serde_json` promotes f32 → f64, turning `0.2f32`
/// into `0.20000000298023224`, which some gateways reject with HTTP 400.
/// Routing through the shortest decimal representation of the f32 keeps
/// the wire value clean (`0.2`).
fn clean_f32(v: f32) -> Value {
    let s = v.to_string();
    match s.parse::<f64>() {
        Ok(clean) => json!(clean),
        Err(_) => json!(v),
    }
}

/// Reads `AI_SDK_DEBUG_WIRE`; `None` when unset or blank. Any non-blank
/// value selects the redacted debug record; exactly `full` opts into also
/// embedding the request body.
fn wire_debug_mode() -> Option<String> {
    std::env::var("AI_SDK_DEBUG_WIRE")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn wire_debug_enabled() -> bool {
    wire_debug_mode().is_some()
}

/// Writes a debug record for an outgoing request to
/// `<tmp>/dsh-wire-debug.json` when `AI_SDK_DEBUG_WIRE` is set.
///
/// The default mode records the method, URL, header names (credential
/// values — `Authorization` and anything containing `api-key` — replaced by
/// `***`), the body byte length, and the body's SHA-256 hex digest; message
/// content never reaches disk. `AI_SDK_DEBUG_WIRE=full` additionally embeds
/// the complete request body for offline diagnosis.
fn write_wire_debug(method: &str, url: &str, headers: &HeaderMap, body: &[u8]) {
    let Some(mode) = wire_debug_mode() else {
        return;
    };
    let mut wire_headers = serde_json::Map::new();
    for (name, value) in headers {
        let name = name.as_str();
        let value = if name.contains("authorization") || name.contains("api-key") {
            "***".to_string()
        } else {
            value
                .to_str()
                .map(str::to_string)
                .unwrap_or_else(|_| "<binary>".to_string())
        };
        wire_headers.insert(name.to_string(), Value::String(value));
    }
    let mut record = json!({
        "method": method,
        "url": url,
        "headers": wire_headers,
        "body_bytes": body.len(),
        "body_sha256": sha256_hex(body),
    });
    if mode == "full" {
        record["body"] = Value::String(String::from_utf8_lossy(body).into_owned());
    }
    let path = std::env::temp_dir().join("dsh-wire-debug.json");
    let _ = std::fs::write(&path, record.to_string());
}

/// SHA-256 message digest (FIPS 180-4), hex-encoded.
///
/// Hand-rolled because the workspace carries no hash dependency and this
/// debug record is its only consumer.
fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    const H0: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    // Pad: 0x80, zeros to 56 mod 64, then the big-endian bit length.
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut message = Vec::with_capacity(data.len() + 72);
    message.extend_from_slice(data);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    let mut state = H0;
    for chunk in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (word, bytes) in w.iter_mut().zip(chunk.chunks_exact(4)) {
            *word = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        for i in 16..w.len() {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for (&k, wi) in K.iter().zip(w.iter()) {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(k)
                .wrapping_add(*wi);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }
    state.iter().map(|word| format!("{word:08x}")).collect()
}

fn build_chat_body(request: &ChatRequest, model: &str, stream: bool) -> Result<Value, AiError> {
    let mut body = json!({
        "model": model,
        "messages": serialize_messages(&request.messages)?,
        "stream": stream,
    });

    if !request.tools.is_empty() {
        let tools: Vec<Value> = request
            .tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    }
                })
            })
            .collect();
        body["tools"] = Value::Array(tools);
    }
    if let Some(temperature) = request.temperature {
        body["temperature"] = clean_f32(temperature);
    }
    if let Some(top_p) = request.top_p {
        body["top_p"] = clean_f32(top_p);
    }
    if let Some(frequency_penalty) = request.frequency_penalty {
        body["frequency_penalty"] = clean_f32(frequency_penalty);
    }
    if let Some(presence_penalty) = request.presence_penalty {
        body["presence_penalty"] = clean_f32(presence_penalty);
    }
    if let Some(max_tokens) = request.max_tokens {
        body["max_tokens"] = json!(max_tokens);
    }
    match &request.response_format {
        ResponseFormat::Text => {}
        ResponseFormat::JsonObject => {
            body["response_format"] = json!({"type": "json_object"});
        }
        ResponseFormat::JsonSchema { schema, name } => {
            body["response_format"] = json!({
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "schema": schema,
                    "strict": true
                }
            });
        }
    }
    if !request.stop.is_empty() {
        body["stop"] = json!(request.stop);
    }
    if let Some(reasoning_effort) = request.reasoning_effort {
        // Upstream gateways (e.g. opencode/ncnio zen at 127.0.0.1:5664)
        // reject `max` with `unknown variant 'max', expected one of
        // 'low','medium','high'` — normalize Max→High rather than error,
        // preserving the \"extra thinking\" intent within the provider's
        // known range. Max is still advertised in the Models UI; the
        // normalization is a wire-level tolerance.
        let wire_effort = match reasoning_effort {
            ai_core::ReasoningEffort::Max => ai_core::ReasoningEffort::High,
            other => other,
        };
        body["reasoning_effort"] = json!(wire_effort.to_string());
    }
    if let Some(seed) = request.seed {
        body["seed"] = json!(seed);
    }
    if let Some(user) = &request.user {
        body["user"] = json!(user);
    }
    if let Some(parallel_tool_calls) = request.parallel_tool_calls {
        body["parallel_tool_calls"] = json!(parallel_tool_calls);
    }
    // Merge provider-specific options into the top-level body.
    if let Some(extra) = request.provider_options.as_object() {
        for (key, value) in extra {
            body[key] = value.clone();
        }
    }
    Ok(body)
}

/// Serializes unified messages to the OpenAI wire format.
fn serialize_messages(messages: &[Message]) -> Result<Vec<Value>, AiError> {
    let mut out = Vec::with_capacity(messages.len());

    for message in messages {
        match message.role {
            Role::Tool => {
                // Tool results: one OpenAI "tool" message per result part.
                for part in &message.parts {
                    if let ContentPart::ToolResult { result } = part {
                        out.push(json!({
                            "role": "tool",
                            "tool_call_id": result.id,
                            "content": result.output,
                        }));
                    }
                }
            }
            Role::System => out.push(json!({
                "role": "system",
                "content": message.text_content(),
            })),
            Role::Assistant => {
                let mut wire = serde_json::Map::new();
                wire.insert("role".into(), json!("assistant"));

                // Content: null when the message carries only tool calls.
                let text = message.text_content();
                let has_tool_calls = message
                    .parts
                    .iter()
                    .any(|p| matches!(p, ContentPart::ToolCall { .. }));
                if has_tool_calls {
                    wire.insert("content".into(), Value::Null);
                    let tool_calls: Vec<Value> = message
                        .parts
                        .iter()
                        .filter_map(|p| match p {
                            ContentPart::ToolCall { call } => Some(json!({
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": call.arguments,
                                }
                            })),
                            _ => None,
                        })
                        .collect();
                    wire.insert("tool_calls".into(), Value::Array(tool_calls));
                } else {
                    wire.insert("content".into(), json!(text));
                }
                out.push(Value::Object(wire));
            }
            Role::User => out.push(json!({
                "role": "user",
                "content": serialize_user_content(message)?,
            })),
        }
    }

    Ok(out)
}

/// Serializes user message content parts (text/image/audio) into the OpenAI
/// content format: a plain string for single-text parts, otherwise an array
/// of typed parts.
fn serialize_user_content(message: &Message) -> Result<Value, AiError> {
    // Fast path: single text part → plain string.
    let text_parts: Vec<&ContentPart> = message
        .parts
        .iter()
        .filter(|p| matches!(p, ContentPart::Text { .. }))
        .collect();
    let non_text_parts: Vec<&ContentPart> = message
        .parts
        .iter()
        .filter(|p| !matches!(p, ContentPart::Text { .. }))
        .collect();

    if non_text_parts.is_empty() {
        return Ok(json!(
            text_parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }

    let mut parts = Vec::new();
    for part in &message.parts {
        match part {
            ContentPart::Text { text } => parts.push(json!({"type": "text", "text": text})),
            ContentPart::Image { image } => match image {
                ai_types::ImageSource::Url { url } => parts.push(json!({
                    "type": "image_url",
                    "image_url": {"url": url}
                })),
                ai_types::ImageSource::Base64 { media_type, data } => parts.push(json!({
                    "type": "image_url",
                    "image_url": {"url": format!("data:{media_type};base64,{data}")}
                })),
            },
            ContentPart::Audio { audio } => match audio {
                // The OpenAI `input_audio` format only accepts inline base64
                // data; serializing a URL into `input_audio.data` would send
                // a broken payload, so reject it at construction time.
                ai_types::AudioSource::Url { .. } => {
                    return Err(AiError::Validation(ai_errors::ValidationError::new(
                        "audio input by URL is not supported by this adapter",
                    )));
                }
                ai_types::AudioSource::Base64 { media_type, data } => parts.push(json!({
                    "type": "input_audio",
                    "input_audio": {"data": data, "format": media_type.trim_start_matches("audio/")}
                })),
            },
            _ => {}
        }
    }
    Ok(Value::Array(parts))
}

/// Parses a non-streaming chat completion response.
fn parse_completion(
    provider: &ProviderId,
    model: &ModelId,
    json: &Value,
) -> Result<Completion, AiError> {
    let choice = json
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|c| c.first())
        .ok_or_else(|| {
            AiError::Serialization(SerializationError::new("response missing `choices`"))
        })?;

    let message = choice.get("message").cloned().unwrap_or(Value::Null);
    let text = message
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    // Reasoning: DeepSeek-style `reasoning_content`, OpenRouter/Nous-style
    // `reasoning`, then OpenRouter's structured `reasoning_details` array.
    let reasoning = message
        .get("reasoning_content")
        .and_then(|c| c.as_str())
        .filter(|s| !s.is_empty())
        .or_else(|| message.get("reasoning").and_then(|c| c.as_str()))
        .map(|s| s.to_string())
        .or_else(|| {
            message
                .get("reasoning_details")
                .and_then(|d| d.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|i| i.get("text").and_then(|t| t.as_str()))
                        .collect::<Vec<_>>()
                        .join("")
                })
        })
        .filter(|s| !s.is_empty());
    let finish_reason = choice
        .get("finish_reason")
        .and_then(|f| f.as_str())
        .map(|s| s.to_string());

    let mut tool_calls = Vec::new();
    if let Some(calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
        for (index, call) in calls.iter().enumerate() {
            let id = call
                .get("id")
                .and_then(|i| i.as_str())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    AiError::Validation(ai_errors::ValidationError::new(format!(
                        "tool_calls[{index}] missing id"
                    )))
                })?;
            let name = call
                .pointer("/function/name")
                .and_then(|n| n.as_str())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    AiError::Validation(ai_errors::ValidationError::new(format!(
                        "tool_calls[{index}] missing name"
                    )))
                })?;
            // Providers may omit `arguments` entirely or send an empty
            // string; both mean "no arguments".
            let arguments = match call.pointer("/function/arguments").and_then(|a| a.as_str()) {
                Some(args) if !args.is_empty() => args,
                _ => "{}",
            };
            tool_calls.push(ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments: arguments.to_string(),
            });
        }
    }

    let usage = json.get("usage").map(parse_usage).unwrap_or_default();

    Ok(Completion {
        provider: provider.clone(),
        model: model.clone(),
        text,
        tool_calls,
        usage,
        reasoning,
        raw: json.clone(),
        finish_reason,
    })
}

/// One model behind an OpenAI-compatible provider.
pub struct OpenAiCompatModel {
    provider: Arc<OpenAiCompatProvider>,
    info: ModelInfo,
}

impl OpenAiCompatModel {
    fn new(provider: Arc<OpenAiCompatProvider>, info: ModelInfo) -> Self {
        Self { provider, info }
    }
}

#[async_trait]
impl Model for OpenAiCompatModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    async fn generate(&self, request: ChatRequest) -> Result<Completion, AiError> {
        let body = build_chat_body(&request, self.info.id.as_str(), false)?;
        let json = self.provider.request_json("chat/completions", body).await?;
        parse_completion(&self.info.provider, &self.info.id, &json)
    }

    async fn stream(&self, request: ChatRequest) -> Result<EventStream, AiError> {
        let body = build_chat_body(&request, self.info.id.as_str(), true)?;
        self.provider.request_stream("chat/completions", body).await
    }
}

#[async_trait]
impl Provider for OpenAiCompatProvider {
    fn id(&self) -> &str {
        &self.config.provider_id
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, AiError> {
        let operation = format!("{}.list_models", self.config.provider_id);
        let url = self.url("models");
        let headers = self.request_headers()?;
        if wire_debug_enabled() {
            write_wire_debug("GET", &url, &headers, &[]);
        }
        let response = tokio::time::timeout(
            self.config.timeout,
            self.http.execute(self.http.get(url).headers(headers)),
        )
        .await
        .map_err(|_| {
            AiError::Timeout(ai_errors::TimeoutError::new(
                &operation,
                self.config.timeout,
            ))
        })?
        .map_err(|e| map_reqwest_error(&operation, e))?;

        let status = response.status();
        let retry_after = retry_after_from_headers(response.headers());
        let bytes = response
            .bytes()
            .await
            .map_err(|e| map_reqwest_error(&operation, e))?
            .to_vec();
        if !status.is_success() {
            return Err(
                map_response_error(&self.config.provider_id, status, retry_after, &bytes).await,
            );
        }

        let json: Value = parse_json(&operation, &bytes)?;
        let data = json.get("data").and_then(|d| d.as_array()).ok_or_else(|| {
            AiError::Serialization(SerializationError::new("models response missing `data`"))
        })?;

        let provider = ProviderId::new(self.config.provider_id.clone());
        Ok(data
            .iter()
            .filter_map(|m| model_info_from_entry(&provider, m))
            .collect())
    }

    fn model(&self, model_id: &str) -> Result<Arc<dyn Model>, AiError> {
        let provider = ProviderId::new(self.config.provider_id.clone());
        let info = model_info_for_id(&provider, model_id);
        Ok(Arc::new(OpenAiCompatModel::new(
            Arc::new(self.clone_for_model()),
            info,
        )))
    }
}

impl OpenAiCompatProvider {
    /// Cheap clone for sharing the provider across model handles.
    fn clone_for_model(&self) -> Self {
        Self {
            config: self.config.clone(),
            http: self.http.clone(),
            capabilities: self.capabilities.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_core::ToolDefinition;

    #[test]
    fn usage_parsing_maps_cache_and_reasoning() {
        let usage = parse_usage(&json!({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_cache_hit_tokens": 40,
            "completion_tokens_details": {"reasoning_tokens": 30}
        }));
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.cached_input_tokens, Some(40));
        assert_eq!(usage.reasoning_tokens, Some(30));
        assert_eq!(usage.total(), 150);
    }

    #[test]
    fn messages_serialize_plain_text_fast_path() {
        let messages = vec![
            Message::text(Role::System, "be concise"),
            Message::text(Role::User, "hi"),
        ];
        let wire = serialize_messages(&messages).unwrap();
        assert_eq!(wire[0]["content"], json!("be concise"));
        assert_eq!(wire[1]["content"], json!("hi"));
    }

    #[test]
    fn tool_results_serialize_as_tool_messages() {
        let messages = vec![Message::new(
            Role::Tool,
            vec![ContentPart::tool_result(
                "call_1",
                "calc",
                r#"{"result":4}"#,
                false,
            )],
        )];
        let wire = serialize_messages(&messages).unwrap();
        assert_eq!(wire[0]["role"], json!("tool"));
        assert_eq!(wire[0]["tool_call_id"], json!("call_1"));
        assert_eq!(wire[0]["content"], json!(r#"{"result":4}"#));
    }

    #[test]
    fn assistant_tool_calls_serialize_with_null_content() {
        let messages = vec![Message::new(
            Role::Assistant,
            vec![ContentPart::tool_call(
                "call_1",
                "calc",
                r#"{"expr":"2+2"}"#,
            )],
        )];
        let wire = serialize_messages(&messages).unwrap();
        assert!(wire[0]["content"].is_null());
        assert_eq!(wire[0]["tool_calls"][0]["function"]["name"], json!("calc"));
        assert_eq!(
            wire[0]["tool_calls"][0]["function"]["arguments"],
            json!(r#"{"expr":"2+2"}"#)
        );
    }

    #[test]
    fn vision_parts_serialize_as_image_url() {
        let messages = vec![Message::new(
            Role::User,
            vec![
                ContentPart::text("what is this?"),
                ContentPart::image_url("https://example.com/x.png"),
            ],
        )];
        let wire = serialize_messages(&messages).unwrap();
        let content = &wire[0]["content"];
        assert!(content.is_array());
        assert_eq!(content[0]["type"], json!("text"));
        assert_eq!(content[1]["type"], json!("image_url"));
        assert_eq!(
            content[1]["image_url"]["url"],
            json!("https://example.com/x.png")
        );
    }

    #[test]
    fn build_body_includes_tools_and_response_format() {
        let request = ChatRequest::new(vec![Message::text(Role::User, "hi")])
            .with_tools(vec![ToolDefinition::new(
                "calc",
                "calculates",
                json!({"type": "object"}),
            )])
            .with_response_format(ResponseFormat::JsonObject)
            .with_max_tokens(64);
        let body = build_chat_body(&request, "deepseek-v4-flash", false).unwrap();
        assert_eq!(body["model"], json!("deepseek-v4-flash"));
        assert_eq!(body["stream"], json!(false));
        assert_eq!(body["tools"][0]["type"], json!("function"));
        assert_eq!(body["tools"][0]["function"]["name"], json!("calc"));
        assert_eq!(body["response_format"]["type"], json!("json_object"));
        assert_eq!(body["max_tokens"], json!(64));
    }

    #[test]
    fn chunk_to_events_maps_reasoning_and_text() {
        let chunk = json!({
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello", "reasoning_content": "thinking"},
                "finish_reason": null
            }],
            "usage": null
        });
        let mut state = ToolCallStreamState::default();
        let mut events = Vec::new();
        chunk_to_events(&mut state, &chunk, &mut events);
        assert!(matches!(events[0], StreamEvent::TextDelta { ref delta } if delta == "Hello"));
        assert!(
            matches!(events[1], StreamEvent::ReasoningDelta { ref delta } if delta == "thinking")
        );
    }

    #[test]
    fn chunk_to_events_maps_tool_call_fragments() {
        let chunk = json!({
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "{\"e"}
                    }]
                },
                "finish_reason": null
            }],
            "usage": null
        });
        let mut state = ToolCallStreamState::default();
        let mut events = Vec::new();
        chunk_to_events(&mut state, &chunk, &mut events);
        assert!(matches!(events[0], StreamEvent::ToolCallStarted { ref id, .. } if id == "call_1"));
        assert!(
            matches!(events[1], StreamEvent::ToolCallDelta { ref arguments_delta, .. } if arguments_delta == "{\"e")
        );
    }

    #[test]
    fn chunk_to_events_ignores_empty_choices_tail() {
        let chunk = json!({"choices": [], "cost": "0"});
        let mut state = ToolCallStreamState::default();
        let mut events = Vec::new();
        chunk_to_events(&mut state, &chunk, &mut events);
        assert!(events.is_empty());
    }

    /// Drives SSE `data:` payloads through the adapter's full mapping path
    /// (SSE parse → chunk → unified events), like a real streamed response.
    async fn map_sse_payloads(chunks: &[serde_json::Value]) -> Vec<StreamEvent> {
        use bytes::Bytes;
        let body: String = chunks
            .iter()
            .map(|c| format!("data: {c}\n\n"))
            .collect::<Vec<_>>()
            .join("");
        let input: Vec<Result<Bytes, AiError>> = vec![Ok(Bytes::from(body))];
        map_sse_to_events(ai_stream::sse_parse(futures::stream::iter(input)))
            .map(|event| event.expect("stream event"))
            .collect()
            .await
    }

    #[test]
    fn continuation_fragments_resolve_id_by_index() {
        // First fragment announces id + name; continuations carry only the
        // index. Deltas must stay keyed to the announced call id.
        let mut state = ToolCallStreamState::default();
        let started = json!({
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call_abc", "type": "function",
                 "function": {"name": "calc", "arguments": ""}}
            ]}, "finish_reason": null}]
        });
        let continuation = json!({
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "{\"expr\""}}
            ]}, "finish_reason": null}]
        });

        let mut events = Vec::new();
        chunk_to_events(&mut state, &started, &mut events);
        assert!(
            matches!(&events[0], StreamEvent::ToolCallStarted { id, name } if id == "call_abc" && name == "calc")
        );

        chunk_to_events(&mut state, &continuation, &mut events);
        assert_eq!(events.len(), 2);
        assert!(
            matches!(&events[1], StreamEvent::ToolCallDelta { id, arguments_delta }
            if id == "call_abc" && arguments_delta == "{\"expr\"")
        );
    }

    #[test]
    fn repeated_name_fragments_emit_started_only_once_per_index() {
        let chunk_with_name = json!({
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call_x", "type": "function",
                 "function": {"name": "calc", "arguments": "{}"}}
            ]}, "finish_reason": null}]
        });
        // Some providers repeat id + name on every fragment; shared stream
        // state must emit Started exactly once for the slot.
        let mut state = ToolCallStreamState::default();
        let mut all = Vec::new();
        chunk_to_events(&mut state, &chunk_with_name, &mut all);
        chunk_to_events(&mut state, &chunk_with_name, &mut all);
        chunk_to_events(&mut state, &chunk_with_name, &mut all);
        assert_eq!(
            all.iter()
                .filter(|e| matches!(e, StreamEvent::ToolCallStarted { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn unannounced_index_gets_stable_synthesized_call_id() {
        // A delta with neither id nor a previously announced index gets a
        // stable placeholder, reused when the name arrives later.
        let args_first = json!({
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 2, "function": {"arguments": "{}"}}
            ]}, "finish_reason": null}]
        });
        let mut state = ToolCallStreamState::default();
        let mut events = Vec::new();
        chunk_to_events(&mut state, &args_first, &mut events);
        assert!(matches!(&events[0], StreamEvent::ToolCallDelta { id, .. } if id == "call-2"));

        let name_after = json!({
            "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 2, "id": "", "type": "function",
                 "function": {"name": "late", "arguments": ""}}
            ]}, "finish_reason": null}]
        });
        chunk_to_events(&mut state, &name_after, &mut events);
        assert!(
            matches!(&events[1], StreamEvent::ToolCallStarted { id, name }
            if id == "call-2" && name == "late")
        );
    }

    #[tokio::test]
    async fn streamed_tool_call_fragments_survive_multi_chunk_streams() {
        // Regression test for streamed argument loss: one Started chunk
        // (id + function.name), then ≥3 args-only chunks carrying just the
        // index — every fragment must survive into the finalized call.
        let chunks = vec![
            json!({"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[
                {"index":0,"id":"call_abc","type":"function","function":{"name":"calc","arguments":""}}
            ]},"finish_reason":null}],"usage":null}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[
                {"index":0,"function":{"arguments":"{\"expr\":"}}
            ]},"finish_reason":null}],"usage":null}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[
                {"index":0,"function":{"arguments":"\"2+2\","}}
            ]},"finish_reason":null}],"usage":null}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[
                {"index":0,"function":{"arguments":"\"precision\":2}"}}
            ]},"finish_reason":null}],"usage":null}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[
                {"index":0,"function":{"arguments":""}}
            ]},"finish_reason":null}],"usage":null}),
            json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],
                   "usage":{"prompt_tokens":9,"completion_tokens":4}}),
        ];
        let events = map_sse_payloads(&chunks).await;

        // Exactly one Started for the slot, keyed by the announced id.
        let started: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCallStarted { id, name } => Some((id.as_str(), name.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(started, vec![("call_abc", "calc")]);

        // Every argument delta is keyed to that same id.
        let deltas: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCallDelta { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(deltas.len(), 3);
        assert!(deltas.iter().all(|id| *id == "call_abc"));

        // Downstream assembly yields the FULL concatenated arguments.
        let mut acc = ToolCallAccumulator::new();
        for event in &events {
            acc.push(event);
        }
        acc.finalize();
        let calls = acc.completed();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_abc");
        assert_eq!(calls[0].name, "calc");
        assert_eq!(calls[0].arguments, r#"{"expr":"2+2","precision":2}"#);
    }

    #[tokio::test]
    async fn interleaved_parallel_tool_calls_do_not_cross_contaminate() {
        // Two calls interleaved by index: fragments of index 1 and index 2
        // alternate, and each call must assemble only its own fragments.
        let chunks = vec![
            json!({"choices":[{"index":0,"delta":{"tool_calls":[
                {"index":1,"id":"call_a","type":"function","function":{"name":"alpha","arguments":""}}
            ]},"finish_reason":null}],"usage":null}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[
                {"index":2,"id":"call_b","type":"function","function":{"name":"beta","arguments":""}}
            ]},"finish_reason":null}],"usage":null}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[
                {"index":2,"function":{"arguments":"[1"}}
            ]},"finish_reason":null}],"usage":null}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[
                {"index":1,"function":{"arguments":"{\"x\":"}}
            ]},"finish_reason":null}],"usage":null}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[
                {"index":2,"function":{"arguments":",2]"}}
            ]},"finish_reason":null}],"usage":null}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[
                {"index":1,"function":{"arguments":"1}"}}
            ]},"finish_reason":null}],"usage":null}),
            json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":null}),
        ];
        let events = map_sse_payloads(&chunks).await;

        let mut acc = ToolCallAccumulator::new();
        for event in &events {
            acc.push(event);
        }
        acc.finalize();

        // The adapter's finalize order follows HashMap drain order, so key
        // by id instead of position.
        let by_id: HashMap<&str, &ToolCall> =
            acc.completed().iter().map(|c| (c.id.as_str(), c)).collect();
        assert_eq!(by_id.len(), 2);
        let a = by_id.get("call_a").expect("call_a assembled");
        assert_eq!(a.name, "alpha");
        assert_eq!(a.arguments, r#"{"x":1}"#);
        let b = by_id.get("call_b").expect("call_b assembled");
        assert_eq!(b.name, "beta");
        assert_eq!(b.arguments, "[1,2]");
    }

    #[test]
    fn build_body_float_params_have_no_widening_artifacts() {
        let request = ChatRequest::new(vec![Message::text(Role::User, "hi")])
            .with_temperature(0.2)
            .with_top_p(0.9)
            .with_frequency_penalty(0.5)
            .with_presence_penalty(-0.25);
        let body = build_chat_body(&request, "m", false).unwrap();
        let raw = body.to_string();
        assert!(raw.contains("\"temperature\":0.2"), "{raw}");
        assert!(!raw.contains("0.20000000298"), "{raw}");
        assert!(raw.contains("\"top_p\":0.9"), "{raw}");
        assert!(raw.contains("\"frequency_penalty\":0.5"), "{raw}");
        assert!(raw.contains("\"presence_penalty\":-0.25"), "{raw}");
    }

    #[test]
    fn parse_completion_includes_reasoning_and_tools() {
        let response = json!({
            "id": "x",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "thought",
                    "tool_calls": [{
                        "id": "call_9",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "{\"expr\":\"2+2\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let completion = parse_completion(
            &ProviderId::new("opencode"),
            &ModelId::new("deepseek-v4-flash"),
            &response,
        )
        .unwrap();
        assert_eq!(completion.text, "answer");
        assert_eq!(completion.reasoning.as_deref(), Some("thought"));
        assert_eq!(completion.tool_calls.len(), 1);
        assert_eq!(completion.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(completion.usage.total(), 15);
    }

    #[test]
    fn parse_completion_accepts_openrouter_style_reasoning() {
        // OpenRouter/Nous-style: `reasoning` string + `reasoning_details`
        // array instead of DeepSeek's `reasoning_content`.
        let response = json!({
            "id": "x",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "391",
                    "reasoning": "17*23 = 391",
                    "reasoning_details": [{"type": "text", "text": "17*23 = 391"}]
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}
        });
        let completion = parse_completion(
            &ProviderId::new("opencode"),
            &ModelId::new("stealth/ox-alpha"),
            &response,
        )
        .unwrap();
        assert_eq!(completion.reasoning.as_deref(), Some("17*23 = 391"));
    }

    #[test]
    fn parse_completion_reasoning_details_fallback_when_no_string_field() {
        let response = json!({
            "id": "x",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "done",
                    "reasoning_details": [
                        {"type": "text", "text": "step one. "},
                        {"type": "text", "text": "step two."}
                    ]
                },
                "finish_reason": "stop"
            }]
        });
        let completion = parse_completion(
            &ProviderId::new("opencode"),
            &ModelId::new("stealth/ox-alpha"),
            &response,
        )
        .unwrap();
        assert_eq!(completion.reasoning.as_deref(), Some("step one. step two."));
    }

    #[test]
    fn chunk_to_events_maps_openrouter_style_reasoning_delta() {
        let chunk = json!({
            "choices": [{"delta": {"content": "", "reasoning": "thinking..."}}]
        });
        let mut state = ToolCallStreamState::default();
        let mut events = Vec::new();
        chunk_to_events(&mut state, &chunk, &mut events);
        assert!(
            events.iter().any(|e| matches!(
                e,
                StreamEvent::ReasoningDelta { delta } if delta == "thinking..."
            )),
            "expected ReasoningDelta from `reasoning` key: {events:?}"
        );
    }

    #[test]
    fn sha256_matches_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    /// Serializes tests that touch `AI_SDK_DEBUG_WIRE` / the shared
    /// `<tmp>/dsh-wire-debug.json` record.
    static WIRE_DEBUG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn wire_debug_record_is_redacted_unless_full_is_requested() {
        let _guard = WIRE_DEBUG_LOCK.lock().unwrap();
        const KEY: &str = "AI_SDK_DEBUG_WIRE";
        let secret_key = "sk-super-secret-wire-key";
        let secret_message = "confidential-user-message-content";
        let body = format!(r#"{{"messages":[{{"content":"{secret_message}"}}]}}"#);
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {secret_key}")).unwrap(),
        );
        headers.insert("x-api-key", HeaderValue::from_str("api-key-12345").unwrap());
        headers.insert("x-trace-id", HeaderValue::from_str("trace-open").unwrap());

        let path = std::env::temp_dir().join("dsh-wire-debug.json");

        // SAFETY (test-only): single-threaded mutation of this variable under
        // WIRE_DEBUG_LOCK, removed afterwards so other tests are unaffected.
        unsafe { std::env::set_var(KEY, "1") };
        write_wire_debug(
            "POST",
            "https://gateway.example/v1/chat/completions",
            &headers,
            body.as_bytes(),
        );
        let written = std::fs::read_to_string(&path).expect("redacted record written");
        assert!(written.contains(r#""method":"POST""#), "{written}");
        assert!(
            written.contains("gateway.example/v1/chat/completions"),
            "{written}"
        );
        assert!(written.contains(r#""authorization":"***""#), "{written}");
        assert!(written.contains(r#""x-api-key":"***""#), "{written}");
        assert!(
            written.contains(r#""x-trace-id":"trace-open""#),
            "non-credential headers keep their value: {written}"
        );
        assert!(
            written.contains(&format!(r#""body_bytes":{}"#, body.len())),
            "{written}"
        );
        assert!(
            written.contains(&format!(
                r#""body_sha256":"{}""#,
                sha256_hex(body.as_bytes())
            )),
            "{written}"
        );
        // Neither the key nor any message text may leak into the record.
        assert!(!written.contains(secret_key), "{written}");
        assert!(!written.contains("api-key-12345"), "{written}");
        assert!(!written.contains(secret_message), "{written}");

        // `full` opts into embedding the body.
        unsafe { std::env::set_var(KEY, "full") };
        write_wire_debug(
            "POST",
            "https://gateway.example/v1/chat/completions",
            &headers,
            body.as_bytes(),
        );
        let written = std::fs::read_to_string(&path).expect("full record written");
        assert!(
            written.contains(secret_message),
            "full mode embeds the body: {written}"
        );

        unsafe { std::env::remove_var(KEY) };
    }

    /// Spawns a minimal HTTP/1.1 server that records each request's raw head
    /// and answers `/models` with a model list, streaming chat requests with
    /// SSE, and everything else with a plain completion. Every response uses
    /// `connection: close`, so each request arrives on its own connection.
    fn spawn_capture_server() -> (String, Arc<std::sync::Mutex<Vec<String>>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        fn find_head_end(buf: &[u8]) -> Option<usize> {
            buf.windows(4).position(|w| w == b"\r\n\r\n")
        }

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let listener = tokio::net::TcpListener::from_std(listener).unwrap();
        let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured_task = captured.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                let captured_conn = captured_task.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 4096];
                    let head_end = loop {
                        let n = socket.read(&mut chunk).await.unwrap_or(0);
                        if n == 0 {
                            return;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                        if let Some(pos) = find_head_end(&buf) {
                            break pos;
                        }
                    };
                    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
                    let content_length = head
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            if name.trim().eq_ignore_ascii_case("content-length") {
                                value.trim().parse::<usize>().ok()
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0);
                    while buf.len() < head_end + 4 + content_length {
                        let n = socket.read(&mut chunk).await.unwrap_or(0);
                        if n == 0 {
                            break;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                    }
                    let is_get = head.starts_with("GET");
                    captured_conn.lock().unwrap().push(head);

                    let body_text = String::from_utf8_lossy(&buf[head_end + 4..]).to_string();
                    let payload = if is_get {
                        r#"{"data":[{"id":"mock-model"}]}"#.to_string()
                    } else if body_text.contains(r#""stream":true"#) {
                        format!(
                            "data: {}\n\ndata: [DONE]\n\n",
                            r#"{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}"#
                        )
                    } else {
                        r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#
                            .to_string()
                    };
                    let response = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
                        payload.len()
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                });
            }
        });
        (format!("http://{addr}"), captured)
    }

    #[tokio::test]
    // Test-only: the guard serializes against `wire_debug_record_is_redacted…`
    // over the shared `<tmp>/dsh-wire-debug.json` record and its env var.
    #[allow(clippy::await_holding_lock)]
    async fn extra_headers_reach_the_wire_and_authorization_wins() {
        let _guard = WIRE_DEBUG_LOCK.lock().unwrap();
        let (base_url, captured) = spawn_capture_server();

        let mut config = OpenAiCompatConfig::new("mock", "sk-live-secret-key", base_url);
        config.extra_headers = vec![
            ("X-Custom-Trace".to_string(), "trace-42".to_string()),
            ("authorization".to_string(), "Bearer attacker".to_string()),
        ];
        let provider = OpenAiCompatProvider::new(config).expect("provider builds");
        let model = provider.model("mock-model").expect("model resolves");

        // GET /models via list_models.
        let models = provider.list_models().await.expect("models listed");
        assert_eq!(models.len(), 1);

        // POST /chat/completions via generate.
        let completion = model
            .generate(ChatRequest::new(vec![Message::text(Role::User, "hi")]))
            .await
            .expect("generate succeeds");
        assert_eq!(completion.text, "hi");

        // POST /chat/completions with stream:true via stream.
        let events: Vec<_> = model
            .stream(ChatRequest::new(vec![Message::text(Role::User, "hi")]))
            .await
            .expect("stream opens")
            .collect()
            .await;
        assert!(
            events.iter().all(Result::is_ok),
            "stream errors: {events:?}"
        );

        // Every outgoing request carried the custom header, exactly one
        // Authorization header, and the adapter-owned credential won.
        let heads = captured.lock().unwrap();
        assert_eq!(heads.len(), 3, "all three requests seen: {heads:?}");
        for head in heads.iter() {
            let lower = head.to_ascii_lowercase();
            assert!(
                lower.contains("x-custom-trace: trace-42"),
                "custom header missing from request:\n{head}"
            );
            let auth_lines: Vec<_> = lower
                .lines()
                .filter(|line| line.starts_with("authorization:"))
                .collect();
            assert_eq!(
                auth_lines,
                vec!["authorization: bearer sk-live-secret-key"],
                "adapter-owned authorization must win over extra_headers:\n{head}"
            );
        }
    }

    #[test]
    fn parse_completion_defaults_missing_arguments_to_empty_object() {
        let response = json!({
            "choices": [{
                "message": {
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "no_args"}},
                        {"id": "call_2", "function": {"name": "empty_args", "arguments": ""}}
                    ]
                }
            }]
        });
        let completion =
            parse_completion(&ProviderId::new("opencode"), &ModelId::new("m"), &response)
                .expect("missing/empty arguments are valid");
        let args: Vec<&str> = completion
            .tool_calls
            .iter()
            .map(|c| c.arguments.as_str())
            .collect();
        assert_eq!(args, vec!["{}", "{}"]);
    }

    #[test]
    fn parse_completion_rejects_tool_call_missing_id_or_name() {
        let missing_id = json!({
            "choices": [{
                "message": {"tool_calls": [
                    {"id": "call_ok", "function": {"name": "calc", "arguments": "{}"}},
                    {"function": {"name": "broken"}}
                ]}
            }]
        });
        let err = parse_completion(
            &ProviderId::new("opencode"),
            &ModelId::new("m"),
            &missing_id,
        )
        .expect_err("missing id must fail loudly");
        assert!(matches!(err, AiError::Validation(_)), "{err:?}");
        assert!(
            err.to_string().contains("tool_calls[1] missing id"),
            "{err}"
        );

        let missing_name = json!({
            "choices": [{
                "message": {"tool_calls": [
                    {"id": "call_broken", "function": {"arguments": "{}"}}
                ]}
            }]
        });
        let err = parse_completion(
            &ProviderId::new("opencode"),
            &ModelId::new("m"),
            &missing_name,
        )
        .expect_err("missing name must fail loudly");
        assert!(matches!(err, AiError::Validation(_)), "{err:?}");
        assert!(
            err.to_string().contains("tool_calls[0] missing name"),
            "{err}"
        );
    }

    #[test]
    fn user_audio_url_part_is_rejected_at_construction() {
        let messages = vec![Message::new(
            Role::User,
            vec![
                ContentPart::text("listen:"),
                ContentPart::Audio {
                    audio: ai_types::AudioSource::Url {
                        url: "https://example.com/clip.wav".to_string(),
                    },
                },
            ],
        )];
        let err = serialize_messages(&messages).expect_err("audio URL must be rejected");
        assert!(matches!(err, AiError::Validation(_)), "{err:?}");
        assert!(
            err.to_string()
                .contains("audio input by URL is not supported by this adapter"),
            "{err}"
        );
    }
}
