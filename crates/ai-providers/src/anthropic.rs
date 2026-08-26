//! Anthropic native adapter (Messages API): real wire format with
//! `x-api-key` + `anthropic-version` headers, tool use blocks, prompt
//! caching passthrough, SSE streaming via `content_block_*` events, and a
//! live paginated listing over the `/v1/models` endpoint.
//!
//! Requires `ANTHROPIC_API_KEY`. Untested live without credentials
//! (documented); unit tests cover the wire shapes.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};

use ai_core::{ChatRequest, EventStream, Model, Provider, ResponseFormat};
use ai_errors::{AiError, ProviderError, SerializationError};
use ai_models::{ModelCapabilities, ModelInfo};
use ai_types::{
    Completion, ContentPart, Message, ModelId, ProviderId, Role, StreamEvent, ToolCall, Usage,
};

use ai_stream::sse_parse;

use crate::http::{
    HttpClient, map_reqwest_error, map_response_error, parse_json, retry_after_from_headers,
};

/// The Anthropic API version header value.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";
/// The Messages API path.
pub const MESSAGES_PATH: &str = "v1/messages";
/// The Models API path (paginated model listing).
pub const MODELS_PATH: &str = "v1/models";
/// Hard cap on listing pages fetched per `list_models` call, guarding against
/// an endpoint that never clears `has_more`.
const MODEL_LISTING_MAX_PAGES: usize = 10;
/// Context window reported for every listed Anthropic model.
const MODEL_CONTEXT_WINDOW: u64 = 200_000;
/// Maximum output tokens reported for every listed Anthropic model.
const MODEL_MAX_OUTPUT_TOKENS: u64 = 8_192;

/// Configuration for the Anthropic provider.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    pub api_key: String,
    pub base_url: String,
    pub timeout: Duration,
}

impl AnthropicConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com".to_string(),
            timeout: Duration::from_secs(60),
        }
    }

    pub fn from_provider_config(cfg: &ai_config::ProviderConfig) -> Result<Self, AiError> {
        let api_key = cfg.require_api_key("anthropic")?.to_string();
        let base_url = cfg
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.anthropic.com".to_string());
        Ok(Self {
            api_key,
            base_url,
            timeout: Duration::from_secs(60),
        })
    }
}

/// The Anthropic provider (real HTTP adapter).
pub struct AnthropicProvider {
    config: AnthropicConfig,
    http: HttpClient,
}

impl std::fmt::Debug for AnthropicProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicProvider")
            .field("base_url", &self.config.base_url)
            .field("api_key", &"***redacted***")
            .finish()
    }
}

impl AnthropicProvider {
    pub fn new(config: AnthropicConfig) -> Result<Self, AiError> {
        Ok(Self {
            config,
            http: HttpClient::shared(),
        })
    }

    /// Overrides the HTTP pool, e.g. with [`HttpClient::new`] so tests get
    /// their own counters instead of sharing the process-wide one.
    pub fn with_http_client(mut self, http: HttpClient) -> Self {
        self.http = http;
        self
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.config.base_url.trim_end_matches('/'), path)
    }

    async fn request_json(&self, body: Value) -> Result<Value, AiError> {
        let response = tokio::time::timeout(
            self.config.timeout,
            self.http.execute(
                self.http
                    .post(self.url(MESSAGES_PATH))
                    .header("x-api-key", &self.config.api_key)
                    .header("anthropic-version", ANTHROPIC_VERSION)
                    .header("content-type", "application/json")
                    .json(&body),
            ),
        )
        .await
        .map_err(|_| {
            AiError::Timeout(ai_errors::TimeoutError::new(
                "anthropic.messages",
                self.config.timeout,
            ))
        })?
        .map_err(|e| map_reqwest_error("anthropic.messages", e))?;

        let status = response.status();
        let retry_after = retry_after_from_headers(response.headers());
        let bytes = response
            .bytes()
            .await
            .map_err(|e| map_reqwest_error("anthropic.messages", e))?
            .to_vec();
        if !status.is_success() {
            return Err(map_response_error("anthropic", status, retry_after, &bytes).await);
        }
        parse_json("anthropic.messages", &bytes)
    }

    async fn request_stream(&self, body: Value) -> Result<EventStream, AiError> {
        let response = tokio::time::timeout(
            self.config.timeout,
            self.http.execute(
                self.http
                    .post(self.url(MESSAGES_PATH))
                    .header("x-api-key", &self.config.api_key)
                    .header("anthropic-version", ANTHROPIC_VERSION)
                    .header("content-type", "application/json")
                    .json(&body),
            ),
        )
        .await
        .map_err(|_| {
            AiError::Timeout(ai_errors::TimeoutError::new(
                "anthropic.messages",
                self.config.timeout,
            ))
        })?
        .map_err(|e| map_reqwest_error("anthropic.messages", e))?;

        let status = response.status();
        let retry_after = retry_after_from_headers(response.headers());
        if !status.is_success() {
            let bytes = response
                .bytes()
                .await
                .map_err(|e| map_reqwest_error("anthropic.messages", e))?
                .to_vec();
            return Err(map_response_error("anthropic", status, retry_after, &bytes).await);
        }

        let operation = "anthropic.messages".to_string();
        let byte_stream = response
            .bytes_stream()
            .map(move |item| item.map_err(|e| map_reqwest_error(&operation, e)));
        Ok(map_anthropic_sse(sse_parse(byte_stream)))
    }
}

/// Serializes a unified request into the Anthropic Messages wire format.
fn build_messages_body(request: &ChatRequest, model: &str) -> Result<Value, AiError> {
    let mut system: Vec<String> = Vec::new();
    let mut messages: Vec<Value> = Vec::new();
    let mut pending_tool_results: Vec<Value> = Vec::new();

    for message in &request.messages {
        match message.role {
            Role::System => system.push(message.text_content()),
            Role::User => {
                // Flush any tool results accumulated before this user turn
                // (Anthropic requires tool_result blocks in user messages).
                let mut content: Vec<Value> = std::mem::take(&mut pending_tool_results);
                match serialize_user_content(message)? {
                    Value::Array(blocks) => content.extend(blocks),
                    other => content.push(other),
                }
                messages.push(json!({"role": "user", "content": content}));
            }
            Role::Assistant => {
                let mut blocks: Vec<Value> = Vec::new();
                let text = message.text_content();
                if !text.is_empty() {
                    blocks.push(json!({"type": "text", "text": text}));
                }
                for part in &message.parts {
                    if let ContentPart::ToolCall { call } = part {
                        blocks.push(json!({
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": parse_json_args(&call.arguments),
                        }));
                    }
                }
                messages.push(json!({"role": "assistant", "content": blocks}));
            }
            Role::Tool => {
                for part in &message.parts {
                    if let ContentPart::ToolResult { result } = part {
                        pending_tool_results.push(json!({
                            "type": "tool_result",
                            "tool_use_id": result.id,
                            "content": parse_json_args(&result.output),
                            "is_error": result.is_error,
                        }));
                    }
                }
            }
        }
    }

    // Trailing tool results (no following user message yet).
    if !pending_tool_results.is_empty() {
        messages.push(json!({"role": "user", "content": pending_tool_results}));
    }

    let mut body = json!({
        "model": model,
        "max_tokens": request.max_tokens.unwrap_or(4096),
        "messages": messages,
    });
    if !system.is_empty() {
        body["system"] = Value::Array(system.into_iter().map(Value::String).collect());
    }
    if !request.tools.is_empty() {
        let tools: Vec<Value> = request
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                })
            })
            .collect();
        body["tools"] = Value::Array(tools);
    }
    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }
    if matches!(
        request.response_format,
        ResponseFormat::JsonObject | ResponseFormat::JsonSchema { .. }
    ) {
        // Anthropic: instruct plain JSON output in the system prompt.
        let system_arr = body
            .get("system")
            .and_then(|s| s.as_array())
            .cloned()
            .unwrap_or_default();
        let mut parts = system_arr;
        parts.push(Value::String(
            "Respond with valid JSON only, no markdown fences.".to_string(),
        ));
        body["system"] = Value::Array(parts);
    }
    if let Some(extra) = request.provider_options.as_object() {
        for (key, value) in extra {
            body[key] = value.clone();
        }
    }
    Ok(body)
}

fn parse_json_args(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| Value::String(arguments.to_string()))
}

fn serialize_user_content(message: &Message) -> Result<Value, AiError> {
    let text = message.text_content();
    let mut blocks: Vec<Value> = Vec::new();
    if !text.is_empty() {
        blocks.push(json!({"type": "text", "text": text}));
    }
    for part in &message.parts {
        if let ContentPart::Image { image } = part {
            let source = match image {
                ai_types::ImageSource::Url { url } => json!({"type": "url", "url": url}),
                ai_types::ImageSource::Base64 { media_type, data } => {
                    json!({"type": "base64", "media_type": media_type, "data": data})
                }
            };
            blocks.push(json!({"type": "image", "source": source}));
        }
    }
    if blocks.is_empty() {
        return Err(AiError::Serialization(SerializationError::new(
            "user message has no serializable content",
        )));
    }
    Ok(Value::Array(blocks))
}

/// Parses a non-streaming Messages response.
fn parse_messages_response(
    provider: &ProviderId,
    model: &ModelId,
    json: &Value,
) -> Result<Completion, AiError> {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut reasoning = None;

    if let Some(content) = json.get("content").and_then(|c| c.as_array()) {
        for block in content {
            match block.get("type").and_then(|t| t.as_str()) {
                Some("text") => {
                    if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                        text.push_str(t);
                    }
                }
                Some("thinking") => {
                    if let Some(t) = block.get("thinking").and_then(|t| t.as_str()) {
                        reasoning = Some(t.to_string());
                    }
                }
                Some("tool_use") => {
                    let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let input = block.get("input").cloned().unwrap_or(Value::Null);
                    tool_calls.push(ToolCall {
                        id: id.to_string(),
                        name: name.to_string(),
                        arguments: serde_json::to_string(&input).unwrap_or_else(|_| "{}".into()),
                    });
                }
                _ => {}
            }
        }
    }

    let usage = Usage {
        input_tokens: json
            .pointer("/usage/input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        output_tokens: json
            .pointer("/usage/output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        // Anthropic does not expose reasoning tokens separately in usage.
        reasoning_tokens: None,
        cached_input_tokens: json
            .pointer("/usage/cache_read_input_tokens")
            .and_then(|v| v.as_u64()),
        total_tokens: None,
    };

    let finish_reason = json
        .get("stop_reason")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

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

/// Computes the query string for the next `/v1/models` listing page.
///
/// Returns `None` when pagination is complete (`has_more` is false), when the
/// response omits `last_id` (no cursor to continue from), or when the new
/// cursor equals the cursor already sent (`after`) — repeating it would fetch
/// the same page forever.
fn next_page_params(has_more: bool, last_id: Option<&str>, after: Option<&str>) -> Option<String> {
    if !has_more {
        return None;
    }
    let last_id = last_id?;
    let query = format!("after_id={last_id}");
    if Some(query.as_str()) == after {
        return None;
    }
    Some(query)
}

/// Converts Anthropic SSE events into unified stream events.
fn map_anthropic_sse(sse: ai_stream::SseStream) -> EventStream {
    // Tool calls being streamed, keyed by content-block index so parallel
    // tool_use blocks accumulate their arguments independently.
    let mut in_flight_tools: BTreeMap<u64, (String, String, String)> = BTreeMap::new();
    // Usage seen so far: input tokens arrive at `message_start`, output
    // tokens are updated by each `message_delta`.
    let mut input_tokens = 0_u64;
    let mut cached_input_tokens = None;
    let mut output_tokens = 0_u64;

    let stream = sse.flat_map(move |sse_result| {
        let mut out: Vec<Result<StreamEvent, AiError>> = Vec::new();
        match sse_result {
            Err(e) => out.push(Err(e)),
            Ok(event) => {
                let Ok(payload) = serde_json::from_str::<Value>(&event.data) else {
                    out.push(Err(AiError::Serialization(SerializationError::new(
                        format!(
                            "invalid Anthropic SSE payload: {}",
                            &event.data[..event.data.len().min(200)]
                        ),
                    ))));
                    return futures::stream::iter(out);
                };
                let event_type = payload.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match event_type {
                    "content_block_start" => {
                        let index = payload.get("index").and_then(|v| v.as_u64());
                        let block = payload.get("content_block").cloned().unwrap_or(Value::Null);
                        if block.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                            let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                            let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                            if let Some(index) = index {
                                in_flight_tools.insert(
                                    index,
                                    (id.to_string(), name.to_string(), String::new()),
                                );
                            }
                            out.push(Ok(StreamEvent::ToolCallStarted {
                                id: id.to_string(),
                                name: name.to_string(),
                            }));
                        }
                    }
                    "content_block_delta" => {
                        let index = payload.get("index").and_then(|v| v.as_u64());
                        let delta = payload.get("delta").cloned().unwrap_or(Value::Null);
                        match delta.get("type").and_then(|t| t.as_str()) {
                            Some("text_delta") => {
                                if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                    out.push(Ok(StreamEvent::TextDelta {
                                        delta: text.to_string(),
                                    }));
                                }
                            }
                            Some("thinking_delta") => {
                                if let Some(text) = delta.get("thinking").and_then(|t| t.as_str()) {
                                    out.push(Ok(StreamEvent::ReasoningDelta {
                                        delta: text.to_string(),
                                    }));
                                }
                            }
                            Some("input_json_delta") => {
                                if let Some(partial) =
                                    delta.get("partial_json").and_then(|t| t.as_str())
                                {
                                    if let Some((id, _name, args)) =
                                        index.and_then(|i| in_flight_tools.get_mut(&i))
                                    {
                                        args.push_str(partial);
                                        out.push(Ok(StreamEvent::ToolCallDelta {
                                            id: id.clone(),
                                            arguments_delta: partial.to_string(),
                                        }));
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    "content_block_stop" => {
                        // Finalizes the tool_use block at this index; text
                        // blocks and unknown indexes carry no state here.
                        if let Some(index) = payload.get("index").and_then(|v| v.as_u64()) {
                            if let Some((id, name, arguments)) = in_flight_tools.remove(&index) {
                                out.push(Ok(StreamEvent::ToolCallCompleted {
                                    call: ai_types::ToolCall {
                                        id,
                                        name,
                                        arguments,
                                    },
                                }));
                            }
                        }
                    }
                    "message_delta" => {
                        if let Some(output) = payload
                            .pointer("/usage/output_tokens")
                            .and_then(|v| v.as_u64())
                        {
                            output_tokens = output;
                        }
                        if let Some(reason) = payload
                            .get("delta")
                            .and_then(|d| d.get("stop_reason"))
                            .and_then(|v| v.as_str())
                        {
                            // Finalize any tool blocks the provider never
                            // closed explicitly, in block-index order.
                            for (_, (id, name, arguments)) in std::mem::take(&mut in_flight_tools) {
                                out.push(Ok(StreamEvent::ToolCallCompleted {
                                    call: ai_types::ToolCall {
                                        id,
                                        name,
                                        arguments,
                                    },
                                }));
                            }
                            out.push(Ok(StreamEvent::Completed {
                                finish_reason: Some(reason.to_string()),
                            }));
                        }
                    }
                    "message_start" => {
                        if let Some(usage) = payload.pointer("/message/usage") {
                            input_tokens = usage
                                .get("input_tokens")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            cached_input_tokens = usage
                                .get("cache_read_input_tokens")
                                .and_then(|v| v.as_u64());
                            out.push(Ok(StreamEvent::UsageUpdate {
                                usage: Usage {
                                    input_tokens,
                                    output_tokens: usage
                                        .get("output_tokens")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0),
                                    reasoning_tokens: None,
                                    cached_input_tokens,
                                    total_tokens: None,
                                },
                            }));
                        }
                    }
                    "message_stop" => {
                        out.push(Ok(StreamEvent::UsageUpdate {
                            usage: Usage {
                                input_tokens,
                                output_tokens,
                                reasoning_tokens: None,
                                cached_input_tokens,
                                total_tokens: None,
                            },
                        }));
                    }
                    "error" => {
                        let err_type = payload
                            .pointer("/error/type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown");
                        let message = payload
                            .pointer("/error/message")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown stream error");
                        out.push(Err(AiError::Provider(
                            ProviderError::new("anthropic", message).with_code(err_type),
                        )));
                    }
                    _ => {}
                }
            }
        }
        futures::stream::iter(out)
    });
    Box::pin(stream)
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn id(&self) -> &str {
        "anthropic"
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, AiError> {
        let operation = "anthropic.list_models";
        let provider = ProviderId::new("anthropic");
        let mut models = Vec::new();
        // Query string of the page just fetched; `None` for the first page.
        let mut after: Option<String> = None;

        for _ in 0..MODEL_LISTING_MAX_PAGES {
            let mut url = self.url(MODELS_PATH);
            if let Some(query) = &after {
                url.push('?');
                url.push_str(query);
            }
            let response = tokio::time::timeout(
                self.config.timeout,
                self.http.execute(
                    self.http
                        .get(url)
                        .header("x-api-key", &self.config.api_key)
                        .header("anthropic-version", ANTHROPIC_VERSION),
                ),
            )
            .await
            .map_err(|_| {
                AiError::Timeout(ai_errors::TimeoutError::new(operation, self.config.timeout))
            })?
            .map_err(|e| map_reqwest_error(operation, e))?;

            let status = response.status();
            let retry_after = retry_after_from_headers(response.headers());
            let bytes = response
                .bytes()
                .await
                .map_err(|e| map_reqwest_error(operation, e))?
                .to_vec();
            if !status.is_success() {
                return Err(map_response_error("anthropic", status, retry_after, &bytes).await);
            }
            let json: Value = parse_json(operation, &bytes)?;

            let data = json.get("data").and_then(|d| d.as_array()).ok_or_else(|| {
                AiError::Serialization(SerializationError::new("models response missing `data`"))
            })?;
            for entry in data {
                let Some(id) = entry.get("id").and_then(|v| v.as_str()) else {
                    continue;
                };
                models.push(
                    ModelInfo::new(
                        provider.clone(),
                        ModelId::new(id),
                        MODEL_CONTEXT_WINDOW,
                        MODEL_MAX_OUTPUT_TOKENS,
                    )
                    .with_name(
                        entry
                            .get("display_name")
                            .and_then(|v| v.as_str())
                            .unwrap_or(id),
                    ),
                );
            }

            after = next_page_params(
                json.get("has_more")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                json.get("last_id").and_then(|v| v.as_str()),
                after.as_deref(),
            );
            if after.is_none() {
                break;
            }
        }
        Ok(models)
    }

    fn model(&self, model_id: &str) -> Result<Arc<dyn Model>, AiError> {
        let info = ModelInfo::new(
            ProviderId::new("anthropic"),
            ModelId::new(model_id),
            200_000,
            64_000,
        )
        .with_name(model_id)
        .with_capabilities(ModelCapabilities {
            input_modalities: vec![ai_types::Modality::Text, ai_types::Modality::Image],
            output_modalities: vec![ai_types::Modality::Text],
            supports_streaming: true,
            supports_tools: true,
            supports_structured_output: false,
            supports_embeddings: false,
            supports_vision: true,
            supports_fine_tuning: false,
        });
        Ok(Arc::new(AnthropicModel {
            provider: Arc::new(self.clone_for_model()),
            info,
        }))
    }
}

impl AnthropicProvider {
    fn clone_for_model(&self) -> Self {
        Self {
            config: self.config.clone(),
            http: self.http.clone(),
        }
    }
}

/// A model behind the Anthropic provider.
pub struct AnthropicModel {
    provider: Arc<AnthropicProvider>,
    info: ModelInfo,
}

#[async_trait]
impl Model for AnthropicModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    async fn generate(&self, request: ChatRequest) -> Result<Completion, AiError> {
        let body = build_messages_body(&request, self.info.id.as_str())?;
        let json = self.provider.request_json(body).await?;
        parse_messages_response(&self.info.provider, &self.info.id, &json)
    }

    async fn stream(&self, request: ChatRequest) -> Result<EventStream, AiError> {
        let mut body = build_messages_body(&request, self.info.id.as_str())?;
        body["stream"] = json!(true);
        self.provider.request_stream(body).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn serializes_tool_calls_and_results() {
        let request = ChatRequest::new(vec![
            Message::text(Role::User, "What is 6*7?"),
            Message::new(
                Role::Assistant,
                vec![ContentPart::tool_call("call_1", "calculator", r#"{"expression":"6 * 7"}"#)],
            ),
            Message::new(
                Role::Tool,
                vec![ContentPart::tool_result("call_1", "calculator", r#"{"result":42}"#, false)],
            ),
            Message::text(Role::User, "Thanks"),
        ])
        .with_tools(vec![ai_core::ToolDefinition::new(
            "calculator",
            "Evaluates expressions",
            json!({"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}),
        )]);

        let body = build_messages_body(&request, "claude-sonnet-4").unwrap();
        assert_eq!(body["messages"].as_array().unwrap().len(), 3);
        // Tool result lands in the next user turn as a tool_result block.
        let thanks_turn = &body["messages"][2];
        assert_eq!(thanks_turn["role"], "user");
        assert_eq!(thanks_turn["content"][0]["type"], "tool_result");
        assert_eq!(thanks_turn["content"][0]["tool_use_id"], "call_1");
        assert_eq!(thanks_turn["content"][0]["content"]["result"], 42);
        assert_eq!(body["tools"][0]["name"], "calculator");
        assert_eq!(body["max_tokens"], 4096);
    }

    #[test]
    fn serializes_images_as_base64_sources() {
        let request = ChatRequest::new(vec![Message::new(
            Role::User,
            vec![
                ContentPart::text("what is this?"),
                ContentPart::Image {
                    image: ai_types::ImageSource::Base64 {
                        media_type: "image/png".into(),
                        data: "AAAA".into(),
                    },
                },
            ],
        )]);
        let body = build_messages_body(&request, "claude-sonnet-4").unwrap();
        let content = &body["messages"][0]["content"];
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[1]["source"]["type"], "base64");
        assert_eq!(content[1]["source"]["media_type"], "image/png");
    }

    #[test]
    fn parses_non_streaming_response() {
        let json = json!({
            "id": "msg_1",
            "content": [
                {"type": "text", "text": "The answer is "},
                {"type": "tool_use", "id": "call_9", "name": "calculator", "input": {"expression": "6 * 7"}},
                {"type": "text", "text": "42."}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 12, "output_tokens": 8, "cache_read_input_tokens": 4}
        });
        let completion = parse_messages_response(
            &ProviderId::new("anthropic"),
            &ModelId::new("claude-sonnet-4"),
            &json,
        )
        .unwrap();
        assert_eq!(completion.text, "The answer is 42.");
        assert_eq!(completion.tool_calls.len(), 1);
        assert_eq!(completion.tool_calls[0].name, "calculator");
        assert!(completion.tool_calls[0].arguments.contains("6 * 7"));
        assert_eq!(completion.finish_reason.as_deref(), Some("tool_use"));
        assert_eq!(completion.usage.cached_input_tokens, Some(4));
    }

    #[tokio::test]
    async fn maps_sse_events_to_unified_stream() {
        let chunks = vec![
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"calculator\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"expression\\\":\\\"6 * 7\\\"}\"}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n",
        ];
        let input = futures::stream::iter(
            chunks
                .into_iter()
                .map(|c| Ok::<bytes::Bytes, AiError>(bytes::Bytes::from(c))),
        );
        let events: Vec<StreamEvent> = map_anthropic_sse(sse_parse(input))
            .map(|e| e.unwrap())
            .collect()
            .await;

        assert!(matches!(events[0], StreamEvent::UsageUpdate { .. }));
        assert!(matches!(
            events[1],
            StreamEvent::ToolCallStarted { ref id, .. } if id == "call_1"
        ));
        assert!(matches!(events[2], StreamEvent::ToolCallDelta { .. }));
        // Finalization happens at message_delta.
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::ToolCallCompleted { .. }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::Completed { .. }))
        );
    }

    #[tokio::test]
    async fn mid_stream_error_event_surfaces_as_err() {
        let chunks = vec![
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
            "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        ];
        let input = futures::stream::iter(
            chunks
                .into_iter()
                .map(|c| Ok::<bytes::Bytes, AiError>(bytes::Bytes::from(c))),
        );
        let results: Vec<Result<StreamEvent, AiError>> =
            map_anthropic_sse(sse_parse(input)).collect().await;

        assert!(results[0].is_ok(), "events before the error stay usable");
        match &results[1] {
            Err(AiError::Provider(err)) => {
                assert_eq!(err.code.as_deref(), Some("overloaded_error"));
                assert_eq!(err.message, "Overloaded");
            }
            other => panic!("expected provider error event, got {other:?}"),
        }
        // The error aborts the stream: nothing is emitted after it.
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn interleaved_tool_blocks_complete_independently() {
        let chunks = vec![
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_a\",\"name\":\"weather\"}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_b\",\"name\":\"clock\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":9,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"IGNORED\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"tz\\\":\\\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"Paris\\\"}\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"UTC\\\"}\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":9}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":5}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        ];
        let input = futures::stream::iter(
            chunks
                .into_iter()
                .map(|c| Ok::<bytes::Bytes, AiError>(bytes::Bytes::from(c))),
        );
        let events: Vec<StreamEvent> = map_anthropic_sse(sse_parse(input))
            .map(|e| e.unwrap())
            .collect()
            .await;

        // Deltas are routed to their own block by index.
        let mut args_a = String::new();
        let mut args_b = String::new();
        let started = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::ToolCallStarted { .. }))
            .count();
        assert_eq!(started, 2);
        for e in &events {
            if let StreamEvent::ToolCallDelta {
                id,
                arguments_delta,
            } = e
            {
                if id == "call_a" {
                    args_a.push_str(arguments_delta);
                } else if id == "call_b" {
                    args_b.push_str(arguments_delta);
                }
            }
        }
        assert_eq!(args_a, r#"{"city":"Paris"}"#);
        assert_eq!(args_b, r#"{"tz":"UTC"}"#);

        // Each block completes at its own content_block_stop with its own
        // arguments; the unknown index (9) emits nothing.
        let completed: Vec<&ai_types::ToolCall> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCallCompleted { call } => Some(call),
                _ => None,
            })
            .collect();
        assert_eq!(completed.len(), 2);
        assert_eq!(completed[0].id, "call_a");
        assert_eq!(completed[0].name, "weather");
        assert_eq!(completed[0].arguments, r#"{"city":"Paris"}"#);
        assert_eq!(completed[1].id, "call_b");
        assert_eq!(completed[1].name, "clock");
        assert_eq!(completed[1].arguments, r#"{"tz":"UTC"}"#);

        // The final UsageUpdate carries message_start input tokens together
        // with the message_delta output tokens.
        match events.last() {
            Some(StreamEvent::UsageUpdate { usage }) => {
                assert_eq!(usage.input_tokens, 10);
                assert_eq!(usage.output_tokens, 5);
            }
            other => panic!("expected final usage update, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn final_usage_update_carries_message_start_and_delta_totals() {
        let chunks = vec![
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":25,\"output_tokens\":0,\"cache_read_input_tokens\":4}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":17}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        ];
        let input = futures::stream::iter(
            chunks
                .into_iter()
                .map(|c| Ok::<bytes::Bytes, AiError>(bytes::Bytes::from(c))),
        );
        let events: Vec<StreamEvent> = map_anthropic_sse(sse_parse(input))
            .map(|e| e.unwrap())
            .collect()
            .await;

        // Per-event behavior preserved: message_start still yields its own
        // UsageUpdate immediately.
        assert!(
            matches!(&events[0], StreamEvent::UsageUpdate { usage } if usage.input_tokens == 25)
        );
        assert!(matches!(&events[1], StreamEvent::TextDelta { .. }));
        // Text blocks produce no ToolCallCompleted at content_block_stop.
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, StreamEvent::ToolCallCompleted { .. }))
        );
        match events.last() {
            Some(StreamEvent::UsageUpdate { usage }) => {
                assert_eq!(usage.input_tokens, 25);
                assert_eq!(usage.output_tokens, 17);
                assert_eq!(usage.cached_input_tokens, Some(4));
            }
            other => panic!("expected final usage update, got {other:?}"),
        }
    }

    #[test]
    fn next_page_params_walks_pages_then_stops() {
        // First page carries no cursor; a page with more results continues.
        assert_eq!(
            next_page_params(true, Some("claude-b"), None).as_deref(),
            Some("after_id=claude-b")
        );
        // Completion ends pagination regardless of cursor.
        assert_eq!(
            next_page_params(false, Some("claude-c"), Some("after_id=claude-b")),
            None
        );
        // A page without last_id cannot be continued.
        assert_eq!(next_page_params(true, None, None), None);
        // Repeating the cursor just sent would never advance.
        assert_eq!(
            next_page_params(true, Some("claude-b"), Some("after_id=claude-b")),
            None
        );
    }

    /// Builds a full HTTP/1.1 response with a JSON body and `connection:
    /// close` so each listing request uses its own connection.
    fn http_response(head: &str, body: &str) -> String {
        format!(
            "{head}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    /// Spawns a minimal local HTTP server that answers each request with the
    /// next response in `responses` (repeating the last one) and records every
    /// raw request it receives.
    async fn spawn_http_server(
        responses: Vec<String>,
    ) -> (String, Arc<std::sync::Mutex<Vec<String>>>) {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let listener = tokio::net::TcpListener::from_std(listener).unwrap();
        let requests: Arc<std::sync::Mutex<Vec<String>>> = Arc::default();
        let served = Arc::new(AtomicUsize::new(0));
        let responses = Arc::new(responses);
        let requests_task = requests.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                let responses = responses.clone();
                let requests = requests_task.clone();
                let served = served.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 4096];
                    loop {
                        match socket.read(&mut chunk).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => {
                                buf.extend_from_slice(&chunk[..n]);
                                if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                                    break;
                                }
                            }
                        }
                    }
                    requests
                        .lock()
                        .unwrap()
                        .push(String::from_utf8_lossy(&buf).into_owned());
                    let index = served.fetch_add(1, Ordering::SeqCst);
                    let response = responses
                        .get(index)
                        .or_else(|| responses.last())
                        .expect("server is spawned with at least one response");
                    // A client that hung up before reading the response is
                    // irrelevant to the assertions; drop the connection.
                    let _ = socket.write_all(response.as_bytes()).await;
                });
            }
        });
        (format!("http://{addr}"), requests)
    }

    #[tokio::test]
    async fn list_models_drives_pagination_against_local_server() {
        let page1 = json!({
            "data": [
                {"id": "claude-a", "display_name": "Claude A"},
                {"id": "claude-b", "display_name": "Claude B"}
            ],
            "has_more": true,
            "last_id": "claude-b"
        })
        .to_string();
        let page2 = json!({
            "data": [{"id": "claude-c", "display_name": "Claude C"}],
            "has_more": false,
            "last_id": "claude-c"
        })
        .to_string();
        let (base_url, requests) = spawn_http_server(vec![
            http_response("HTTP/1.1 200 OK", &page1),
            http_response("HTTP/1.1 200 OK", &page2),
        ])
        .await;

        let mut config = AnthropicConfig::new("test-key");
        config.base_url = base_url;
        // Isolated pool: the shared client's request counter is process-wide
        // and other tests assert on it.
        let provider = AnthropicProvider::new(config)
            .unwrap()
            .with_http_client(HttpClient::new().unwrap());
        let models = provider.list_models().await.unwrap();

        let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["claude-a", "claude-b", "claude-c"]);
        assert_eq!(models[0].name, "Claude A");
        assert_eq!(models[2].name, "Claude C");
        assert_eq!(models[0].context_window, MODEL_CONTEXT_WINDOW);
        assert_eq!(models[0].max_output_tokens, MODEL_MAX_OUTPUT_TOKENS);

        let reqs = requests.lock().unwrap();
        assert_eq!(reqs.len(), 2);
        // First request: correct path, auth headers, no cursor.
        assert!(
            reqs[0].starts_with("GET /v1/models HTTP/1.1"),
            "{}",
            reqs[0]
        );
        assert!(reqs[0].contains("x-api-key: test-key"), "{}", reqs[0]);
        assert!(
            reqs[0].contains(&format!("anthropic-version: {ANTHROPIC_VERSION}")),
            "{}",
            reqs[0]
        );
        assert!(!reqs[0].contains("after_id"), "{}", reqs[0]);
        // Second request continues from the first page's last_id.
        assert!(
            reqs[1].starts_with("GET /v1/models?after_id=claude-b HTTP/1.1"),
            "{}",
            reqs[1]
        );
    }

    #[tokio::test]
    async fn list_models_maps_http_error_responses() {
        let body = json!({
            "type": "error",
            "error": {"type": "authentication_error", "message": "invalid x-api-key"}
        })
        .to_string();
        let (base_url, requests) =
            spawn_http_server(vec![http_response("HTTP/1.1 401 Unauthorized", &body)]).await;

        let mut config = AnthropicConfig::new("test-key");
        config.base_url = base_url;
        let provider = AnthropicProvider::new(config)
            .unwrap()
            .with_http_client(HttpClient::new().unwrap());

        let err = provider.list_models().await.unwrap_err();
        assert!(matches!(err, AiError::Authentication(_)), "{err:?}");
        // Mapping fails fast on the failed page without further requests.
        assert_eq!(requests.lock().unwrap().len(), 1);
    }
}
