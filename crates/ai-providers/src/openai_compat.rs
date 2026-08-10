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

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};

use ai_core::{ChatRequest, EventStream, Model, Provider, ResponseFormat};
use ai_errors::{AiError, SerializationError};
use ai_models::{ModelCapabilities, ModelInfo};
use ai_stream::{ToolCallAccumulator, sse_parse};
use ai_types::{
    Completion, ContentPart, Message, Modality, ModelId, ProviderId, Role, StreamEvent, ToolCall,
    Usage,
};

use crate::http::{HttpClient, map_reqwest_error, map_response_error, parse_json};

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
    /// Extra static headers (e.g. OpenRouter HTTP-Referer).
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
            timeout: Duration::from_secs(30),
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

impl OpenAiCompatProvider {
    pub fn new(config: OpenAiCompatConfig) -> Result<Self, AiError> {
        let http = HttpClient::new()?;
        Ok(Self {
            config,
            http,
            capabilities: ModelCapabilities {
                input_modalities: vec![Modality::Text, Modality::Image],
                output_modalities: vec![Modality::Text],
                supports_streaming: true,
                supports_tools: true,
                supports_structured_output: true,
                supports_embeddings: true,
                supports_vision: true,
                supports_fine_tuning: true,
            },
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.config.base_url.trim_end_matches('/'), path)
    }

    /// Executes a JSON request and returns the parsed body, mapping HTTP
    /// errors to typed [`AiError`]s.
    async fn request_json(&self, path: &str, body: Value) -> Result<Value, AiError> {
        let operation = format!("{}.{}", self.config.provider_id, path);
        let response = tokio::time::timeout(
            self.config.timeout,
            self.http
                .inner()
                .post(self.url(path))
                .bearer_auth(&self.config.api_key)
                .json(&body)
                .send(),
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
        let bytes = response
            .bytes()
            .await
            .map_err(|e| map_reqwest_error(&operation, e))?
            .to_vec();

        if !status.is_success() {
            return Err(map_response_error(&self.config.provider_id, status, &bytes).await);
        }
        parse_json(&operation, &bytes)
    }

    /// Opens a streaming request and returns the raw SSE byte stream.
    async fn request_stream(&self, path: &str, body: Value) -> Result<EventStream, AiError> {
        let operation = format!("{}.{}", self.config.provider_id, path);
        let response = tokio::time::timeout(
            self.config.timeout,
            self.http
                .inner()
                .post(self.url(path))
                .bearer_auth(&self.config.api_key)
                .json(&body)
                .send(),
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
        if !status.is_success() {
            let bytes = response
                .bytes()
                .await
                .map_err(|e| map_reqwest_error(&operation, e))?
                .to_vec();
            return Err(map_response_error(&self.config.provider_id, status, &bytes).await);
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
                        chunk_to_events(&chunk, &mut chunk_events);
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

/// Converts one streaming chunk JSON into unified events.
fn chunk_to_events(chunk: &Value, out: &mut Vec<StreamEvent>) {
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

    // Reasoning content (DeepSeek-style).
    if let Some(text) = delta.get("reasoning_content").and_then(|c| c.as_str()) {
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
            let id = tc.get("id").and_then(|i| i.as_str());
            let name = tc.pointer("/function/name").and_then(|n| n.as_str());
            let args = tc.pointer("/function/arguments").and_then(|a| a.as_str());
            let call_id = id.unwrap_or_default();

            if let (Some(id), Some(name)) = (id, name) {
                out.push(StreamEvent::ToolCallStarted {
                    id: id.to_string(),
                    name: name.to_string(),
                });
            }
            if let Some(args) = args {
                if !args.is_empty() {
                    out.push(StreamEvent::ToolCallDelta {
                        id: call_id.to_string(),
                        arguments_delta: args.to_string(),
                    });
                }
            }
            let _ = index;
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
        body["temperature"] = json!(temperature);
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
                "json_schema": {"name": name, "schema": schema}
            });
        }
    }
    if !request.stop.is_empty() {
        body["stop"] = json!(request.stop);
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
                ai_types::AudioSource::Url { url } => parts.push(json!({
                    "type": "input_audio",
                    "input_audio": {"data": url, "format": "wav"}
                })),
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
    let reasoning = message
        .get("reasoning_content")
        .and_then(|c| c.as_str())
        .map(|s| s.to_string());
    let finish_reason = choice
        .get("finish_reason")
        .and_then(|f| f.as_str())
        .map(|s| s.to_string());

    let mut tool_calls = Vec::new();
    if let Some(calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
        for call in calls {
            if let (Some(id), Some(name), Some(args)) = (
                call.get("id").and_then(|i| i.as_str()),
                call.pointer("/function/name").and_then(|n| n.as_str()),
                call.pointer("/function/arguments").and_then(|a| a.as_str()),
            ) {
                tool_calls.push(ToolCall {
                    id: id.to_string(),
                    name: name.to_string(),
                    arguments: args.to_string(),
                });
            }
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
        let response = tokio::time::timeout(
            self.config.timeout,
            self.http
                .inner()
                .get(self.url("models"))
                .bearer_auth(&self.config.api_key)
                .send(),
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
        let bytes = response
            .bytes()
            .await
            .map_err(|e| map_reqwest_error(&operation, e))?
            .to_vec();
        if !status.is_success() {
            return Err(map_response_error(&self.config.provider_id, status, &bytes).await);
        }

        let json: Value = parse_json(&operation, &bytes)?;
        let data = json.get("data").and_then(|d| d.as_array()).ok_or_else(|| {
            AiError::Serialization(SerializationError::new("models response missing `data`"))
        })?;

        let provider = ProviderId::new(self.config.provider_id.clone());
        Ok(data
            .iter()
            .filter_map(|m| {
                let id = m.get("id").and_then(|i| i.as_str())?;
                Some(
                    ModelInfo::new(provider.clone(), ModelId::new(id), 128_000, 8_192)
                        .with_name(id)
                        .with_capabilities(self.capabilities.clone()),
                )
            })
            .collect())
    }

    fn model(&self, model_id: &str) -> Result<Arc<dyn Model>, AiError> {
        let info = ModelInfo::new(
            ProviderId::new(self.config.provider_id.clone()),
            ModelId::new(model_id),
            128_000,
            8_192,
        )
        .with_name(model_id)
        .with_capabilities(self.capabilities.clone());
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
        let mut events = Vec::new();
        chunk_to_events(&chunk, &mut events);
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
        let mut events = Vec::new();
        chunk_to_events(&chunk, &mut events);
        assert!(matches!(events[0], StreamEvent::ToolCallStarted { ref id, .. } if id == "call_1"));
        assert!(
            matches!(events[1], StreamEvent::ToolCallDelta { ref arguments_delta, .. } if arguments_delta == "{\"e")
        );
    }

    #[test]
    fn chunk_to_events_ignores_empty_choices_tail() {
        let chunk = json!({"choices": [], "cost": "0"});
        let mut events = Vec::new();
        chunk_to_events(&chunk, &mut events);
        assert!(events.is_empty());
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
}
