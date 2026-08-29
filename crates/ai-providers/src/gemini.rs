//! Google Gemini native adapter: real `generateContent` wire format with
//! `x-goog-api-key` auth, function declarations, inline data images, and
//! SSE streaming (`alt=sse`).
//!
//! Requires `GOOGLE_API_KEY`. Untested live without credentials
//! (documented); unit tests cover the wire shapes.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};

use ai_core::{ChatRequest, EventStream, Model, Provider, ResponseFormat};
use ai_errors::{AiError, SerializationError};
use ai_models::{ModelCapabilities, ModelInfo};
use ai_types::{
    Completion, ContentPart, Message, ModelId, ProviderId, Role, StreamEvent, ToolCall, Usage,
};

use crate::http::{
    HttpClient, map_reqwest_error, map_response_error, parse_json, retry_after_from_headers,
};

/// Configuration for the Gemini provider.
#[derive(Debug, Clone)]
pub struct GeminiConfig {
    pub api_key: String,
    pub base_url: String,
    pub timeout: Duration,
}

impl GeminiConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            timeout: Duration::from_secs(60),
        }
    }

    pub fn from_provider_config(cfg: &ai_config::ProviderConfig) -> Result<Self, AiError> {
        let api_key = cfg.require_api_key("google")?.to_string();
        let base_url = cfg
            .base_url
            .clone()
            .unwrap_or_else(|| "https://generativelanguage.googleapis.com/v1beta".to_string());
        Ok(Self {
            api_key,
            base_url,
            timeout: Duration::from_secs(60),
        })
    }
}

/// The Gemini provider (real HTTP adapter).
pub struct GeminiProvider {
    config: GeminiConfig,
    http: HttpClient,
}

impl std::fmt::Debug for GeminiProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiProvider")
            .field("base_url", &self.config.base_url)
            .field("api_key", &"***redacted***")
            .finish()
    }
}

impl GeminiProvider {
    pub fn new(config: GeminiConfig) -> Result<Self, AiError> {
        Ok(Self {
            config,
            http: HttpClient::shared(),
        })
    }

    fn url(&self, model: &str, stream: bool) -> String {
        let method = if stream {
            "streamGenerateContent"
        } else {
            "generateContent"
        };
        let mut url = format!(
            "{}/models/{}:{}",
            self.config.base_url.trim_end_matches('/'),
            model,
            method
        );
        if stream {
            url.push_str("?alt=sse");
        }
        url
    }
}

/// Serializes a unified request into the Gemini `generateContent` format.
fn build_generate_body(request: &ChatRequest) -> Result<Value, AiError> {
    let mut contents: Vec<Value> = Vec::new();
    let mut pending_tool_results: Vec<Value> = Vec::new();

    for message in &request.messages {
        match message.role {
            Role::System => {
                // Gemini has no system role; prepend as a user part in the
                // first turn (documented convention).
                if contents.is_empty() {
                    contents.push(json!({
                        "role": "user",
                        "parts": [{"text": message.text_content()}]
                    }));
                } else if let Some(parts) = contents[0]["parts"].as_array_mut() {
                    parts.push(json!({"text": message.text_content()}));
                }
            }
            Role::User => {
                let mut parts: Vec<Value> = std::mem::take(&mut pending_tool_results);
                parts.extend(serialize_parts(message)?);
                contents.push(json!({"role": "user", "parts": parts}));
            }
            Role::Assistant => {
                let mut parts: Vec<Value> = Vec::new();
                let text = message.text_content();
                if !text.is_empty() {
                    parts.push(json!({"text": text}));
                }
                for part in &message.parts {
                    if let ContentPart::ToolCall { call } = part {
                        parts.push(json!({
                            "functionCall": {
                                "name": call.name,
                                "args": parse_json_args(&call.arguments),
                            }
                        }));
                    }
                }
                contents.push(json!({"role": "model", "parts": parts}));
            }
            Role::Tool => {
                for part in &message.parts {
                    if let ContentPart::ToolResult { result } = part {
                        pending_tool_results.push(json!({
                            "functionResponse": {
                                "name": result.name,
                                "response": {
                                    "result": parse_json_args(&result.output),
                                    "is_error": result.is_error,
                                },
                            }
                        }));
                    }
                }
            }
        }
    }

    if !pending_tool_results.is_empty() {
        contents.push(json!({"role": "user", "parts": pending_tool_results}));
    }

    let mut body = json!({
        "contents": contents,
    });
    if !request.tools.is_empty() {
        let declarations: Vec<Value> = request
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                })
            })
            .collect();
        body["tools"] = json!([{"functionDeclarations": declarations}]);
    }
    let mut generation_config = serde_json::Map::new();
    if let Some(temperature) = request.temperature {
        generation_config.insert("temperature".into(), json!(temperature));
    }
    if let Some(max_tokens) = request.max_tokens {
        generation_config.insert("maxOutputTokens".into(), json!(max_tokens));
    }
    if matches!(
        request.response_format,
        ResponseFormat::JsonObject | ResponseFormat::JsonSchema { .. }
    ) {
        generation_config.insert("responseMimeType".into(), json!("application/json"));
    }
    body["generationConfig"] = Value::Object(generation_config);
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

/// Fallback context window when a model entry omits `inputTokenLimit`.
const DEFAULT_CONTEXT_WINDOW: u64 = 1_000_000;
/// Fallback max output tokens when a model entry omits `outputTokenLimit`.
const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 8_192;

/// Upper bound on pages followed in `models.list` pagination so an endless
/// `nextPageToken` chain cannot extend the walk indefinitely.
const MAX_MODEL_PAGES: usize = 10;

/// Maps a raw Gemini `finishReason` (`STOP`, `MAX_TOKENS`, `SAFETY`, …) to
/// the unified lowercase form so downstream matching is case-stable.
fn normalize_finish_reason(reason: &str) -> String {
    reason.to_lowercase()
}

/// Synthesizes the tool-call id for the `call_index`-th call of a response,
/// unique even when several parallel calls target the same tool name.
fn synthesized_call_id(call_index: u64, name: &str) -> String {
    format!("gemini-{call_index}-{name}")
}

fn serialize_parts(message: &Message) -> Result<Vec<Value>, AiError> {
    let mut parts: Vec<Value> = Vec::new();
    let text = message.text_content();
    if !text.is_empty() {
        parts.push(json!({"text": text}));
    }
    for part in &message.parts {
        if let ContentPart::Image { image } = part {
            let inline = match image {
                ai_types::ImageSource::Url { url } => {
                    // Gemini requires inline data; URL images need
                    // fetching, which we do not do silently — the
                    // provider must pass base64 data for remote URLs.
                    return Err(AiError::Serialization(SerializationError::new(format!(
                        "Gemini requires inline image data; URL images are not supported: {url}"
                    ))));
                }
                ai_types::ImageSource::Base64 { media_type, data } => {
                    json!({
                        "mimeType": media_type,
                        "data": data,
                    })
                }
            };
            parts.push(json!({"inlineData": inline}));
        }
    }
    Ok(parts)
}

/// Parses a non-streaming `generateContent` response.
fn parse_generate_response(
    provider: &ProviderId,
    model: &ModelId,
    json: &Value,
) -> Result<Completion, AiError> {
    let candidate = json
        .get("candidates")
        .and_then(|c| c.as_array())
        .and_then(|c| c.first())
        .ok_or_else(|| {
            AiError::Serialization(SerializationError::new(
                "Gemini response missing `candidates`",
            ))
        })?;

    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut call_counter: u64 = 0;
    let mut finish_reason = candidate
        .get("finishReason")
        .and_then(|r| r.as_str())
        .map(|s| s.to_string());

    if let Some(parts) = candidate
        .pointer("/content/parts")
        .and_then(|p| p.as_array())
    {
        for part in parts {
            if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                text.push_str(t);
            }
            if let Some(call) = part.get("functionCall") {
                let name = call.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let args = call.get("args").cloned().unwrap_or(Value::Null);
                call_counter += 1;
                tool_calls.push(ToolCall {
                    id: synthesized_call_id(call_counter, name),
                    name: name.to_string(),
                    arguments: serde_json::to_string(&args).unwrap_or_else(|_| "{}".into()),
                });
            }
        }
    }
    if tool_calls.is_empty() && finish_reason.as_deref() == Some("STOP") {
        finish_reason = Some("stop".to_string());
    }

    let usage = json
        .get("usageMetadata")
        .map(|u| Usage {
            input_tokens: u
                .get("promptTokenCount")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            output_tokens: u
                .get("candidatesTokenCount")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            reasoning_tokens: None,
            cached_input_tokens: None,
            total_tokens: u.get("totalTokenCount").and_then(|v| v.as_u64()),
        })
        .unwrap_or_default();

    Ok(Completion {
        provider: provider.clone(),
        model: model.clone(),
        text,
        tool_calls,
        usage,
        reasoning: None,
        raw: json.clone(),
        finish_reason,
    })
}

/// Converts Gemini SSE chunks (`data: {...}`) into unified events.
///
/// Every chunk carries complete parts: text deltas pass through, and each
/// `functionCall` part emits Started → Delta → Completed under one
/// synthesized id (monotonic per stream, so same-name parallel calls stay
/// distinct). A `data:` line that is not valid JSON aborts the stream with a
/// serialization error instead of being skipped silently.
fn map_gemini_sse(sse: ai_stream::SseStream) -> EventStream {
    let mut call_counter: u64 = 0;
    let stream = sse.flat_map(move |sse_result| {
        let mut out: Vec<Result<StreamEvent, AiError>> = Vec::new();
        match sse_result {
            Err(e) => out.push(Err(e)),
            Ok(event) => {
                if event.data == "[DONE]" {
                    return futures::stream::iter(out);
                }
                let Ok(payload) = serde_json::from_str::<Value>(&event.data) else {
                    out.push(Err(AiError::Serialization(SerializationError::new(
                        format!(
                            "invalid Gemini SSE payload: {}",
                            event.data.get(..200).unwrap_or(event.data.as_str())
                        ),
                    ))));
                    return futures::stream::iter(out);
                };
                if let Some(candidates) = payload.get("candidates").and_then(|c| c.as_array()) {
                    if let Some(candidate) = candidates.first() {
                        if let Some(parts) = candidate
                            .pointer("/content/parts")
                            .and_then(|p| p.as_array())
                        {
                            for part in parts {
                                if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                                    if !t.is_empty() {
                                        out.push(Ok(StreamEvent::TextDelta {
                                            delta: t.to_string(),
                                        }));
                                    }
                                }
                                if let Some(call) = part.get("functionCall") {
                                    let name =
                                        call.get("name").and_then(|n| n.as_str()).unwrap_or("");
                                    let arguments = serde_json::to_string(
                                        &call.get("args").cloned().unwrap_or(Value::Null),
                                    )
                                    .unwrap_or_else(|_| "{}".into());
                                    call_counter += 1;
                                    let id = synthesized_call_id(call_counter, name);
                                    out.push(Ok(StreamEvent::ToolCallStarted {
                                        id: id.clone(),
                                        name: name.to_string(),
                                    }));
                                    out.push(Ok(StreamEvent::ToolCallDelta {
                                        id: id.clone(),
                                        arguments_delta: arguments.clone(),
                                    }));
                                    out.push(Ok(StreamEvent::ToolCallCompleted {
                                        call: ToolCall {
                                            id,
                                            name: name.to_string(),
                                            arguments,
                                        },
                                    }));
                                }
                            }
                        }
                        if let Some(reason) = candidate.get("finishReason").and_then(|r| r.as_str())
                        {
                            out.push(Ok(StreamEvent::Completed {
                                finish_reason: Some(normalize_finish_reason(reason)),
                            }));
                        }
                    }
                }
                if let Some(usage) = payload.get("usageMetadata") {
                    out.push(Ok(StreamEvent::UsageUpdate {
                        usage: Usage {
                            input_tokens: usage
                                .get("promptTokenCount")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0),
                            output_tokens: usage
                                .get("candidatesTokenCount")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0),
                            reasoning_tokens: None,
                            cached_input_tokens: None,
                            total_tokens: usage.get("totalTokenCount").and_then(|v| v.as_u64()),
                        },
                    }));
                }
            }
        }
        futures::stream::iter(out)
    });
    Box::pin(stream)
}

/// Converts one `models.list` entry into [`ModelInfo`].
///
/// Token limits map from `inputTokenLimit`/`outputTokenLimit` with defaults
/// when absent. Embedding support is claimed only for embedding methods; no
/// structured-output or vision claim is made because the listing endpoint
/// exposes no signal for either.
fn model_from_entry(entry: &Value) -> Option<ModelInfo> {
    let api_name = entry.get("name").and_then(|n| n.as_str())?;
    let id = api_name.rsplit('/').next().unwrap_or(api_name);
    let has_method = |needle: &str| {
        entry
            .get("supportedGenerationMethods")
            .and_then(|m| m.as_array())
            .is_some_and(|ms| ms.iter().any(|m| m.as_str() == Some(needle)))
    };
    let context_window = entry
        .get("inputTokenLimit")
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_CONTEXT_WINDOW);
    let max_output_tokens = entry
        .get("outputTokenLimit")
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS);
    // Vision: only claim Image when the entry advertises it — otherwise
    // text-only is the conservative truth (no hardcoded blanket Image).
    let supports_vision = entry
        .get("supportedActions")
        .and_then(|v| v.as_array())
        .is_some_and(|arr| {
            arr.iter().any(|v| {
                v.as_str().is_some_and(|s| {
                    s.to_ascii_lowercase().contains("image")
                        || s.to_ascii_lowercase().contains("vision")
                })
            })
        });
    let input_modalities = if supports_vision {
        vec![ai_types::Modality::Text, ai_types::Modality::Image]
    } else {
        vec![ai_types::Modality::Text]
    };
    Some(
        ModelInfo::new(
            ProviderId::new("google"),
            ModelId::new(id),
            context_window,
            max_output_tokens,
        )
        .with_name(id)
        .with_capabilities(ModelCapabilities {
            input_modalities,
            output_modalities: vec![ai_types::Modality::Text],
            supports_streaming: has_method("streamGenerateContent"),
            supports_tools: has_method("generateContent"),
            supports_structured_output: false,
            supports_embeddings: has_method("embedContent") || has_method("embedText"),
            supports_vision,
            supports_fine_tuning: false,
        }),
    )
}

/// Parses one `models.list` page into [`ModelInfo`]s plus the continuation
/// token (`nextPageToken`) to send as `pageToken` on the next request.
fn parse_models_page(json: &Value) -> (Vec<ModelInfo>, Option<String>) {
    let models = json
        .get("models")
        .and_then(|m| m.as_array())
        .cloned()
        .unwrap_or_default();
    let infos = models.iter().filter_map(model_from_entry).collect();
    let next = json
        .get("nextPageToken")
        .and_then(|t| t.as_str())
        .map(str::to_string);
    (infos, next)
}

/// Walks `models.list` pages through `fetch_page`, passing each page's
/// continuation token as the next request's `pageToken`, until a page has no
/// `nextPageToken` or [`MAX_MODEL_PAGES`] requests have been made.
async fn collect_model_pages<F, Fut>(mut fetch_page: F) -> Result<Vec<ModelInfo>, AiError>
where
    F: FnMut(Option<String>) -> Fut,
    Fut: std::future::Future<Output = Result<(Vec<ModelInfo>, Option<String>), AiError>>,
{
    let mut models = Vec::new();
    let mut page_token: Option<String> = None;
    for _ in 0..MAX_MODEL_PAGES {
        let (page, next) = fetch_page(page_token.take()).await?;
        models.extend(page);
        match next {
            Some(token) => page_token = Some(token),
            None => break,
        }
    }
    Ok(models)
}

#[async_trait]
impl Provider for GeminiProvider {
    fn id(&self) -> &str {
        "google"
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, AiError> {
        let http = self.http.clone();
        let base_url = self.config.base_url.trim_end_matches('/').to_string();
        let api_key = self.config.api_key.clone();
        let timeout = self.config.timeout;
        collect_model_pages(move |page_token| {
            let http = http.clone();
            let base_url = base_url.clone();
            let api_key = api_key.clone();
            async move {
                let mut request = http.get(format!("{base_url}/models"));
                if let Some(token) = page_token {
                    request = request.query(&[("pageToken", token)]);
                }
                let response = tokio::time::timeout(
                    timeout,
                    http.execute(request.header("x-goog-api-key", api_key)),
                )
                .await
                .map_err(|_| {
                    AiError::Timeout(ai_errors::TimeoutError::new("google.models", timeout))
                })?
                .map_err(|e| map_reqwest_error("google.models", e))?;

                let status = response.status();
                let retry_after = retry_after_from_headers(response.headers());
                let bytes = response
                    .bytes()
                    .await
                    .map_err(|e| map_reqwest_error("google.models", e))?
                    .to_vec();
                if !status.is_success() {
                    return Err(map_response_error("google", status, retry_after, &bytes).await);
                }
                let json: Value = parse_json("google.models", &bytes)?;
                Ok(parse_models_page(&json))
            }
        })
        .await
    }

    fn model(&self, model_id: &str) -> Result<Arc<dyn Model>, AiError> {
        let info = ModelInfo::new(
            ProviderId::new("google"),
            ModelId::new(model_id),
            1_000_000,
            8_192,
        )
        .with_name(model_id)
        .with_capabilities(ModelCapabilities {
            input_modalities: vec![ai_types::Modality::Text, ai_types::Modality::Image],
            output_modalities: vec![ai_types::Modality::Text],
            supports_streaming: true,
            supports_tools: true,
            supports_structured_output: true,
            supports_embeddings: true,
            supports_vision: true,
            supports_fine_tuning: false,
        });
        Ok(Arc::new(GeminiModel {
            provider: Arc::new(self.clone_for_model()),
            info,
        }))
    }
}

impl GeminiProvider {
    fn clone_for_model(&self) -> Self {
        Self {
            config: self.config.clone(),
            http: self.http.clone(),
        }
    }
}

/// A model behind the Gemini provider.
pub struct GeminiModel {
    provider: Arc<GeminiProvider>,
    info: ModelInfo,
}

#[async_trait]
impl Model for GeminiModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    async fn generate(&self, request: ChatRequest) -> Result<Completion, AiError> {
        let body = build_generate_body(&request)?;
        let response = tokio::time::timeout(
            self.provider.config.timeout,
            self.provider.http.execute(
                self.provider
                    .http
                    .post(self.provider.url(self.info.id.as_str(), false))
                    .header("x-goog-api-key", &self.provider.config.api_key)
                    .json(&body),
            ),
        )
        .await
        .map_err(|_| {
            AiError::Timeout(ai_errors::TimeoutError::new(
                "google.generate",
                self.provider.config.timeout,
            ))
        })?
        .map_err(|e| map_reqwest_error("google.generate", e))?;

        let status = response.status();
        let retry_after = retry_after_from_headers(response.headers());
        let bytes = response
            .bytes()
            .await
            .map_err(|e| map_reqwest_error("google.generate", e))?
            .to_vec();
        if !status.is_success() {
            return Err(map_response_error("google", status, retry_after, &bytes).await);
        }
        let json: Value = parse_json("google.generate", &bytes)?;
        parse_generate_response(&self.info.provider, &self.info.id, &json)
    }

    async fn stream(&self, request: ChatRequest) -> Result<EventStream, AiError> {
        let body = build_generate_body(&request)?;
        let response = tokio::time::timeout(
            self.provider.config.timeout,
            self.provider.http.execute(
                self.provider
                    .http
                    .post(self.provider.url(self.info.id.as_str(), true))
                    .header("x-goog-api-key", &self.provider.config.api_key)
                    .json(&body),
            ),
        )
        .await
        .map_err(|_| {
            AiError::Timeout(ai_errors::TimeoutError::new(
                "google.stream",
                self.provider.config.timeout,
            ))
        })?
        .map_err(|e| map_reqwest_error("google.stream", e))?;

        let status = response.status();
        let retry_after = retry_after_from_headers(response.headers());
        if !status.is_success() {
            let bytes = response
                .bytes()
                .await
                .map_err(|e| map_reqwest_error("google.stream", e))?
                .to_vec();
            return Err(map_response_error("google", status, retry_after, &bytes).await);
        }
        let operation = "google.stream".to_string();
        let byte_stream = response
            .bytes_stream()
            .map(move |item| item.map_err(|e| map_reqwest_error(&operation, e)));
        Ok(map_gemini_sse(ai_stream::sse_parse(byte_stream)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_messages_and_function_declarations() {
        let request = ChatRequest::new(vec![
            Message::text(Role::System, "You are helpful."),
            Message::text(Role::User, "What is 6*7?"),
            Message::new(
                Role::Assistant,
                vec![ContentPart::tool_call("gemini-calculator", "calculator", r#"{"expression":"6 * 7"}"#)],
            ),
            Message::new(
                Role::Tool,
                vec![ContentPart::tool_result("gemini-calculator", "calculator", r#"{"result":42}"#, false)],
            ),
        ])
        .with_tools(vec![ai_core::ToolDefinition::new(
            "calculator",
            "Evaluates expressions",
            json!({"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}),
        )])
        .with_temperature(0.5)
        .with_max_tokens(256);

        let body = build_generate_body(&request).unwrap();
        assert_eq!(body["contents"][0]["role"], "user");
        assert!(
            body["contents"][0]["parts"][0]["text"]
                .as_str()
                .unwrap()
                .contains("You are helpful")
        );
        // contents[0] carries the system text; [1] the user question;
        // [2] the assistant tool call; [3] the tool response.
        assert_eq!(
            body["contents"][2]["parts"][0]["functionCall"]["name"],
            "calculator"
        );
        assert_eq!(
            body["contents"][3]["parts"][0]["functionResponse"]["name"],
            "calculator"
        );
        assert_eq!(
            body["tools"][0]["functionDeclarations"][0]["name"],
            "calculator"
        );
        assert_eq!(body["generationConfig"]["temperature"], 0.5);
        assert_eq!(body["generationConfig"]["maxOutputTokens"], 256);
    }

    #[test]
    fn json_format_sets_response_mime_type() {
        let request = ChatRequest::new(vec![Message::text(Role::User, "hi")])
            .with_response_format(ResponseFormat::JsonObject);
        let body = build_generate_body(&request).unwrap();
        assert_eq!(
            body["generationConfig"]["responseMimeType"],
            "application/json"
        );
    }

    #[test]
    fn parses_generate_response_with_tool_call() {
        let json = json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "Let me compute."},
                        {"functionCall": {"name": "calculator", "args": {"expression": "6 * 7"}}}
                    ]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 7, "totalTokenCount": 16}
        });
        let completion = parse_generate_response(
            &ProviderId::new("google"),
            &ModelId::new("gemini-1.5-pro"),
            &json,
        )
        .unwrap();
        assert_eq!(completion.text, "Let me compute.");
        assert_eq!(completion.tool_calls.len(), 1);
        assert!(completion.tool_calls[0].arguments.contains("6 * 7"));
        assert_eq!(completion.finish_reason.as_deref(), Some("STOP"));
        assert_eq!(completion.usage.total(), 16);
    }

    #[tokio::test]
    async fn maps_streaming_chunks() {
        let chunks = vec![
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hel\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"lo\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":2,\"totalTokenCount\":5}}\n\n",
            "data: [DONE]\n\n",
        ];
        let input = futures::stream::iter(
            chunks
                .into_iter()
                .map(|c| Ok::<bytes::Bytes, AiError>(bytes::Bytes::from(c))),
        );
        let events: Vec<StreamEvent> = map_gemini_sse(ai_stream::sse_parse(input))
            .map(|e| e.unwrap())
            .collect()
            .await;
        assert!(matches!(events[0], StreamEvent::TextDelta { ref delta } if delta == "Hel"));
        assert!(matches!(events[1], StreamEvent::TextDelta { ref delta } if delta == "lo"));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::Completed { .. }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::UsageUpdate { .. }))
        );
    }

    #[test]
    fn url_images_are_rejected_explicitly() {
        let request = ChatRequest::new(vec![Message::new(
            Role::User,
            vec![ContentPart::image_url("https://example.com/x.png")],
        )]);
        let err = build_generate_body(&request).unwrap_err();
        assert!(err.to_string().contains("inline"), "{err}");
    }

    #[test]
    fn generate_response_ids_are_unique_across_same_name_calls() {
        let json = json!({
            "candidates": [{
                "content": {"role": "model", "parts": [
                    {"functionCall": {"name": "lookup", "args": {"q": "a"}}},
                    {"functionCall": {"name": "lookup", "args": {"q": "b"}}}
                ]}
            }]
        });
        let completion = parse_generate_response(
            &ProviderId::new("google"),
            &ModelId::new("gemini-1.5-pro"),
            &json,
        )
        .unwrap();
        assert_eq!(completion.tool_calls[0].id, "gemini-1-lookup");
        assert_eq!(completion.tool_calls[1].id, "gemini-2-lookup");
    }

    #[tokio::test]
    async fn emits_completed_with_matching_id_for_function_call() {
        let chunks = vec![
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"calculator\",\"args\":{\"expression\":\"6 * 7\"}}}]},\"finishReason\":\"STOP\"}]}\n\n",
        ];
        let input = futures::stream::iter(
            chunks
                .into_iter()
                .map(|c| Ok::<bytes::Bytes, AiError>(bytes::Bytes::from(c))),
        );
        let events: Vec<StreamEvent> = map_gemini_sse(ai_stream::sse_parse(input))
            .map(|e| e.unwrap())
            .collect()
            .await;

        assert!(
            matches!(&events[0], StreamEvent::ToolCallStarted { id, name }
                if id == "gemini-1-calculator" && name == "calculator"),
            "{events:?}"
        );
        assert!(
            matches!(&events[1], StreamEvent::ToolCallDelta { id, .. } if id == "gemini-1-calculator"),
            "{events:?}"
        );
        match &events[2] {
            StreamEvent::ToolCallCompleted { call } => {
                assert_eq!(call.id, "gemini-1-calculator");
                assert_eq!(call.name, "calculator");
                assert!(call.arguments.contains("6 * 7"));
            }
            other => panic!("expected ToolCallCompleted, got {other:?}"),
        }
        // Finish reasons are normalized to lowercase.
        assert!(
            matches!(&events[3], StreamEvent::Completed { finish_reason }
                if finish_reason.as_deref() == Some("stop")),
            "{events:?}"
        );
    }

    #[tokio::test]
    async fn parallel_same_name_streamed_calls_get_distinct_ids() {
        let chunks = vec![
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"lookup\",\"args\":{\"q\":1}}},{\"functionCall\":{\"name\":\"lookup\",\"args\":{\"q\":2}}}]}}]}\n\n",
        ];
        let input = futures::stream::iter(
            chunks
                .into_iter()
                .map(|c| Ok::<bytes::Bytes, AiError>(bytes::Bytes::from(c))),
        );
        let events: Vec<StreamEvent> = map_gemini_sse(ai_stream::sse_parse(input))
            .map(|e| e.unwrap())
            .collect()
            .await;

        let started_ids: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCallStarted { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        let completed_ids: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCallCompleted { call } => Some(call.id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(started_ids, ["gemini-1-lookup", "gemini-2-lookup"]);
        assert_eq!(completed_ids, started_ids);
    }

    #[tokio::test]
    async fn malformed_data_line_yields_serialization_error() {
        let chunks = vec![
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]}}]}\n\n",
            "data: {{{definitely not json\n\n",
        ];
        let input = futures::stream::iter(
            chunks
                .into_iter()
                .map(|c| Ok::<bytes::Bytes, AiError>(bytes::Bytes::from(c))),
        );
        let results: Vec<Result<StreamEvent, AiError>> =
            map_gemini_sse(ai_stream::sse_parse(input)).collect().await;
        assert!(results[0].is_ok());
        let err = results
            .iter()
            .filter_map(|r| r.as_ref().err())
            .next()
            .expect("malformed data line must error");
        assert!(matches!(err, AiError::Serialization(_)), "{err:?}");
        let message = err.to_string();
        assert!(message.contains("invalid Gemini SSE payload"), "{message}");
        assert!(message.contains("not json"), "{message}");
    }

    #[test]
    fn finish_reasons_normalize_to_lowercase() {
        for (raw, expected) in [
            ("STOP", "stop"),
            ("MAX_TOKENS", "max_tokens"),
            ("SAFETY", "safety"),
        ] {
            assert_eq!(normalize_finish_reason(raw), expected);
        }
    }

    #[test]
    fn parses_model_page_metadata_honestly() {
        let (models, next) = parse_models_page(&json!({
            "models": [
                {
                    "name": "models/gemini-2.0-flash",
                    "inputTokenLimit": 1_048_576,
                    "outputTokenLimit": 8_192,
                    "supportedGenerationMethods": ["generateContent", "streamGenerateContent"]
                },
                {
                    "name": "models/text-embedding-004",
                    "inputTokenLimit": 2_048,
                    "supportedGenerationMethods": ["embedContent"]
                },
                {"name": "models/minimal"}
            ],
            "nextPageToken": "cursor"
        }));
        assert_eq!(next.as_deref(), Some("cursor"));

        let flash = &models[0];
        assert_eq!(flash.context_window, 1_048_576);
        assert_eq!(flash.max_output_tokens, 8_192);
        assert!(flash.capabilities.supports_streaming);
        assert!(flash.capabilities.supports_tools);
        assert!(!flash.capabilities.supports_structured_output);
        assert!(!flash.capabilities.supports_embeddings);
        assert!(!flash.capabilities.supports_vision);

        let embedding = &models[1];
        assert_eq!(embedding.context_window, 2_048);
        assert_eq!(embedding.max_output_tokens, 8_192);
        assert!(embedding.capabilities.supports_embeddings);
        assert!(!embedding.capabilities.supports_tools);
        assert!(!embedding.capabilities.supports_structured_output);

        let minimal = &models[2];
        assert_eq!(minimal.context_window, 1_000_000);
        assert_eq!(minimal.max_output_tokens, 8_192);
        assert!(!minimal.capabilities.supports_streaming);
    }

    #[tokio::test]
    async fn model_pagination_follows_next_page_token_then_stops() {
        use std::sync::Mutex;
        let served = vec![
            json!({"models": [{"name": "models/gemini-page-a"}], "nextPageToken": "cursor-2"}),
            json!({"models": [{"name": "models/gemini-page-b"}]}),
        ];
        let requested: std::sync::Arc<Mutex<Vec<Option<String>>>> =
            std::sync::Arc::new(Mutex::new(Vec::new()));
        let requested_in_fetch = std::sync::Arc::clone(&requested);
        let fetch = move |token: Option<String>| {
            let served = served.clone();
            let requested = std::sync::Arc::clone(&requested_in_fetch);
            async move {
                let index = requested.lock().unwrap().len();
                requested.lock().unwrap().push(token);
                Ok(parse_models_page(&served[index]))
            }
        };

        let models = collect_model_pages(fetch).await.unwrap();

        assert_eq!(
            *requested.lock().unwrap(),
            vec![None, Some("cursor-2".to_string())]
        );
        let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["gemini-page-a", "gemini-page-b"]);
    }

    #[tokio::test]
    async fn model_pagination_caps_at_ten_pages() {
        use std::sync::Mutex;
        let requests: std::sync::Arc<Mutex<usize>> = std::sync::Arc::new(Mutex::new(0));
        let requests_in_fetch = std::sync::Arc::clone(&requests);
        let fetch = move |_token: Option<String>| {
            let requests = std::sync::Arc::clone(&requests_in_fetch);
            async move {
                *requests.lock().unwrap() += 1;
                Ok((
                    vec![ModelInfo::new(
                        ProviderId::new("google"),
                        ModelId::new("m"),
                        1,
                        1,
                    )],
                    Some("more".to_string()),
                ))
            }
        };

        let models = collect_model_pages(fetch).await.unwrap();

        assert_eq!(*requests.lock().unwrap(), MAX_MODEL_PAGES);
        assert_eq!(models.len(), MAX_MODEL_PAGES);
    }
}
