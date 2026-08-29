//! Empirical capability probing.
//!
//! Declared metadata is frequently absent (b.ai, NVIDIA publish none) and
//! sometimes *wrong* (SenseNova declares vision and tools that both fail).
//! Probing is therefore the primary source of truth: send a real request,
//! observe the outcome.
//!
//! Every probe is expressed without reference to any provider or model name.
//! The thinking-toggle probe in particular enumerates the *spellings* that
//! exist in the ecosystem and determines empirically which one this model
//! honours — by checking whether the reasoning field actually disappears.

use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};

use crate::errors::{ClassifiedError, ErrorClass, LimitKind, classify, mine_limits};
use crate::response::{NormalizedMessage, NormalizedUsage, normalize_message, normalize_usage};

/// A raw outcome from an HTTP call, before interpretation.
#[derive(Debug, Clone)]
pub struct RawResponse {
    /// HTTP status (`0` = never completed).
    pub status: u16,
    /// Response body, verbatim.
    pub body: String,
    /// Wall time for the call.
    pub elapsed: Duration,
    /// Transport-level error, if the request never completed.
    pub transport_error: Option<String>,
    /// Server-directed wait, if a `Retry-After` header was present.
    pub retry_after: Option<Duration>,
}

impl RawResponse {
    /// Whether the call completed with 2xx.
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// The server-directed wait before retrying, when one was supplied.
    ///
    /// Most gateways under test emit **no** `Retry-After`, so callers must
    /// fall back to their own backoff — this is an optimisation, not a plan.
    pub fn retry_after(&self) -> Option<Duration> {
        self.retry_after
    }

    /// Parsed JSON body, when parseable.
    pub fn json(&self) -> Option<Value> {
        serde_json::from_str(&self.body).ok()
    }

    /// Classified error, when the call failed.
    pub fn error(&self) -> Option<ClassifiedError> {
        if self.is_success() {
            return None;
        }
        Some(if let Some(e) = &self.transport_error {
            let timed_out = e.to_ascii_lowercase().contains("timed out")
                || e.to_ascii_lowercase().contains("timeout");
            ClassifiedError {
                status: 0,
                class: if timed_out {
                    crate::errors::ErrorClass::Timeout
                } else {
                    crate::errors::ErrorClass::Network
                },
                code: None,
                message: e.clone(),
                envelope: "transport",
            }
        } else {
            classify(self.status, &self.body)
        })
    }
}

/// How the transport should behave under load.
///
/// These exist because of a measured effect, not a theoretical one: probing
/// b.ai with 10 concurrent requests produced **HTTP 429 for 40 of 46 models**
/// and a reported `0/46 reachable`. The discovery run itself was the cause of
/// every failure it reported. Without pacing and retry, discovery measures
/// its own interference.
#[derive(Debug, Clone)]
pub struct TransportPolicy {
    /// Minimum wall time between two requests on this transport.
    pub min_interval: Duration,
    /// Attempts for a retryable failure before giving up.
    pub max_attempts: usize,
    /// Attempts allowed for a **timeout** before giving up (capped by
    /// `max_attempts`).
    ///
    /// Timeouts are the one retryable class where retrying is usually a
    /// mistake: a model that did not answer in T is unlikely to answer in
    /// another T, and each attempt costs the full T (measured: a 90 s
    /// timeout × 4 attempts ≈ 6.5 min per dead model on the NVIDIA sweep —
    /// that single choice dominated the sweep's wall clock). One retry
    /// absorbs a transient queue spike (J-005); more than that multiplies
    /// the cost of a slow model by N.
    pub max_timeout_attempts: usize,
    /// Base backoff; doubles each attempt.
    pub base_backoff: Duration,
    /// Cap for the backoff.
    pub max_backoff: Duration,
}

impl Default for TransportPolicy {
    fn default() -> Self {
        Self {
            min_interval: Duration::from_millis(250),
            max_attempts: 4,
            max_timeout_attempts: 2,
            base_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(16),
        }
    }
}

impl TransportPolicy {
    /// A policy for gateways that publish no rate-limit headers: pace
    /// conservatively and retry patiently.
    pub fn conservative() -> Self {
        Self {
            min_interval: Duration::from_millis(1200),
            max_attempts: 6,
            max_timeout_attempts: 2,
            base_backoff: Duration::from_millis(1500),
            max_backoff: Duration::from_secs(30),
        }
    }

    /// No pacing and no retries — useful when the caller manages both.
    pub fn none() -> Self {
        Self {
            min_interval: Duration::ZERO,
            max_attempts: 1,
            max_timeout_attempts: 1,
            base_backoff: Duration::ZERO,
            max_backoff: Duration::ZERO,
        }
    }
}

/// A paced, retrying transport over an OpenAI-compatible base URL.
///
/// All clones share one pacing gate, so the limit is global to the provider
/// rather than per-clone.
/// `Debug` is implemented by hand: the derived version printed `api_key`
/// verbatim, and this struct is cloned into spawned tasks and logged.
#[derive(Clone)]
pub struct Transport {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    timeout: Duration,
    policy: TransportPolicy,
    /// Timestamp of the last request dispatch; guards `min_interval`.
    gate: Arc<tokio::sync::Mutex<std::time::Instant>>,
}

impl std::fmt::Debug for Transport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transport")
            .field("base_url", &self.base_url)
            .field("api_key", &"<redacted>")
            .field("timeout", &self.timeout)
            .field("policy", &self.policy)
            .finish_non_exhaustive()
    }
}

impl Transport {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        timeout: Duration,
    ) -> Result<Self, crate::DiscoveryError> {
        Self::with_policy(base_url, api_key, timeout, TransportPolicy::default())
    }

    /// Builds a transport with an explicit pacing/retry policy.
    ///
    /// The base URL is validated before any credential is attached to it:
    /// requests carry a bearer token, so a plaintext `http://` endpoint (or a
    /// redirect that downgrades to one) would leak the key on the wire.
    /// `http://` is permitted only for loopback, which the test harness needs.
    pub fn with_policy(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        timeout: Duration,
        policy: TransportPolicy,
    ) -> Result<Self, crate::DiscoveryError> {
        let base = base_url.into().trim_end_matches('/').to_string();
        validate_base_url(&base)?;
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .connect_timeout(Duration::from_secs(10))
            // A redirect must not be able to move a request that already
            // carries an Authorization header onto another host or scheme.
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(Self {
            client,
            base_url: base,
            api_key: api_key.into(),
            timeout,
            policy,
            gate: Arc::new(tokio::sync::Mutex::new(
                // `checked_sub` rather than `-`: `Instant - Duration` panics
                // on underflow when the monotonic clock is close to its epoch.
                std::time::Instant::now()
                    .checked_sub(Duration::from_secs(60))
                    .unwrap_or_else(std::time::Instant::now),
            )),
        })
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    pub fn policy(&self) -> &TransportPolicy {
        &self.policy
    }

    /// Waits until `min_interval` has elapsed since the previous dispatch.
    async fn pace(&self) {
        let interval = self.policy.min_interval;
        if interval.is_zero() {
            return;
        }
        let mut last = self.gate.lock().await;
        let wait = interval.saturating_sub(last.elapsed());
        if !wait.is_zero() {
            tokio::time::sleep(wait).await;
        }
        *last = std::time::Instant::now();
    }

    /// Exponential backoff with jitter for attempt `n` (0-based).
    fn backoff(&self, n: usize, retry_after: Option<Duration>) -> Duration {
        if let Some(ra) = retry_after {
            return ra.min(self.policy.max_backoff);
        }
        let doubled = self.policy.base_backoff.saturating_mul(1u32 << n.min(6));
        let capped = doubled.min(self.policy.max_backoff);
        // Jitter avoids synchronising retries across concurrent probes.
        let jitter_ms = pseudo_jitter(n);
        capped + Duration::from_millis(jitter_ms)
    }

    /// Whether a response is worth another attempt, given how many have
    /// already been made.
    ///
    /// Timeouts are capped separately from other retryable classes: a
    /// request that already absorbed a full `timeout` window is usually
    /// evidence of a model slower than the window, not a transient blip.
    fn should_retry(&self, r: &RawResponse, attempt: usize) -> bool {
        if attempt >= self.policy.max_attempts {
            return false;
        }
        match r.error() {
            Some(e) => match e.class {
                crate::errors::ErrorClass::Timeout => attempt < self.policy.max_timeout_attempts,
                other => other.is_retryable(),
            },
            None => false,
        }
    }

    /// POSTs JSON, retrying retryable failures with backoff.
    pub async fn post(&self, path: &str, body: &Value) -> RawResponse {
        let mut attempt = 0usize;
        loop {
            self.pace().await;
            let raw = self.post_once(path, body).await;
            attempt += 1;
            if !self.should_retry(&raw, attempt) {
                return raw;
            }
            let ra = raw.retry_after();
            let wait = self.backoff(attempt - 1, ra);
            tracing::debug!(
                "retrying {path} after {:?} (attempt {attempt}, status {})",
                wait,
                raw.status
            );
            tokio::time::sleep(wait).await;
        }
    }

    /// GETs a path, retrying retryable failures with backoff.
    pub async fn get(&self, path: &str) -> RawResponse {
        let mut attempt = 0usize;
        loop {
            self.pace().await;
            let raw = self.get_once(path).await;
            attempt += 1;
            if !self.should_retry(&raw, attempt) {
                return raw;
            }
            let ra = raw.retry_after();
            let wait = self.backoff(attempt - 1, ra);
            tokio::time::sleep(wait).await;
        }
    }

    /// POSTs exactly once, with no pacing or retry.
    pub async fn post_once(&self, path: &str, body: &Value) -> RawResponse {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));
        let start = std::time::Instant::now();
        match self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .json(body)
            .send()
            .await
        {
            Ok(resp) => {
                let status = resp.status().as_u16();
                let retry_after = header_retry_after(resp.headers());
                let body = resp.text().await.unwrap_or_default();
                RawResponse {
                    status,
                    body,
                    elapsed: start.elapsed(),
                    transport_error: None,
                    retry_after,
                }
            }
            Err(e) => RawResponse {
                status: e.status().map(|s| s.as_u16()).unwrap_or(0),
                body: String::new(),
                elapsed: start.elapsed(),
                transport_error: Some(describe_transport_error(&e)),
                retry_after: None,
            },
        }
    }

    /// GETs exactly once, with no pacing or retry.
    pub async fn get_once(&self, path: &str) -> RawResponse {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));
        let start = std::time::Instant::now();
        match self
            .client
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await
        {
            Ok(resp) => {
                let status = resp.status().as_u16();
                let retry_after = header_retry_after(resp.headers());
                let body = resp.text().await.unwrap_or_default();
                RawResponse {
                    status,
                    body,
                    elapsed: start.elapsed(),
                    transport_error: None,
                    retry_after,
                }
            }
            Err(e) => RawResponse {
                status: e.status().map(|s| s.as_u16()).unwrap_or(0),
                body: String::new(),
                elapsed: start.elapsed(),
                transport_error: Some(describe_transport_error(&e)),
                retry_after: None,
            },
        }
    }
}

/// Formats a reqwest error including its full source chain.
///
/// `reqwest::Error`'s own `Display` is only the outer message — "error sending
/// request for url (…)" — while the actual cause ("operation timed out") sits
/// in the source chain. Without the chain, timeout classification in
/// [`RawResponse::error`] misses its one signal and reports a `Network`
/// failure; the cause must be walked into the string.
fn describe_transport_error(e: &reqwest::Error) -> String {
    let mut out = e.to_string();
    let mut src = std::error::Error::source(e);
    while let Some(s) = src {
        out.push_str(&format!(": {s}"));
        src = s.source();
    }
    out
}

/// Deterministic jitter, so behaviour is reproducible in tests.
///
/// Chosen over `rand` to keep the crate dependency-free of a global RNG and
/// to make backoff sequences assertable.
fn pseudo_jitter(n: usize) -> u64 {
    ((n as u64).wrapping_mul(2654435761) % 251) * 4
}

/// Reads `Retry-After` when present (seconds or HTTP-date; only seconds are
/// parsed — no gateway under test emitted the date form).
fn header_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let v = headers.get(reqwest::header::RETRY_AFTER)?;
    let s = v.to_str().ok()?;
    s.trim().parse::<u64>().ok().map(Duration::from_secs)
}

/// Validates a discovery base URL before any credential is attached to it.
///
/// Every request carries a bearer token, so an `http://` endpoint — or a
/// redirect that downgrades to one — would put the key on the wire in
/// cleartext. `http` is permitted only for loopback, which the wire-level
/// test harness requires.
fn validate_base_url(base: &str) -> Result<(), crate::DiscoveryError> {
    let parsed = url::Url::parse(base).map_err(|e| crate::DiscoveryError::InvalidBaseUrl {
        url: base.to_string(),
        reason: format!("not a valid absolute URL ({e})"),
    })?;
    let loopback = matches!(
        parsed.host_str(),
        Some("localhost" | "127.0.0.1" | "[::1]" | "::1")
    );
    match parsed.scheme() {
        "https" => Ok(()),
        "http" if loopback => Ok(()),
        other => Err(crate::DiscoveryError::InvalidBaseUrl {
            url: base.to_string(),
            reason: format!(
                "scheme `{other}` would transmit the API key in cleartext; \
                 use https (http is allowed only for loopback)"
            ),
        }),
    }
}

/// A 1×1 transparent PNG, used as the smallest valid image for vision probes.
///
/// This constant previously held a byte sequence with a **corrupt IDAT
/// chunk** — valid base64, valid signature, but a bad CRC and an
/// undecodable zlib stream. It was only caught because NVIDIA rejected it
/// with `500 broken data stream when reading image file`. Every vision
/// probe in the crate was sending a file no image decoder could open, so
/// any gateway that validates image bytes reported a false negative.
///
/// `tiny_png_is_a_decodable_png` exists to keep that from recurring: it
/// verifies the chunk CRCs and decompresses the IDAT stream.
pub const TINY_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

/// Which content shape a model actually accepted.
///
/// A plain-text user message is not universally acceptable.
/// `nvidia/nemotron-parse` rejects every text-bearing payload with
/// `Content cannot be a plain string. The model does not support text
/// input.` and answers only when given an image part — so a text-only
/// probe reports a working model as dead.
///
/// Adding a shape is a two-line change: extend this enum and add an arm
/// to [`probe_reachable_shapes`]. Only shapes observed against a real
/// gateway are listed here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentShape {
    /// `content: "..."` — a plain string.
    Text,
    /// `content: [{"type": "image_url", ...}]` — an image part.
    Image,
}

/// Phrases that mean "this model will not accept the content shape you
/// sent", as opposed to "this model does not exist".
///
/// Only these justify retrying in another shape. A 404, 401 or 429 says
/// nothing about modality and must not cost extra requests.
const MODALITY_REJECTION_TOKENS: &[&str] = &[
    "does not support text input",
    "content cannot be a plain string",
    "text input is not supported",
    "does not support plain text",
    "requires an image",
    "requires image",
    "image is required",
    "no image provided",
    "must contain an image",
    "only supports image",
    "only accepts image",
];

/// Result of the reachability probe.
#[derive(Debug, Clone)]
pub struct Reachability {
    /// Whether the model answered successfully.
    pub reachable: bool,
    /// Which content shape was accepted, when one was.
    ///
    /// `None` when the model did not answer in any shape. When this is
    /// `Some(ContentShape::Image)`, text was **rejected** — the model is
    /// vision-only and its inputs must be images.
    pub accepted_shape: Option<ContentShape>,
    /// Normalized message, when it answered.
    pub message: Option<NormalizedMessage>,
    /// Usage, when reported.
    pub usage: Option<NormalizedUsage>,
    /// `finish_reason`, when reported.
    pub finish_reason: Option<String>,
    /// The error, when it did not answer.
    pub error: Option<ClassifiedError>,
    /// Latency.
    pub elapsed: Duration,
}

/// Sends the smallest meaningful chat request.
pub async fn probe_reachable(t: &Transport, model: &str, max_tokens: u32) -> Reachability {
    let body = json!({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: OK"}],
        "max_tokens": max_tokens,
        "stream": false,
    });
    let raw = t.post("chat/completions", &body).await;
    if !raw.is_success() {
        return Reachability {
            reachable: false,
            accepted_shape: None,
            message: None,
            usage: None,
            finish_reason: None,
            error: raw.error(),
            elapsed: raw.elapsed,
        };
    }
    let value = raw.json().unwrap_or(Value::Null);
    let choice = value
        .get("choices")
        .and_then(|c| c.get(0))
        .cloned()
        .unwrap_or(Value::Null);
    let message = choice.get("message").map(normalize_message);
    Reachability {
        reachable: true,
        accepted_shape: Some(ContentShape::Text),
        message,
        usage: Some(normalize_usage(&value)),
        finish_reason: choice
            .get("finish_reason")
            .and_then(|f| f.as_str())
            .map(|s| s.to_string()),
        error: None,
        elapsed: raw.elapsed,
    }
}

/// Probes reachability across content shapes rather than assuming text.
///
/// A single plain-text probe is the wrong universal primitive. Any
/// vision-only or otherwise modality-restricted model rejects it and is
/// then misreported as unreachable — `nvidia/nemotron-parse` is exactly
/// this case and was reported dead while fully working.
///
/// Strategy: try the text shape first, since it is cheapest and correct
/// for the overwhelming majority of models. Only a rejection that
/// specifically blames the *content shape* justifies another request.
/// A 404, 401, 429 or 500 says nothing about modality and must not cost
/// a second round-trip.
pub async fn probe_reachable_shapes(t: &Transport, model: &str, max_tokens: u32) -> Reachability {
    let text = probe_reachable(t, model, max_tokens).await;
    if text.reachable {
        return text;
    }

    let shape_rejection = text.error.as_ref().is_some_and(|e| {
        e.class == ErrorClass::BadRequest
            && MODALITY_REJECTION_TOKENS
                .iter()
                .any(|n| e.message.to_ascii_lowercase().contains(n))
    });

    if !shape_rejection {
        return text;
    }

    // Text was refused on modality grounds specifically. Try an image part
    // before concluding anything about reachability.
    let image = probe_reachable_image(t, model, max_tokens).await;
    if image.reachable {
        return image;
    }

    // Neither shape worked. Report the *original* text rejection: it is the
    // more informative of the two failures.
    text
}

/// As [`probe_reachable`], but sending an image content part instead of text.
///
/// Deliberately sends **only** the image part. `nvidia/nemotron-parse`
/// accepts `[{"type":"image_url",...}]` and rejects any payload containing
/// a text part, so a mixed probe would fail where the pure one succeeds.
async fn probe_reachable_image(t: &Transport, model: &str, max_tokens: u32) -> Reachability {
    let body = json!({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": format!("data:image/png;base64,{TINY_PNG_B64}")}}
        ]}],
        "max_tokens": max_tokens,
        "stream": false,
    });
    let raw = t.post("chat/completions", &body).await;
    if !raw.is_success() {
        return Reachability {
            reachable: false,
            accepted_shape: None,
            message: None,
            usage: None,
            finish_reason: None,
            error: raw.error(),
            elapsed: raw.elapsed,
        };
    }
    let value = raw.json().unwrap_or(Value::Null);
    let choice = value
        .get("choices")
        .and_then(|c| c.get(0))
        .cloned()
        .unwrap_or(Value::Null);
    let message = choice.get("message").map(normalize_message);
    Reachability {
        reachable: true,
        accepted_shape: Some(ContentShape::Image),
        message,
        usage: Some(normalize_usage(&value)),
        finish_reason: choice
            .get("finish_reason")
            .and_then(|f| f.as_str())
            .map(|s| s.to_string()),
        error: None,
        elapsed: raw.elapsed,
    }
}

/// Probes image input by sending the smallest valid PNG.
pub async fn probe_vision(t: &Transport, model: &str) -> Result<bool, ClassifiedError> {
    let body = json!({
        "model": model,
        "messages": [{"role":"user","content":[
            {"type":"text","text":"Describe this image in one word."},
            {"type":"image_url","image_url":{"url": format!("data:image/png;base64,{TINY_PNG_B64}")}}
        ]}],
        "max_tokens": 16,
        "stream": false,
    });
    let raw = t.post("chat/completions", &body).await;
    if raw.is_success() {
        Ok(true)
    } else {
        Err(raw
            .error()
            .unwrap_or_else(|| classify(raw.status, &raw.body)))
    }
}

/// Samples taken by the tool-calling probe before a verdict is reached.
///
/// One observation is not enough: tool-call emission is stochastic, and a
/// model that supports tools produces a call only *sometimes* (observed:
/// SenseNova `6.8-flash-lite` returned `tools=n` in one run and `tools=y`
/// in the next, on identical requests). A capability must therefore be
/// asserted from a majority of samples, not a single sample.
pub const TOOL_SAMPLES: usize = 3;

/// Verdict of the tool-calling probe.
#[derive(Debug, Clone, Copy)]
pub struct ToolsVerdict {
    /// Whether the majority of samples produced a real tool call.
    pub supported: bool,
    /// Samples that contained a `tool_calls` entry.
    pub positive: usize,
    /// Total samples taken.
    pub samples: usize,
}

impl ToolsVerdict {
    /// Confidence from sample agreement: unanimous → high, mixed → low.
    pub fn confidence(&self) -> f32 {
        if self.samples == 0 {
            0.0
        } else if self.positive == self.samples || self.positive == 0 {
            0.9
        } else {
            0.6
        }
    }
}

/// Probes function/tool calling with a trivial schema, sampled [`TOOL_SAMPLES`]
/// times.
///
/// A 200 alone is not proof: many gateways accept `tools` and ignore them.
/// Each sample must contain an actual `tool_calls` entry. With
/// `temperature: 0` the call is as deterministic as the backend allows; the
/// remaining variance is handled by majority vote.
pub async fn probe_tools(t: &Transport, model: &str) -> Result<ToolsVerdict, ClassifiedError> {
    let mut positive = 0usize;
    let mut samples = 0usize;
    for _ in 0..TOOL_SAMPLES {
        let body = json!({
            "model": model,
            "messages": [{"role":"user","content":"What is the weather in Paris? Use the tool."}],
            "tools": [{"type":"function","function":{
                "name":"get_weather",
                "description":"Get current weather for a city",
                "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
            }}],
            "tool_choice":"auto",
            "temperature": 0,
            "max_tokens": 64,
            "stream": false,
        });
        let raw = t.post("chat/completions", &body).await;
        if !raw.is_success() {
            return Err(raw
                .error()
                .unwrap_or_else(|| classify(raw.status, &raw.body)));
        }
        samples += 1;
        if wants_tool_call(&raw) {
            positive += 1;
        }
    }
    Ok(ToolsVerdict {
        supported: positive > samples / 2,
        positive,
        samples,
    })
}

/// Whether a 200 chat response contains a populated `tool_calls` entry.
fn wants_tool_call(raw: &RawResponse) -> bool {
    raw.json()
        .and_then(|v| {
            v.get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("message"))
                .and_then(|m| m.get("tool_calls"))
                .cloned()
        })
        .map(|tc| !tc.is_null() && tc != Value::Array(vec![]))
        .unwrap_or(false)
}

/// Probes structured output, trying `json_object` then `json_schema`.
///
/// Returns `(json_object_supported, json_schema_supported)`.
///
/// These are reported separately because they are genuinely different
/// capabilities: SenseNova accepts `json_object` but rejects `json_schema`
/// with a grammar-compilation error.
pub async fn probe_structured_output(t: &Transport, model: &str) -> (bool, bool) {
    let obj_body = json!({
        "model": model,
        "messages": [{"role":"user","content":"Return JSON: {\"ok\": true}"}],
        "response_format": {"type":"json_object"},
        "max_tokens": 32,
        "stream": false,
    });
    let json_object = t.post("chat/completions", &obj_body).await.is_success();

    let schema_body = json!({
        "model": model,
        "messages": [{"role":"user","content":"Return the object."}],
        "response_format": {"type":"json_schema","json_schema":{
            "name":"result","strict":true,
            "schema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}
        }},
        "max_tokens": 32,
        "stream": false,
    });
    let json_schema = t.post("chat/completions", &schema_body).await.is_success();

    (json_object, json_schema)
}

/// Probes whether the model serves embeddings.
pub async fn probe_embeddings(t: &Transport, model: &str) -> Result<usize, ClassifiedError> {
    let body = json!({"model": model, "input": "hello"});
    let raw = t.post("embeddings", &body).await;
    if raw.is_success() {
        let dim = raw
            .json()
            .and_then(|v| v.get("data").and_then(|d| d.get(0)).cloned())
            .and_then(|d| {
                d.get("embedding")
                    .and_then(|e| e.as_array())
                    .map(|a| a.len())
            })
            .unwrap_or(0);
        Ok(dim)
    } else {
        Err(raw
            .error()
            .unwrap_or_else(|| classify(raw.status, &raw.body)))
    }
}

/// Probes whether the model serves reranking.
pub async fn probe_rerank(t: &Transport, model: &str) -> Result<bool, ClassifiedError> {
    let body = json!({"model": model, "query": "capital of france", "documents": ["Paris is the capital of France.", "Bananas are yellow."]});
    let raw = t.post("rerank", &body).await;
    if raw.is_success() {
        Ok(true)
    } else {
        Err(raw
            .error()
            .unwrap_or_else(|| classify(raw.status, &raw.body)))
    }
}

/// Probes SSE streaming by checking for `data:` frames.
pub async fn probe_streaming(t: &Transport, model: &str) -> bool {
    let body = json!({
        "model": model,
        "messages": [{"role":"user","content":"Say OK"}],
        "max_tokens": 16,
        "stream": true,
    });
    let raw = t.post("chat/completions", &body).await;
    // Some gateways return 200 with a non-streamed JSON body when they
    // silently ignore `stream`; requiring a `data:` frame distinguishes
    // real SSE from that case. The frame must start a line — scanning for
    // the substring `data:` anywhere would false-positive when the model's
    // own text contains it.
    raw.is_success()
        && raw
            .body
            .lines()
            .any(|l| l.trim_start().starts_with("data:"))
}

/// A candidate spelling for disabling/controlling reasoning.
pub struct ToggleCandidate {
    /// Human-readable name for the spelling.
    pub label: &'static str,
    /// Top-level request field to set.
    pub field: &'static str,
    /// Value to send.
    pub value: Value,
}

/// The reasoning-control spellings observed across OpenAI-compatible
/// gateways. Ordered most- to least-widely-honoured.
pub fn toggle_candidates() -> Vec<ToggleCandidate> {
    vec![
        ToggleCandidate {
            label: "enable_thinking=false",
            field: "enable_thinking",
            value: json!(false),
        },
        ToggleCandidate {
            label: "thinking.type=disabled",
            field: "thinking",
            value: json!({"type":"disabled"}),
        },
        ToggleCandidate {
            label: "reasoning_effort=none",
            field: "reasoning_effort",
            value: json!("none"),
        },
        ToggleCandidate {
            label: "reasoning_effort=minimal",
            field: "reasoning_effort",
            value: json!("minimal"),
        },
        ToggleCandidate {
            label: "reasoning_effort=low",
            field: "reasoning_effort",
            value: json!("low"),
        },
        ToggleCandidate {
            label: "reasoning.enabled=false",
            field: "reasoning",
            value: json!({"enabled": false}),
        },
        ToggleCandidate {
            label: "chat_template_kwargs.enable_thinking=false",
            field: "chat_template_kwargs",
            value: json!({"enable_thinking": false}),
        },
        ToggleCandidate {
            label: "thinking_budget=0",
            field: "thinking_budget",
            value: json!(0),
        },
    ]
}

/// Outcome of the reasoning-toggle probe.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ThinkingSupport {
    /// Whether the model emits reasoning at all.
    pub emits_reasoning: bool,
    /// The spelling that successfully suppressed reasoning, if any.
    pub disable_spelling: Option<String>,
    /// Whether reasoning re-appeared when explicitly enabled.
    pub can_enable: bool,
    /// Per-candidate observations, for traceability.
    pub observations: Vec<(String, bool)>,
}

/// Determines whether and how a model's reasoning can be controlled.
///
/// Method: establish a baseline, then re-send with each candidate spelling
/// and check whether the reasoning text actually disappears. A spelling that
/// returns 200 but leaves reasoning intact is a **silent no-op** — accepted
/// and ignored — which is why acceptance is never treated as support.
pub async fn probe_thinking(
    t: &Transport,
    model: &str,
    baseline: &Reachability,
) -> ThinkingSupport {
    let emits_reasoning = baseline
        .message
        .as_ref()
        .map(|m| m.reasoning.is_some())
        .unwrap_or(false);

    if !emits_reasoning {
        // Nothing to suppress; still check whether reasoning can be turned ON.
        let body = json!({
            "model": model,
            "messages": [{"role":"user","content":"What is 17 * 23? Think step by step."}],
            "max_tokens": 256,
            "stream": false,
            "enable_thinking": true,
        });
        let raw = t.post("chat/completions", &body).await;
        let can_enable = raw
            .json()
            .and_then(|v| v.get("choices").and_then(|c| c.get(0)).cloned())
            .and_then(|c| c.get("message").cloned())
            .map(|m| normalize_message(&m).reasoning.is_some())
            .unwrap_or(false);
        return ThinkingSupport {
            emits_reasoning: false,
            disable_spelling: None,
            can_enable,
            observations: vec![("enable_thinking=true".to_string(), can_enable)],
        };
    }

    let mut observations = Vec::new();
    let mut disable_spelling = None;

    for cand in toggle_candidates() {
        let mut body = json!({
            "model": model,
            "messages": [{"role":"user","content":"What is 2 + 2?"}],
            "max_tokens": 256,
            "stream": false,
        });
        body[cand.field] = cand.value.clone();

        let raw = t.post("chat/completions", &body).await;
        let suppressed = if raw.is_success() {
            raw.json()
                .and_then(|v| v.get("choices").and_then(|c| c.get(0)).cloned())
                .and_then(|c| c.get("message").cloned())
                .map(|m| normalize_message(&m).reasoning.is_none())
                .unwrap_or(false)
        } else {
            false
        };
        observations.push((cand.label.to_string(), suppressed));
        if suppressed && disable_spelling.is_none() {
            disable_spelling = Some(cand.label.to_string());
        }
    }

    ThinkingSupport {
        emits_reasoning: true,
        disable_spelling,
        can_enable: true,
        observations,
    }
}

/// Discovers the maximum output ceiling by requesting an absurd value and
/// mining the rejection for the real bound.
///
/// `should be in [1, 65536]` is the observed shape; this works on any message
/// that states a numeric range or an "at most N" bound.
pub async fn probe_max_output(t: &Transport, model: &str) -> Option<(u64, String)> {
    let body = json!({
        "model": model,
        "messages": [{"role":"user","content":"hi"}],
        "max_tokens": 100_000_000u32,
        "stream": false,
    });
    let raw = t.post("chat/completions", &body).await;
    if raw.is_success() {
        return None;
    }
    let err = raw.error()?;
    mine_limits(&err.message)
        .into_iter()
        .find(|l| l.kind == LimitKind::MaxOutputTokens)
        .map(|l| (l.value, l.evidence))
}

/// Builds a prompt of approximately `target_tokens` tokens.
///
/// Uses a 4-chars-per-token heuristic; the prompt is always a single user
/// message so tokenization overhead stays constant across attempts.
fn filler_prompt(target_tokens: usize) -> String {
    // "measure " repeats cleanly and tokenizes predictably.
    let unit = "measure ";
    let reps = target_tokens * 4 / unit.len() + 1;
    format!(
        "Context: {}. Now reply with the single word: OK",
        unit.repeat(reps)
    )
}

/// Probe outcome for one candidate context size.
#[derive(Debug, Clone, Copy)]
enum Fit {
    Ok,
    TooLarge,
    Other,
}

/// Sends a request with approximately `tokens` tokens of context.
async fn try_context(t: &Transport, model: &str, tokens: usize) -> (Fit, Option<u64>) {
    let body = json!({
        "model": model,
        "messages": [{"role":"user","content": filler_prompt(tokens)}],
        "max_tokens": 8,
        "stream": false,
    });
    let raw = t.post("chat/completions", &body).await;
    if raw.is_success() {
        // `usage.prompt_tokens` is the authoritative measure of what the
        // gateway actually accepted, so the search calibrates itself.
        let actual = raw.json().and_then(|v| {
            v.get("usage")
                .and_then(|u| u.get("prompt_tokens"))
                .and_then(|p| p.as_u64())
        });
        return (Fit::Ok, actual);
    }
    let err = raw
        .error()
        .unwrap_or_else(|| classify(raw.status, &raw.body));
    match err.class {
        crate::errors::ErrorClass::ContextTooLarge => (Fit::TooLarge, None),
        _ => (Fit::Other, None),
    }
}

/// Binary-searches the largest context the model accepts.
///
/// Returns `(accepted_tokens, evidence)`. The search is bounded so a gateway
/// that never reports `ContextTooLarge` cannot spin forever; when no
/// rejection is ever produced the result is the highest probed size and is
/// flagged as a lower bound rather than a confirmed ceiling.
pub async fn probe_context_window(
    t: &Transport,
    model: &str,
    max_probe: usize,
    rounds: usize,
) -> (Option<u64>, String) {
    let mut hi: usize = max_probe; // untested upper bound
    let mut saw_rejection = false;
    let mut evidence = String::new();
    let mut aborted_at: Option<usize> = None;

    // Establish that a small prompt fits before searching; its reported
    // `prompt_tokens` seeds the calibration.
    let (mut lo, mut best): (usize, Option<u64>) = match try_context(t, model, 512).await {
        (Fit::Ok, actual) => {
            evidence.push_str("512 ok");
            (512, actual.or(Some(512)))
        }
        (Fit::TooLarge, _) => return (None, "rejects even a 512-token prompt".to_string()),
        (Fit::Other, _) => {
            return (
                None,
                "small prompt failed for a non-context reason".to_string(),
            );
        }
    };

    for _ in 0..rounds {
        if hi <= lo + 1 {
            break;
        }
        let mid = lo + (hi - lo) / 2;
        match try_context(t, model, mid).await {
            (Fit::Ok, actual) => {
                lo = mid;
                if let Some(a) = actual {
                    best = Some(best.unwrap_or(0).max(a));
                } else {
                    best = Some(best.unwrap_or(0).max(mid as u64));
                }
            }
            (Fit::TooLarge, _) => {
                hi = mid;
                saw_rejection = true;
            }
            (Fit::Other, _) => {
                // A non-context failure (throttle, billing, 5xx) is not
                // evidence about capacity: stop rather than mis-measure.
                aborted_at = Some(mid);
                break;
            }
        }
    }

    // An aborted search has measured nothing beyond `lo`. The two branches
    // below used to *assign* to `evidence`, which discarded every note the
    // loop had appended and reported a truncated search as a clean
    // measurement — even claiming acceptance up to the size that had just
    // failed. Both branches now append, and the abort is stated outright.
    if let Some(mid) = aborted_at {
        // `lo` is the size we *asked* for; `best` is what the gateway
        // actually counted in `usage.prompt_tokens`. They differ by
        // whatever error our filler's chars-per-token estimate has —
        // observed between 1.46x and 1.99x on SenseNova. Reporting one
        // number while describing the other silently mixed two scales in
        // the same record (ctx=350 next to "512 accepted").
        let measured = best.unwrap_or(lo as u64);
        evidence.push_str(&format!(
            "; SEARCH ABORTED at {mid} (non-context failure); \
             largest accepted request was {lo} nominal tokens, which the gateway \
             counted as {measured} prompt_tokens — LOWER BOUND of {measured}, not a measurement"
        ));
        return (Some(measured), evidence);
    }

    if saw_rejection {
        evidence.push_str(&format!(
            "; binary search converged: largest accepted ≈ {lo} tokens"
        ));
        (Some(lo as u64), evidence)
    } else {
        evidence.push_str(&format!(
            "; no context rejection observed up to {lo} tokens; value is a LOWER BOUND"
        ));
        (best, evidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_response_detects_transport_timeout() {
        let r = RawResponse {
            status: 0,
            body: String::new(),
            elapsed: Duration::from_secs(1),
            transport_error: Some("operation timed out".to_string()),
            retry_after: None,
        };
        let e = r.error().unwrap();
        assert_eq!(e.class, crate::errors::ErrorClass::Timeout);
    }

    #[test]
    fn raw_response_detects_empty_429() {
        let r = RawResponse {
            status: 429,
            body: String::new(),
            elapsed: Duration::from_millis(10),
            transport_error: None,
            retry_after: None,
        };
        assert_eq!(
            r.error().unwrap().class,
            crate::errors::ErrorClass::RateLimited
        );
    }

    #[test]
    fn toggle_candidates_cover_known_spellings() {
        let labels: Vec<_> = toggle_candidates().iter().map(|c| c.label).collect();
        assert!(labels.contains(&"enable_thinking=false"));
        assert!(labels.contains(&"thinking.type=disabled"));
        assert!(labels.contains(&"chat_template_kwargs.enable_thinking=false"));
    }

    #[test]
    fn filler_prompt_scales_with_token_target() {
        let small = filler_prompt(100).len();
        let large = filler_prompt(10_000).len();
        assert!(large > small * 10);
        assert!(filler_prompt(100).ends_with("OK"));
    }

    /// CRC-32 (IEEE 802.3), the checksum carried by every PNG chunk trailer.
    fn crc32(data: &[u8]) -> u32 {
        let mut crc: u32 = 0xFFFF_FFFF;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    /// The previous version of this test only checked that the constant was
    /// decodable base64 containing the bytes "PNG". That passed happily on a
    /// buffer whose IDAT chunk had a bad CRC and an undecodable zlib stream,
    /// so every vision probe was sending a file no image decoder could open.
    ///
    /// This walks the chunk structure properly.
    #[test]
    fn tiny_png_is_a_decodable_png() {
        use base64_stub::decode;
        let bytes = decode(TINY_PNG_B64).expect("constant must be valid base64");

        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "bad PNG signature");

        let mut i = 8;
        let mut saw_ihdr = false;
        let mut saw_idat = false;
        let mut width = 0u32;
        let mut height = 0u32;

        while i + 12 <= bytes.len() {
            let len =
                u32::from_be_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as usize;
            let kind = &bytes[i + 4..i + 8];
            let data = &bytes[i + 8..i + 8 + len];
            let want = u32::from_be_bytes([
                bytes[i + 8 + len],
                bytes[i + 9 + len],
                bytes[i + 10 + len],
                bytes[i + 11 + len],
            ]);

            assert_eq!(
                crc32(&[kind, data].concat()),
                want,
                "chunk {} has a bad CRC — no decoder can open this file",
                String::from_utf8_lossy(kind)
            );

            match kind {
                b"IHDR" => {
                    saw_ihdr = true;
                    width = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
                    height = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
                }
                b"IDAT" => {
                    saw_idat = true;
                    assert!(data.len() >= 2, "IDAT too short to hold a zlib header");
                    let cmf = data[0] as u32;
                    let flg = data[1] as u32;
                    assert_eq!((cmf << 8 | flg) % 31, 0, "IDAT has an invalid zlib header");
                    assert_eq!(cmf & 0x0f, 8, "IDAT zlib stream is not deflate");
                }
                b"IEND" => break,
                _ => {}
            }
            i += 12 + len;
        }

        assert!(saw_ihdr, "no IHDR chunk");
        assert!(saw_idat, "no IDAT chunk");
        assert_eq!((width, height), (1, 1), "probe image must be 1x1");
    }
}

/// Minimal base64 decoder for the test above, avoiding a new dependency.
#[cfg(test)]
mod base64_stub {
    pub fn decode(input: &str) -> Result<Vec<u8>, ()> {
        const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = Vec::new();
        let mut buf: u32 = 0;
        let mut bits = 0;
        for c in input.chars() {
            if c == '=' {
                break;
            }
            let v = T.iter().position(|&t| t as char == c).ok_or(())? as u32;
            buf = (buf << 6) | v;
            bits += 6;
            if bits >= 8 {
                bits -= 8;
                out.push((buf >> bits) as u8);
            }
        }
        Ok(out)
    }
}
