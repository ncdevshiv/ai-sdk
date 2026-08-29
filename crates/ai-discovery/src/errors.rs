//! Provider-agnostic error classification and limit mining.
//!
//! Two problems this module solves, both observed against real gateways:
//!
//! 1. **Error envelopes differ per provider.** The same `404` arrives as
//!    `{"error":{"message":...}}`, `{"status":404,"title":...,"detail":...}`
//!    (RFC 7807), `{"object":"error","message":...}`, bare text
//!    (`404 page not found`), an nginx HTML page, or a completely **empty
//!    body** (b.ai throttling). Classifying by HTTP status alone loses the
//!    reason; parsing only `error.message` returns garbage for four of those.
//!
//! 2. **Error messages frequently state the limit they enforce.** NVIDIA and
//!    SenseNova both return `should be in [1, 65536]`. Mining that text turns
//!    a rejection into a capability fact, which is the only way to learn
//!    output ceilings from gateways that publish no metadata.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// What went wrong, independent of provider vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorClass {
    /// The model id is not served by this gateway.
    ModelNotFound,
    /// The model exists and is served, but this account has not been granted
    /// access to it. Distinct from `ModelNotFound`: the remediation is to
    /// request entitlement, not to pick a different model.
    ///
    /// Observed on NVIDIA, which returns HTTP 404 with
    /// `Function '<uuid>': Not found for account '<account-id>'` for 45 of
    /// its 83 listed models.
    NotEntitled,
    /// The model has been permanently retired by the provider.
    ///
    /// Observed on NVIDIA as HTTP 410 with an RFC 7807 body carrying the
    /// end-of-life date. This is the strongest possible negative signal —
    /// stronger than `ModelNotFound` — and is never retryable.
    Gone,
    /// Credentials are absent, malformed or rejected.
    Authentication,
    /// Credentials are valid but this account may not use the model.
    PermissionDenied,
    /// The account has no balance/credit left.
    Billing,
    /// The gateway throttled us.
    RateLimited,
    /// The resource exists but is not invokable right now (deployment
    /// rotation, degraded function, upstream provider pool mismatch).
    ///
    /// Distinct from [`ErrorClass::RateLimited`]: no Retry-After is promised
    /// and the wait is unbounded (minutes, not seconds). Distinct from
    /// [`ErrorClass::BadRequest`]: it is the *state*, not the request, that
    /// is at fault — retrying later can succeed.
    TemporarilyUnavailable,
    /// The request was malformed or used an unsupported parameter.
    BadRequest,
    /// The context/payload was too large.
    ContextTooLarge,
    /// The gateway or its upstream failed.
    ServerError,
    /// The request did not complete in time.
    Timeout,
    /// Transport-level failure.
    Network,
    /// Unclassified.
    Other,
}

impl ErrorClass {
    /// Whether retrying the same request could plausibly succeed.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::RateLimited
                | Self::TemporarilyUnavailable
                | Self::ServerError
                | Self::Timeout
                | Self::Network
        )
    }
}

impl std::fmt::Display for ErrorClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::ModelNotFound => "model_not_found",
            Self::NotEntitled => "not_entitled",
            Self::Gone => "gone",
            Self::Authentication => "authentication",
            Self::PermissionDenied => "permission_denied",
            Self::Billing => "billing",
            Self::RateLimited => "rate_limited",
            Self::TemporarilyUnavailable => "temporarily_unavailable",
            Self::BadRequest => "bad_request",
            Self::ContextTooLarge => "context_too_large",
            Self::ServerError => "server_error",
            Self::Timeout => "timeout",
            Self::Network => "network",
            Self::Other => "other",
        };
        f.write_str(s)
    }
}

/// A normalized view of a failed HTTP response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifiedError {
    /// HTTP status (`0` when the request never completed).
    pub status: u16,
    /// Normalized class.
    pub class: ErrorClass,
    /// Provider error code, when the body carried one.
    pub code: Option<String>,
    /// Best-effort human-readable message from any envelope shape.
    pub message: String,
    /// Which envelope shape the message came from (for traceability).
    pub envelope: &'static str,
}

/// The envelope shapes we know how to read, in the order they are tried.
///
/// Exposed so callers can assert coverage: an SDK that claims to handle
/// arbitrary providers should be able to state which error shapes it reads.
pub const ENVELOPES: [&str; 6] = [
    "openai.error",
    "object.message",
    "rfc7807",
    "bare.message",
    "text",
    "empty",
];

/// Extracts a human-readable message from any known error envelope.
///
/// Returns `(message, envelope_name)`. Never fails: an unparseable or empty
/// body still yields a message describing that fact, because "empty 404" is
/// itself diagnostic (b.ai returns exactly that when throttling).
/// Upper bound on a stored error message.
///
/// Gateways occasionally return an entire backend crash dump — NVIDIA
/// returned a ~4 KB TensorRT-LLM CUDA stack trace as the `message` of a
/// 500. Carrying that verbatim into a discovery report drowns the signal
/// and bloats the output for every consumer downstream.
///
/// The cap is deliberately generous because `mine_limits` reads this same
/// string: limit phrasing ("should be in [1, 65536]") must survive.
/// Validation messages are far shorter than this; crash dumps are far
/// longer.
const MAX_MESSAGE_CHARS: usize = 1000;

pub fn extract_message(status: u16, body: &str) -> (String, &'static str) {
    let (msg, envelope) = extract_message_verbatim(status, body);
    (truncate(&msg, MAX_MESSAGE_CHARS), envelope)
}

/// As [`extract_message`], but without the cap — used where the full text
/// still has to be reasoned about.
fn extract_message_verbatim(status: u16, body: &str) -> (String, &'static str) {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return (format!("HTTP {status} with empty body"), "empty");
    }

    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        // {"error": {"message": "...", "code": "..."}}
        if let Some(err) = v.get("error") {
            if let Some(m) = err.get("message").and_then(|m| m.as_str()) {
                if !m.trim().is_empty() {
                    return (m.trim().to_string(), "openai.error");
                }
            }
            if let Some(m) = err.as_str() {
                return (m.trim().to_string(), "openai.error");
            }
            // Some gateways put the detail on a sibling of `error`.
            if let Some(m) = v.get("message").and_then(|m| m.as_str()) {
                return (m.trim().to_string(), "openai.error");
            }
        }
        // {"object": "error", "message": "..."}
        if let Some(m) = v.get("message").and_then(|m| m.as_str()) {
            if !m.trim().is_empty() {
                return (m.trim().to_string(), "object.message");
            }
        }
        // RFC 7807: {"status":..,"title":..,"detail":..}
        if v.get("status").is_some() && (v.get("title").is_some() || v.get("detail").is_some()) {
            let title = v.get("title").and_then(|t| t.as_str()).unwrap_or("");
            let detail = v.get("detail").and_then(|t| t.as_str()).unwrap_or("");
            let combined = match (title.is_empty(), detail.is_empty()) {
                (false, false) => format!("{title}: {detail}"),
                (false, true) => title.to_string(),
                (true, false) => detail.to_string(),
                (true, true) => String::new(),
            };
            if !combined.trim().is_empty() {
                return (combined, "rfc7807");
            }
        }
        // Bare {"detail": "..."} / {"title": "..."}
        for k in ["detail", "title", "error_description", "msg"] {
            if let Some(m) = v.get(k).and_then(|m| m.as_str()) {
                if !m.trim().is_empty() {
                    return (m.trim().to_string(), "bare.message");
                }
            }
        }
    }

    // Not JSON (or JSON without a usable field): strip HTML if present and
    // use the raw text. nginx 404 pages land here.
    let text = strip_html(trimmed);
    (truncate(&text, 300), "text")
}

/// Extracts the provider's error code, if any.
pub fn extract_code(body: &str) -> Option<String> {
    let v = serde_json::from_str::<Value>(body.trim()).ok()?;
    for path in [
        v.get("error").and_then(|e| e.get("code")),
        v.get("code"),
        v.get("type"),
        v.get("error").and_then(|e| e.get("type")),
    ] {
        match path {
            Some(Value::String(s)) if !s.is_empty() => return Some(s.clone()),
            Some(Value::Number(n)) => return Some(n.to_string()),
            _ => {}
        }
    }
    None
}

fn strip_html(s: &str) -> String {
    if !(s.trim_start().starts_with('<')) {
        return s.to_string();
    }
    let mut out = String::new();
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max).collect::<String>())
    }
}

/// Classifies a failed response from status + body.
///
/// Status alone is insufficient: billing failures were observed as **400**
/// (`insufficient_user_quota`), **403** (`access_denied`) and **429** across
/// three gateways, so vocabulary in the body is required to disambiguate.
pub fn classify(status: u16, body: &str) -> ClassifiedError {
    let (message, envelope) = extract_message(status, body);
    let code = extract_code(body);
    let haystack = format!(
        "{} {}",
        message.to_ascii_lowercase(),
        code.clone().unwrap_or_default().to_ascii_lowercase()
    );

    let class = if status == 0 {
        ErrorClass::Network
    } else if status == 401 {
        ErrorClass::Authentication
    } else if status == 429 {
        ErrorClass::RateLimited
    } else if status == 410 {
        // A retired model can never come back. Checked before the token
        // scans and before `status >= 500` so the retirement is never
        // mistaken for a transient upstream failure and retried.
        //
        // Observed on NVIDIA: RFC 7807 body carrying the end-of-life date.
        ErrorClass::Gone
    } else if mentions_any(&haystack, BILLING_TOKENS) || mentions_any(&haystack, QUOTA_TOKENS) {
        ErrorClass::Billing
    } else if mentions_any(&haystack, CONTEXT_TOKENS) {
        ErrorClass::ContextTooLarge
    } else if mentions_any(&haystack, TEMPORARY_TOKENS) {
        // Vocabulary beats status: "DEGRADED function cannot be invoked"
        // arrives as 400 and "No allowed providers are available" as 400,
        // both of which would otherwise classify as user errors (J-030,
        // J-029). They are neither — they are temporary infrastructure state
        // that can resolve on its own.
        ErrorClass::TemporarilyUnavailable
    } else if status == 403 {
        if mentions_any(&haystack, AUTH_TOKENS) {
            ErrorClass::Authentication
        } else {
            ErrorClass::PermissionDenied
        }
    } else if status == 404 {
        if mentions_any(&haystack, ENTITLEMENT_TOKENS) {
            // The model exists on the gateway; this key simply has not been
            // granted it. Distinct from `ModelNotFound` because the remedy
            // differs: request access, do not pick another model.
            ErrorClass::NotEntitled
        } else if mentions_any(&haystack, NOT_FOUND_TOKENS)
            || haystack.contains("not found")
            || envelope == "empty"
        {
            // A 404 with an empty body is still a 404: the thing we asked for
            // is not there. Classifying it as `Other` would hide that from
            // callers that only look at the class.
            ErrorClass::ModelNotFound
        } else {
            ErrorClass::Other
        }
    } else if status == 400 {
        ErrorClass::BadRequest
    } else if status >= 500 {
        ErrorClass::ServerError
    } else {
        ErrorClass::Other
    };

    ClassifiedError {
        status,
        class,
        code,
        message,
        envelope,
    }
}

const BILLING_TOKENS: &[&str] = &[
    "insufficient_user_quota",
    "insufficient_quota",
    "insufficient balance",
    "insufficient credit",
    "credit insufficient",
    "balance=0",
    "deposit required",
    "billing",
    "payment required",
    "no balance",
];

const QUOTA_TOKENS: &[&str] = &["quota exceeded", "exceeded your quota", "out of quota"];

/// Vocabulary of *temporary* infrastructure states: the resource exists and
/// the request is fine, but it cannot be invoked right now. Observed live:
/// `Function id '…': DEGRADED function cannot be invoked` (NVIDIA) and
/// `Upstream request failed: [404] No allowed providers are available for
/// the selected model … but your request's ***.only preference permits
/// only: tencent` (b.ai).
const TEMPORARY_TOKENS: &[&str] = &[
    "degraded function",
    "upstream request failed",
    "no allowed providers",
    "temporarily unavailable",
    "not yet available",
    "currently unavailable",
    "retry later",
    "function is not deployed",
];

const CONTEXT_TOKENS: &[&str] = &[
    "context length",
    "context_length",
    "context window",
    "maximum context",
    "too many tokens",
    "token limit",
    "prompt is too long",
    "reduce the length",
];

const AUTH_TOKENS: &[&str] = &[
    "invalid api key",
    "unauthorized",
    "authentication",
    "forbidden",
];

const NOT_FOUND_TOKENS: &[&str] = &[
    "model",
    "not found",
    "does not exist",
    "unknown model",
    "no such model",
];

/// Phrases that mean "this model exists, but this account may not use it".
///
/// Checked **before** `NOT_FOUND_TOKENS`, which contains the substring
/// `"not found"` and would otherwise swallow every entitlement message.
///
/// Observed on NVIDIA (45 of 83 listed models):
/// `Function '<uuid>': Not found for account '<account-id>'`.
const ENTITLEMENT_TOKENS: &[&str] = &[
    "not found for account",
    "not found for this account",
    "not found for your account",
    "not enabled for account",
    "not enabled for this account",
    "not available for your account",
    "not available for this account",
    "not entitled",
    "no access to this model",
];

fn mentions_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

/// A numeric limit recovered from rejection text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinedLimit {
    /// The limit value.
    pub value: u64,
    /// What the limit applies to.
    pub kind: LimitKind,
    /// The substring that produced it.
    pub evidence: String,
}

/// Which knob a mined limit constrains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LimitKind {
    /// Maximum generated tokens.
    MaxOutputTokens,
    /// Maximum input/context tokens.
    ContextWindow,
}

/// Mines numeric limits out of an error message.
///
/// Real examples this parses:
/// - `field MaxTokens invalid, should be in [1, 65536]` → max output 65536
/// - `maximum context length is 8192 tokens`            → context 8192
/// - `Please reduce the length of the messages: at most 4096 tokens`
pub fn mine_limits(text: &str) -> Vec<MinedLimit> {
    let mut out = Vec::new();
    let lower = text.to_ascii_lowercase();

    // "should be in [1, 65536]" / "must be between 1 and 65536"
    if let Some(cap) = bracket_range(&lower) {
        let evidence = cap.evidence.clone();
        out.push(MinedLimit {
            value: cap.value,
            kind: LimitKind::MaxOutputTokens,
            evidence,
        });
    }

    // "maximum context length is N"
    for pattern in [
        "maximum context length is ",
        "maximum context length of ",
        "context length of ",
        "context window is ",
        "context window of ",
        "max context length is ",
    ] {
        if let Some(n) = number_after(&lower, pattern) {
            out.push(MinedLimit {
                value: n,
                kind: LimitKind::ContextWindow,
                evidence: format!("{pattern}{n}"),
            });
        }
    }

    // "at most N tokens" / "no more than N tokens" / "fewer than N tokens"
    //
    // A unit is required after the number. Without it any sentence containing
    // the phrase becomes a token limit — observed false positive: the
    // rate-limit text "You may send up to 40 requests per minute" was mined
    // as `max_output_tokens = 40` at 0.9 confidence and then used to
    // override the curated catalog. The bare "up to " pattern was removed
    // for the same reason; it never carried a token unit reliably.
    for pattern in ["at most ", "no more than ", "less than ", "fewer than "] {
        if let Some(n) = number_after(&lower, pattern) {
            if !mentions_unit_after(&lower, pattern) {
                continue;
            }
            let ctx =
                lower.contains("context") || lower.contains("prompt") || lower.contains("input");
            out.push(MinedLimit {
                value: n,
                kind: if ctx {
                    LimitKind::ContextWindow
                } else {
                    LimitKind::MaxOutputTokens
                },
                evidence: format!("{pattern}{n}"),
            });
        }
    }

    out
}

/// Whether the number following `pattern` is measured in tokens/characters.
///
/// Guards the "at most N" family against mining arbitrary counts: the unit
/// must appear within a short window after the number.
fn mentions_unit_after(lower: &str, pattern: &str) -> bool {
    let Some(idx) = lower.find(pattern) else {
        return false;
    };
    let rest = &lower[idx + pattern.len()..];
    let mut chars = rest.chars().peekable();
    // Skip any leading non-digits, then consume the number itself.
    while let Some(c) = chars.peek() {
        if c.is_ascii_digit() {
            break;
        }
        chars.next();
    }
    while let Some(c) = chars.peek() {
        if c.is_ascii_digit() || *c == ',' || *c == '_' {
            chars.next();
        } else {
            break;
        }
    }
    let tail: String = chars.take(32).collect();
    tail.contains("token") || tail.contains("char")
}

/// Parses `[1, 65536]`-style ranges, returning the upper bound.
///
/// Scans every bracket pair: the previous implementation inspected only the
/// first `[…]`, so a message like "invalid [size] value, should be in
/// [1, 65536]" yielded nothing at all.
fn bracket_range(lower: &str) -> Option<MinedLimit> {
    let mut rest = lower;
    while let Some(start) = rest.find('[') {
        let after = &rest[start + 1..];
        let Some(end) = after.find(']') else {
            break;
        };
        let inner = &after[..end];
        let nums: Vec<u64> = inner
            .split([',', ' ', '-', ';'])
            .filter_map(|p| {
                let p = p.trim().trim_matches(|c: char| !c.is_ascii_digit());
                p.parse::<u64>().ok()
            })
            .collect();
        if let Some(max) = nums.into_iter().max() {
            if max > 0 {
                return Some(MinedLimit {
                    value: max,
                    kind: LimitKind::MaxOutputTokens,
                    evidence: format!("[{inner}]"),
                });
            }
        }
        rest = &after[end + 1..];
    }
    None
}

/// Reads the first integer following `pattern`.
///
/// Thousands separators are tolerated: "at most 128,000 tokens" must not be
/// read as 128.
fn number_after(lower: &str, pattern: &str) -> Option<u64> {
    let idx = lower.find(pattern)?;
    let rest = &lower[idx + pattern.len()..];
    let digits: String = rest
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit() || *c == ',' || *c == '_')
        .filter(|c| c.is_ascii_digit())
        .collect();
    let n = digits.parse::<u64>().ok()?;
    if n == 0 { None } else { Some(n) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_envelope() {
        let b = r#"{"error":{"message":"The model 'x' does not exist","code":"model_not_found"}}"#;
        let c = classify(404, b);
        assert_eq!(c.class, ErrorClass::ModelNotFound);
        assert_eq!(c.code.as_deref(), Some("model_not_found"));
        assert_eq!(c.envelope, "openai.error");
    }

    #[test]
    fn parses_rfc7807_envelope() {
        let b = r#"{"status":404,"title":"Not Found","detail":"Function 'abc': Not found"}"#;
        let c = classify(404, b);
        assert_eq!(c.class, ErrorClass::ModelNotFound);
        assert_eq!(c.envelope, "rfc7807");
        assert!(c.message.contains("Not Found"));
    }

    #[test]
    fn parses_object_message_envelope() {
        let b = r#"{"object":"error","message":"Content cannot be a plain string."}"#;
        let c = classify(400, b);
        assert_eq!(c.class, ErrorClass::BadRequest);
        assert_eq!(c.envelope, "object.message");
    }

    #[test]
    fn empty_body_429_is_rate_limit_not_parse_error() {
        let c = classify(429, "");
        assert_eq!(c.class, ErrorClass::RateLimited);
        assert_eq!(c.envelope, "empty");
        assert!(c.message.contains("empty body"));
        assert!(c.class.is_retryable());
    }

    #[test]
    fn html_404_is_not_json() {
        let b = "<html>\r\n<head><title>404 Not Found</title></head></html>";
        let c = classify(404, b);
        assert_eq!(c.class, ErrorClass::ModelNotFound);
        assert_eq!(c.envelope, "text");
        assert!(!c.message.contains("<html>"));
    }

    #[test]
    fn plain_text_404() {
        let c = classify(404, "404 page not found\n");
        assert_eq!(c.class, ErrorClass::ModelNotFound);
    }

    #[test]
    fn billing_detected_across_status_codes() {
        // Observed as HTTP 400 on b.ai despite being a billing failure.
        let b = r#"{"error":{"message":"credit insufficient balance: balance=0 required=2404","code":"insufficient_user_quota"}}"#;
        assert_eq!(classify(400, b).class, ErrorClass::Billing);
        // And as HTTP 403 with different vocabulary elsewhere.
        let b2 = r#"{"error":{"message":"Access restricted. Deposit required to unlock premium models.","code":"access_denied"}}"#;
        assert_eq!(classify(403, b2).class, ErrorClass::Billing);
    }

    #[test]
    fn degraded_function_is_temporarily_unavailable() {
        // Observed on NVIDIA (J-030): a 400 whose text says the function is
        // degraded. Not a user error, not a rate limit.
        let b = r#"{"error":{"message":"Bad Request: Function id '0a21…': DEGRADED function cannot be invoked"}}"#;
        let c = classify(400, b);
        assert_eq!(c.class, ErrorClass::TemporarilyUnavailable);
        assert!(c.class.is_retryable());
    }

    #[test]
    fn no_allowed_providers_is_temporarily_unavailable() {
        // Observed on b.ai (J-029): upstream pool excludes the account's
        // only-permitted provider; identical request succeeded earlier.
        let b = r#"{"error":{"message":"Error from provider (Console Go): Upstream request failed: [404] No allowed providers are available for the selected model."}}"#;
        let c = classify(400, b);
        assert_eq!(c.class, ErrorClass::TemporarilyUnavailable);
    }

    #[test]
    fn degraded_on_403_still_classified_by_vocabulary() {
        // Status vocabulary beats the 403 default (PermissionDenied).
        let b = r#"{"error":{"message":"DEGRADED function cannot be invoked"}}"#;
        assert_eq!(classify(403, b).class, ErrorClass::TemporarilyUnavailable);
    }

    #[test]
    fn entitlement_404_is_not_temporarily_unavailable() {
        // The NVIDIA entitlement shape must not be captured by the
        // temporary-state vocabulary.
        let b = r#"{"status":404,"title":"Not Found","detail":"Function 'abc': Not found for account 'X'"}"#;
        assert_eq!(classify(404, b).class, ErrorClass::NotEntitled);
    }

    #[test]
    fn mines_output_ceiling_from_range() {
        let l = mine_limits("field MaxTokens invalid, should be in [1, 65536]");
        assert!(
            l.iter()
                .any(|x| x.value == 65536 && x.kind == LimitKind::MaxOutputTokens)
        );
    }

    #[test]
    fn mines_context_from_sentence() {
        let l = mine_limits("maximum context length is 8192 tokens");
        assert!(
            l.iter()
                .any(|x| x.value == 8192 && x.kind == LimitKind::ContextWindow)
        );
    }

    #[test]
    fn mines_at_most_phrase() {
        let l = mine_limits("Please reduce the length: at most 4096 tokens");
        assert!(l.iter().any(|x| x.value == 4096));
    }

    #[test]
    fn no_limits_in_plain_text() {
        assert!(mine_limits("internal error").is_empty());
    }

    /// Guards the envelope inventory: every declared shape must actually be
    /// reachable by `extract_message`, otherwise the list is lying.
    #[test]
    fn declared_envelope_shapes_are_reachable() {
        let samples = [
            (404, r#"{"error":{"message":"no"}}"#, "openai.error"),
            (
                400,
                r#"{"object":"error","message":"no"}"#,
                "object.message",
            ),
            (
                404,
                r#"{"status":404,"title":"Not Found","detail":"no"}"#,
                "rfc7807",
            ),
            (404, r#"{"detail":"no"}"#, "bare.message"),
            (404, "404 page not found", "text"),
            (429, "", "empty"),
        ];
        let mut seen: Vec<&str> = Vec::new();
        for (status, body, _) in samples {
            let (_, env) = extract_message(status, body);
            seen.push(env);
        }
        for e in ENVELOPES {
            assert!(
                seen.contains(&e),
                "envelope {e} is declared but never produced"
            );
        }
    }
}
