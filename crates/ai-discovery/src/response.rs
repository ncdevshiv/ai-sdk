//! Generic response-shape normalization.
//!
//! This module exists because *the same logical thing is spelled differently
//! by every gateway*, and because **the presence of a key does not imply the
//! presence of a value**.
//!
//! Observed across the three test providers:
//!
//! | Provider/model                | Answer field | Reasoning field(s)                          |
//! |-------------------------------|--------------|---------------------------------------------|
//! | b.ai `deepseek-v4-flash`      | `content`    | `reasoning_content`                         |
//! | b.ai `mimo-v2.5`              | `content`    | `reasoning`, `reasoning_details`, `refusal` |
//! | SenseNova `6.7-flash-lite`    | **absent**   | `reasoning`                                 |
//! | NVIDIA (most models)          | `content`    | `reasoning_content` and/or `reasoning`      |
//!
//! Worse, NVIDIA echoes the *full* optional field set — `annotations`,
//! `audio`, `function_call`, `reasoning`, `refusal`, `tool_calls` — as
//! explicit `null`s on models that support none of them. Any capability
//! inference that keys off "does `reasoning` exist?" therefore reports
//! reasoning support for models that never reason.
//!
//! So normalization is value-driven: a field counts only when it carries a
//! non-empty value.

use serde_json::Value;

/// How a given field on a message object was interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldRole {
    /// The model's answer text.
    Answer,
    /// Chain-of-thought / reasoning text.
    Reasoning,
    /// A refusal message.
    Refusal,
    /// Tool/function calls.
    ToolCalls,
    /// Audio output.
    Audio,
    /// Present but null/empty — carries no capability signal.
    Empty,
    /// Not a recognized role.
    Other,
}

/// A normalized chat-completion message.
#[derive(Debug, Clone, Default)]
pub struct NormalizedMessage {
    /// The answer text, if the model produced any.
    pub answer: Option<String>,
    /// Concatenated reasoning text, if any.
    pub reasoning: Option<String>,
    /// Which field(s) supplied the reasoning text.
    pub reasoning_fields: Vec<String>,
    /// Refusal text, if the model refused.
    pub refusal: Option<String>,
    /// Tool calls, if any.
    pub tool_calls: Option<Value>,
    /// Whether audio content was present and non-empty.
    pub has_audio: bool,
    /// Role of every key on the message object, for traceability.
    pub field_roles: Vec<(String, FieldRole)>,
    /// Keys present with a non-empty value.
    pub populated_keys: Vec<String>,
}

impl NormalizedMessage {
    /// True when the model produced no answer text.
    ///
    /// This is the single most important signal in the whole crate: on
    /// reasoning-first models the entire completion budget can be consumed
    /// by chain-of-thought, producing HTTP 200 with no answer at all.
    pub fn answer_is_missing(&self) -> bool {
        self.answer
            .as_deref()
            .map(|a| a.trim().is_empty())
            .unwrap_or(true)
    }
}

/// Substrings that mark a field as carrying chain-of-thought.
const REASONING_TOKENS: &[&str] = &["reason", "think", "cot"];

/// Normalizes the `choices[i].message` object of a chat completion.
pub fn normalize_message(message: &Value) -> NormalizedMessage {
    let mut out = NormalizedMessage::default();
    let obj = match message.as_object() {
        Some(o) => o,
        None => return out,
    };

    // Pass 1: classify every key by name AND value.
    for (k, v) in obj {
        let role = classify_field(k, v);
        if !matches!(role, FieldRole::Empty) {
            out.populated_keys.push(k.clone());
        }
        out.field_roles.push((k.clone(), role));
    }

    // Pass 2: extract the answer.
    // Preferred field is `content`, but fall back to any populated
    // non-reasoning string field so gateways that omit `content` still
    // yield their output.
    if let Some(c) = value_to_text(obj.get("content")) {
        out.answer = Some(c);
    } else {
        // The filter must classify the *actual* value. It previously passed
        // `Value::Null`, which `classify_field` short-circuits to
        // `FieldRole::Empty` — so the predicate was never true, the fallback
        // never ran, and every gateway that spells its answer field anything
        // other than `content` was reported as having produced no answer.
        out.answer = obj
            .iter()
            .filter(|(k, v)| classify_field(k, v) == FieldRole::Other)
            .find_map(|(k, v)| {
                (k != "role" && !is_reasoning_key(k) && !is_reserved_key(k))
                    .then(|| value_to_text(Some(v)))
                    .flatten()
            });
    }

    // Pass 3: extract reasoning from every reasoning-named field that is
    // actually populated.
    for (k, v) in obj {
        if is_reasoning_key(k) {
            if let Some(t) = value_to_text(Some(v)) {
                out.reasoning_fields.push(k.clone());
                out.reasoning = Some(match out.reasoning.take() {
                    Some(prev) => format!("{prev}\n{t}"),
                    None => t,
                });
            }
        }
    }

    if let Some(r) = obj.get("refusal").and_then(|v| value_to_text(Some(v))) {
        out.refusal = Some(r);
    }

    if let Some(tc) = obj.get("tool_calls") {
        if !tc.is_null() && tc != &Value::Array(vec![]) {
            out.tool_calls = Some(tc.clone());
        }
    }

    out.has_audio = obj
        .get("audio")
        .map(|a| !a.is_null() && value_to_text(Some(a)).is_some())
        .unwrap_or(false);

    out
}

fn is_reasoning_key(k: &str) -> bool {
    let lower = k.to_ascii_lowercase();
    REASONING_TOKENS.iter().any(|t| lower.contains(t))
}

/// Keys that are structural, not content.
fn is_reserved_key(k: &str) -> bool {
    matches!(
        k.to_ascii_lowercase().as_str(),
        "role" | "name" | "annotations" | "audio" | "function_call" | "tool_calls" | "refusal"
    )
}

/// Classifies a field by name and value.
///
/// A field whose value is `null`, `""`, `[]` or `{}` is [`FieldRole::Empty`]
/// regardless of its name — this is what stops NVIDIA's echoed nulls from
/// being read as capabilities.
pub fn classify_field(key: &str, value: &Value) -> FieldRole {
    // An empty string is just as absent as a null: gateways that always echo
    // the field set emit `content: ""` on models that produced no text.
    if value.is_null()
        || value == &Value::Array(vec![])
        || value == &Value::Object(Default::default())
        || matches!(value, Value::String(s) if s.trim().is_empty())
    {
        return FieldRole::Empty;
    }
    let lower = key.to_ascii_lowercase();
    if is_reasoning_key(key) {
        return FieldRole::Reasoning;
    }
    match lower.as_str() {
        "content" | "text" | "message" => FieldRole::Answer,
        "refusal" => FieldRole::Refusal,
        "tool_calls" | "function_call" => FieldRole::ToolCalls,
        "audio" => FieldRole::Audio,
        _ => FieldRole::Other,
    }
}

/// Converts a JSON value to text.
///
/// Handles the three shapes `content` takes in the wild: a plain string,
/// an array of content parts (`[{"type":"text","text":…}]`), and an object.
pub fn value_to_text(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::Null => None,
        Value::String(s) if !s.trim().is_empty() => Some(s.clone()),
        Value::String(_) => None,
        Value::Array(parts) => {
            // OpenAI multi-part content: concatenate the text parts.
            let mut buf = String::new();
            for p in parts {
                match p {
                    Value::String(s) => buf.push_str(s),
                    Value::Object(_) => {
                        if let Some(t) = p.get("text").and_then(|t| t.as_str()) {
                            buf.push_str(t);
                        } else if let Some(t) = p.get("content").and_then(|t| t.as_str()) {
                            buf.push_str(t);
                        }
                    }
                    _ => {}
                }
            }
            (!buf.trim().is_empty()).then_some(buf)
        }
        Value::Object(o) => o
            .get("text")
            .or_else(|| o.get("content"))
            .and_then(|t| t.as_str())
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.to_string()),
        other => {
            let s = other.to_string();
            (!s.is_empty()).then_some(s)
        }
    }
}

/// Extracts token usage from a completion, tolerating absent/shaped usage.
#[derive(Debug, Clone, Default)]
pub struct NormalizedUsage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    /// `completion_tokens_details.reasoning_tokens`, when reported.
    pub reasoning_tokens: Option<u64>,
}

pub fn normalize_usage(body: &Value) -> NormalizedUsage {
    let u = body.get("usage");
    let u = match u {
        Some(Value::Object(_)) => u.unwrap(),
        _ => return NormalizedUsage::default(),
    };
    NormalizedUsage {
        prompt_tokens: u.get("prompt_tokens").and_then(|v| v.as_u64()),
        completion_tokens: u.get("completion_tokens").and_then(|v| v.as_u64()),
        reasoning_tokens: u
            .get("completion_tokens_details")
            .and_then(|d| d.get("reasoning_tokens"))
            .and_then(|v| v.as_u64())
            .or_else(|| {
                u.get("completion_tokens_details")
                    .and_then(|d| d.get("reasoning"))
                    .and_then(|v| v.as_u64())
            }),
    }
}

/// Diagnoses why a 200 OK response carried no usable answer.
///
/// Root-cause classification for the "empty output" failure mode. Each
/// variant corresponds to a distinct remedy, which is why they are separated
/// rather than collapsed into one boolean.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EmptyAnswerCause {
    /// The model reasoned and hit the token cap before emitting an answer.
    /// Remedy: raise `max_tokens` or disable thinking.
    BudgetConsumedByReasoning,
    /// `finish_reason` is `length` but no reasoning was reported.
    /// Remedy: raise `max_tokens`.
    BudgetTooSmall,
    /// The model stopped normally and simply returned nothing.
    EmptyByStop,
    /// The answer is present — not empty.
    NotEmpty,
}

/// Determines why a message has no answer text.
pub fn diagnose_empty(
    message: &NormalizedMessage,
    finish_reason: Option<&str>,
    usage: &NormalizedUsage,
) -> EmptyAnswerCause {
    if !message.answer_is_missing() {
        return EmptyAnswerCause::NotEmpty;
    }
    let has_reasoning = message.reasoning.is_some();
    let rtok = usage.reasoning_tokens.unwrap_or(0);
    let ctok = usage.completion_tokens.unwrap_or(0);

    if has_reasoning && (ctok == 0 || rtok >= ctok) {
        EmptyAnswerCause::BudgetConsumedByReasoning
    } else if finish_reason == Some("length") {
        EmptyAnswerCause::BudgetTooSmall
    } else {
        EmptyAnswerCause::EmptyByStop
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reads_plain_content() {
        let m = json!({"role":"assistant","content":"hello","reasoning_content":"hmm"});
        let n = normalize_message(&m);
        assert_eq!(n.answer.as_deref(), Some("hello"));
        assert_eq!(n.reasoning.as_deref(), Some("hmm"));
        assert_eq!(n.reasoning_fields, vec!["reasoning_content"]);
    }

    #[test]
    fn allen_null_fields_are_not_capabilities() {
        // NVIDIA echoes the full optional field set as nulls.
        let m = json!({
            "role":"assistant","content":"hi","annotations":null,"audio":null,
            "function_call":null,"reasoning":null,"refusal":null,"tool_calls":null
        });
        let n = normalize_message(&m);
        assert!(n.reasoning.is_none());
        assert!(!n.has_audio);
        assert!(n.tool_calls.is_none());
        // Only role + content are populated.
        assert_eq!(n.populated_keys, vec!["content", "role"]);
    }

    #[test]
    fn handles_missing_content_with_reasoning_only() {
        // SenseNova returns `reasoning` and no `content` at all.
        let m = json!({"role":"assistant","reasoning":"Thinking Process: ..."});
        let n = normalize_message(&m);
        assert!(n.answer_is_missing());
        assert!(n.reasoning.is_some());
        assert_eq!(n.reasoning_fields, vec!["reasoning"]);
    }

    #[test]
    fn handles_multiple_reasoning_fields() {
        let m = json!({"role":"assistant","content":"a","reasoning":"r1","reasoning_details":"r2"});
        let n = normalize_message(&m);
        assert_eq!(n.reasoning_fields.len(), 2);
        assert!(n.reasoning.unwrap().contains("r1"));
    }

    #[test]
    fn content_as_parts_array() {
        let m = json!({"role":"assistant","content":[{"type":"text","text":"part1"},{"type":"text","text":"part2"}]});
        let n = normalize_message(&m);
        assert_eq!(n.answer.as_deref(), Some("part1part2"));
    }

    #[test]
    fn empty_content_string_is_empty_role() {
        let m = json!({"role":"assistant","content":""});
        let n = normalize_message(&m);
        assert!(n.answer_is_missing());
        assert!(
            n.field_roles
                .iter()
                .any(|(k, r)| k == "content" && *r == FieldRole::Empty)
        );
    }

    #[test]
    fn diagnoses_reasoning_saturation() {
        let m = json!({"role":"assistant","reasoning":"..."});
        let n = normalize_message(&m);
        let u = NormalizedUsage {
            prompt_tokens: Some(84),
            completion_tokens: Some(64),
            reasoning_tokens: Some(64),
        };
        assert_eq!(
            diagnose_empty(&n, Some("length"), &u),
            EmptyAnswerCause::BudgetConsumedByReasoning
        );
    }

    #[test]
    fn diagnoses_plain_budget_exhaustion() {
        let m = json!({"role":"assistant","content":""});
        let n = normalize_message(&m);
        let u = NormalizedUsage {
            prompt_tokens: Some(10),
            completion_tokens: Some(32),
            reasoning_tokens: Some(0),
        };
        assert_eq!(
            diagnose_empty(&n, Some("length"), &u),
            EmptyAnswerCause::BudgetTooSmall
        );
    }

    #[test]
    fn usage_tolerates_absent_block() {
        assert_eq!(normalize_usage(&json!({})).completion_tokens, None);
    }
}
