//! Generic extraction of declared metadata from arbitrary provider payloads.
//!
//! OpenAPI-compatible gateways disagree completely on how to spell the same
//! concept. Observed in the wild:
//!
//! | Concept      | Spellings seen                                                        |
//! |--------------|-----------------------------------------------------------------------|
//! | context      | `context_length`, `context_window`, `max_input_tokens`, `n_ctx`…       |
//! | max output   | `max_output_length`, `max_output_tokens`, `max_completion_tokens`…     |
//! | input mods   | `input_modalities`, `modalities`, `inputModalities`…                   |
//! | features     | `supported_features`, `capabilities`, `features`…                      |
//!
//! Rather than special-casing providers, this module holds a **concept →
//! synonym list** registry and a recursive scanner that walks any JSON tree
//! looking for those concepts. The scanner records the JSON path of every
//! hit so each value is traceable.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::provenance::Fact;

/// The set of things a gateway might tell us about a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Concept {
    /// Total context window in tokens.
    ContextWindow,
    /// Maximum generated tokens.
    MaxOutputTokens,
    /// Input modalities.
    InputModalities,
    /// Output modalities.
    OutputModalities,
    /// Feature/capability flags.
    Features,
    /// Whether vision/image input is supported.
    Vision,
    /// Whether tool/function calling is supported.
    Tools,
    /// Whether streaming is supported.
    Streaming,
    /// Whether structured/JSON output is supported.
    StructuredOutput,
    /// Whether the model produces embeddings.
    Embeddings,
    /// Whether the model supports a reasoning/thinking toggle.
    Reasoning,
    /// Whether the model can be fine-tuned.
    FineTuning,
    /// Human-readable display name.
    Name,
    /// Description text.
    Description,
    /// Creation timestamp.
    Created,
    /// Owning organisation.
    Owner,
    /// Token pricing.
    Pricing,
}

/// Ordered synonym list for each concept.
///
/// Order matters: earlier entries are preferred, because in a single payload
/// the most specific spelling should win over a generic one.
pub fn synonyms(concept: Concept) -> &'static [&'static str] {
    match concept {
        Concept::ContextWindow => &[
            "context_length",
            "context_window",
            "contextlength",
            "contextwindow",
            "max_context_length",
            "max_input_tokens",
            "input_token_limit",
            "max_model_len",
            "max_prompt_tokens",
            "n_ctx",
            "num_ctx",
            "max_seq_len",
            "sequence_length",
            "total_max_tokens",
        ],
        Concept::MaxOutputTokens => &[
            "max_output_length",
            "max_output_tokens",
            "maxoutputtokens",
            "max_completion_tokens",
            "max_tokens",
            "output_token_limit",
            "max_new_tokens",
            "max_response_tokens",
            "max_generation_tokens",
        ],
        Concept::InputModalities => &[
            "input_modalities",
            "inputmodalities",
            "input_modalities_list",
            "modalities",
            "supported_modalities",
            "input_types",
            "input",
            "supported_input_modalities",
        ],
        Concept::OutputModalities => &[
            "output_modalities",
            "outputmodalities",
            "output_types",
            "output",
            "supported_output_modalities",
        ],
        Concept::Features => &[
            "supported_features",
            "features",
            "capabilities",
            "supported_capabilities",
            "support",
            "supported",
            "model_features",
        ],
        Concept::Vision => &[
            "supports_vision",
            "supportsvision",
            "vision",
            "vision_enabled",
            "image_input",
            "supports_image",
            "multimodal",
        ],
        Concept::Tools => &[
            "supports_tools",
            "supports_tools",
            "tools",
            "supports_function_calling",
            "function_calling",
            "tool_use",
            "supports_tool_use",
        ],
        Concept::Streaming => &["supports_streaming", "streaming", "stream"],
        Concept::StructuredOutput => &[
            "supports_structured_output",
            "structured_output",
            "json_mode",
            "supports_json_mode",
            "response_format",
        ],
        Concept::Embeddings => &["supports_embeddings", "embeddings", "embedding"],
        Concept::Reasoning => &[
            "supports_reasoning",
            "reasoning",
            "thinking",
            "supports_thinking",
            "reasoning_effort",
        ],
        Concept::FineTuning => &["supports_fine_tuning", "fine_tuning", "finetune"],
        Concept::Name => &["display_name", "name", "model_name", "title", "id"],
        Concept::Description => &["description", "summary", "about"],
        Concept::Created => &["created", "created_at", "createdat"],
        Concept::Owner => &[
            "owned_by",
            "ownedby",
            "owner",
            "provider",
            "vendor",
            "organization",
        ],
        Concept::Pricing => &["pricing", "price", "cost", "rates"],
    }
}

/// Normalizes a JSON key for comparison: lowercase, strip non-alphanumerics.
///
/// `context_length`, `contextLength` and `context-length` all normalize to
/// `contextlength`, so the synonym table only needs one spelling per concept.
pub fn normalize_key(key: &str) -> String {
    key.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// One hit from the scanner: the JSON path and the raw value.
#[derive(Debug, Clone)]
pub struct Hit<'a> {
    /// JSON path, e.g. `$.capabilities.context_length`.
    pub path: String,
    /// The value found at that path.
    pub value: &'a Value,
}

/// Recursively walks `root`, collecting every location whose normalized key
/// matches one of `concept`'s synonyms.
///
/// Depth is capped so a pathological payload cannot blow the stack. Numeric
/// string values (`"262144"`) are accepted by callers via
/// [`Hit::as_u64`] rather than being rejected here.
pub fn scan_concept<'a>(root: &'a Value, concept: Concept) -> Vec<Hit<'a>> {
    let wanted: Vec<String> = synonyms(concept).iter().map(|s| normalize_key(s)).collect();
    let mut out = Vec::new();
    walk(root, "$", &wanted, 0, &mut out);
    out
}

/// Hits ordered by **semantic preference**, not traversal order.
///
/// The synonym list is ordered on purpose ("earlier entries are preferred;
/// in a single payload the most specific spelling should win over a generic
/// one"), but traversal order is an accident of how the gateway serialised
/// the payload. When a payload spells a concept twice — observed:
/// `{"max_tokens": 100, "max_output_tokens": 8192}` — the higher-rank
/// synonym must win regardless of which key the gateway listed first.
/// Ties break toward shallower paths.
fn ranked_hits<'a>(root: &'a Value, concept: Concept) -> Vec<Hit<'a>> {
    let mut hits = scan_concept(root, concept);
    hits.sort_by_key(|h| (synonym_rank(&h.path, concept), h.path.len()));
    hits
}

fn synonym_rank(path: &str, concept: Concept) -> usize {
    let key = path.rsplit('.').next().unwrap_or("");
    let n = normalize_key(key);
    synonyms(concept)
        .iter()
        .position(|s| normalize_key(s) == n)
        .unwrap_or(usize::MAX)
}

const MAX_DEPTH: usize = 8;

fn walk<'a>(node: &'a Value, path: &str, wanted: &[String], depth: usize, out: &mut Vec<Hit<'a>>) {
    if depth > MAX_DEPTH {
        return;
    }
    match node {
        Value::Object(map) => {
            for (k, v) in map {
                let child = format!("{path}.{k}");
                if wanted.contains(&normalize_key(k)) {
                    out.push(Hit {
                        path: child.clone(),
                        value: v,
                    });
                }
                walk(v, &child, wanted, depth + 1, out);
            }
        }
        Value::Array(items) => {
            for (i, v) in items.iter().enumerate() {
                walk(v, &format!("{path}[{i}]"), wanted, depth + 1, out);
            }
        }
        _ => {}
    }
}

impl<'a> Hit<'a> {
    /// Reads the hit as a u64, accepting JSON numbers and numeric strings.
    ///
    /// Gateways are inconsistent about quoting large integers; a bare
    /// `f64` is also accepted because some emit `262144.0`.
    pub fn as_u64(&self) -> Option<u64> {
        match self.value {
            Value::Number(n) => {
                if let Some(u) = n.as_u64() {
                    return Some(u);
                }
                n.as_f64()
                    .filter(|f| f.is_finite() && *f >= 0.0 && *f <= u64::MAX as f64)
                    .map(|f| f as u64)
            }
            Value::String(s) => s.trim().replace(['_', ','], "").parse::<u64>().ok(),
            _ => None,
        }
    }

    /// Reads the hit as a bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self.value {
            Value::Bool(b) => Some(*b),
            Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
                "true" | "yes" | "1" | "supported" | "enabled" => Some(true),
                "false" | "no" | "0" | "unsupported" | "disabled" => Some(false),
                _ => None,
            },
            Value::Number(n) => Some(n.as_f64().unwrap_or(0.0) != 0.0),
            _ => None,
        }
    }

    /// Reads the hit as a list of strings, flattening string/array shapes.
    ///
    /// Accepts both `["text","image"]` and `"text, image"`.
    pub fn as_strings(&self) -> Option<Vec<String>> {
        match self.value {
            Value::Array(items) => Some(
                items
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect(),
            ),
            Value::String(s) => Some(
                s.split([',', '|', ' ', ';'])
                    .map(|p| p.trim().to_string())
                    .filter(|p| !p.is_empty())
                    .collect(),
            ),
            _ => None,
        }
    }

    /// Reads the hit as a string.
    pub fn as_str(&self) -> Option<&str> {
        self.value.as_str()
    }
}

/// The first hit for `concept` that yields a u64, as a provenance fact.
pub fn first_u64(entry: &Value, concept: Concept) -> Option<Fact<u64>> {
    ranked_hits(entry, concept)
        .into_iter()
        .find_map(|h| h.as_u64().map(|v| Fact::declared(v, h.path)))
}

/// The first hit for `concept` that yields a bool, as a provenance fact.
pub fn first_bool(entry: &Value, concept: Concept) -> Option<Fact<bool>> {
    ranked_hits(entry, concept)
        .into_iter()
        .find_map(|h| h.as_bool().map(|v| Fact::declared(v, h.path)))
}

/// All string-list hits for `concept`, merged and de-duplicated.
pub fn all_strings(entry: &Value, concept: Concept) -> Option<(Vec<String>, String)> {
    let mut merged: Vec<String> = Vec::new();
    let mut origin = String::new();
    for h in scan_concept(entry, concept) {
        if let Some(items) = h.as_strings() {
            if origin.is_empty() {
                origin = h.path.clone();
            }
            for i in items {
                let lower = i.to_ascii_lowercase();
                if !merged.iter().any(|m| m.to_ascii_lowercase() == lower) {
                    merged.push(i);
                }
            }
        }
    }
    if merged.is_empty() || origin.is_empty() {
        None
    } else {
        Some((merged, origin))
    }
}

/// The first string hit for `concept`, with its path.
pub fn first_str(entry: &Value, concept: Concept) -> Option<(String, String)> {
    ranked_hits(entry, concept).into_iter().find_map(|h| {
        let path = h.path.clone();
        h.as_str().map(|s| (s.to_string(), path))
    })
}

/// Whether a feature token appears in any feature-like field.
///
/// Feature lists are spelled inconsistently (`tools` vs `tool_use` vs
/// `function_calling`), so matching is done on normalized substrings.
///
/// Returns `Some` **only on a positive hit**. A feature list that merely omits
/// the token is silence, not a declaration that the capability is absent —
/// returning `false` there made every gateway that publishes a feature list
/// appear to have declared `supports_vision=false`, which the reconciler then
/// reported as a contradiction whenever a probe succeeded. An explicit
/// `supports_vision: false` field is still honoured, via [`first_bool`].
pub fn has_feature(entry: &Value, token: &str) -> Option<Fact<bool>> {
    let needle = normalize_key(token);
    if needle.is_empty() {
        return None;
    }
    let (list, path) = all_strings(entry, Concept::Features)?;
    let hit = list.iter().any(|f| normalize_key(f).contains(&needle));
    if hit {
        Some(Fact::declared(true, path))
    } else {
        None
    }
}

/// Flattens every scalar in the entry into `dotted.path -> string` form.
///
/// Used to preserve the provider's raw metadata verbatim so a reader can see
/// everything the gateway said, including fields this crate does not model.
pub fn flatten(entry: &Value) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    flatten_into(entry, "$", &mut out);
    out
}

fn flatten_into(node: &Value, path: &str, out: &mut BTreeMap<String, String>) {
    match node {
        Value::Object(map) => {
            for (k, v) in map {
                flatten_into(v, &format!("{path}.{k}"), out);
            }
        }
        Value::Array(items) => {
            for (i, v) in items.iter().enumerate() {
                flatten_into(v, &format!("{path}[{i}]"), out);
            }
        }
        Value::Null => {}
        other => {
            let text = match other {
                Value::String(s) => s.clone(),
                _ => other.to_string(),
            };
            out.insert(path.to_string(), text);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn normalize_key_is_spelling_agnostic() {
        assert_eq!(normalize_key("context_length"), "contextlength");
        assert_eq!(normalize_key("contextLength"), "contextlength");
        assert_eq!(normalize_key("context-length"), "contextlength");
    }

    #[test]
    fn finds_context_at_top_level() {
        let e = json!({"id": "m", "context_length": 262144});
        let f = first_u64(&e, Concept::ContextWindow).unwrap();
        assert_eq!(f.value, 262144);
        assert_eq!(f.path.unwrap(), "$.context_length");
    }

    #[test]
    fn finds_context_when_nested_under_capabilities() {
        let e = json!({"id": "m", "capabilities": {"context_window": 8192}});
        let f = first_u64(&e, Concept::ContextWindow).unwrap();
        assert_eq!(f.value, 8192);
        assert_eq!(f.path.unwrap(), "$.capabilities.context_window");
    }

    #[test]
    fn accepts_numeric_strings() {
        let e = json!({"id": "m", "max_output_length": "65536"});
        let f = first_u64(&e, Concept::MaxOutputTokens).unwrap();
        assert_eq!(f.value, 65536);
    }

    #[test]
    fn synonym_priority_beats_traversal_order() {
        // The gateway listed `max_tokens` first, but `max_output_tokens` is
        // the higher-rank synonym — it must win.
        let e = json!({"id": "m", "max_tokens": 100, "max_output_tokens": 8192});
        let f = first_u64(&e, Concept::MaxOutputTokens).unwrap();
        assert_eq!(f.value, 8192);
        assert_eq!(f.path.as_deref(), Some("$.max_output_tokens"));
        // A deeper path with a higher-rank synonym beats a shallower path.
        let e2 = json!({"id": "m", "n_ctx": 4096, "caps": {"context_length": 262144}});
        let f2 = first_u64(&e2, Concept::ContextWindow).unwrap();
        assert_eq!(f2.value, 262144);
        assert_eq!(f2.path.as_deref(), Some("$.caps.context_length"));
    }

    #[test]
    fn finds_modality_lists() {
        let e = json!({"input_modalities": ["text", "image"], "output_modalities": ["text"]});
        let (i, _) = all_strings(&e, Concept::InputModalities).unwrap();
        assert_eq!(i, vec!["text", "image"]);
        let (o, _) = all_strings(&e, Concept::OutputModalities).unwrap();
        assert_eq!(o, vec!["text"]);
    }

    #[test]
    fn bare_payload_yields_nothing() {
        // Shape used by gateways that publish no metadata at all.
        let e = json!({"id": "m", "object": "model", "created": 1, "owned_by": "x"});
        assert!(first_u64(&e, Concept::ContextWindow).is_none());
        assert!(first_u64(&e, Concept::MaxOutputTokens).is_none());
        assert!(all_strings(&e, Concept::InputModalities).is_none());
    }

    #[test]
    fn has_feature_matches_normalized_tokens() {
        let e = json!({"supported_features": ["tools", "json_mode", "reasoning"]});
        assert!(has_feature(&e, "tool").unwrap().value);
        assert!(has_feature(&e, "reason").unwrap().value);
    }

    /// A feature list that omits a token is silence, not a declaration of
    /// absence. Reporting `false` here previously produced false
    /// "declared X but probe observed Y" anomalies for every gateway that
    /// publishes a feature list at all.
    #[test]
    fn has_feature_absence_is_not_a_false_declaration() {
        let e = json!({"supported_features": ["tools", "json_mode", "reasoning"]});
        assert!(
            has_feature(&e, "audio").is_none(),
            "an unlisted feature must not be reported as declared-false"
        );
    }

    /// An explicit boolean, by contrast, *is* a declaration.
    #[test]
    fn explicit_false_flag_is_honoured_as_a_declaration() {
        let e = json!({"supports_vision": false});
        let f = first_bool(&e, Concept::Vision).unwrap();
        assert!(!f.value);
    }

    #[test]
    fn flatten_preserves_raw_metadata() {
        let e = json!({"id": "m", "quantization": "fp8", "pricing": {"prompt": "0"}});
        let f = flatten(&e);
        assert_eq!(f.get("$.quantization").unwrap(), "fp8");
        // JSON strings are unwrapped, so `"0"` surfaces as `0`.
        assert_eq!(f.get("$.pricing.prompt").unwrap(), "0");
    }
}
