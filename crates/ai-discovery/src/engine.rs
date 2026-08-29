//! Discovery orchestration.
//!
//! Ties the layers together into one call:
//!
//! 1. **Enumerate** — `GET {base}/models` and read whatever envelope the
//!    gateway returns.
//! 2. **Harvest declared metadata** — concept-synonym scan of each entry.
//! 3. **Classify role by endpoint routing** — a model's *type* is defined by
//!    which endpoint accepts it, never by its name. A model that answers on
//!    `/embeddings` is an embedding model; one that answers on
//!    `/images/generations` is an image model.
//! 4. **Probe capabilities** — empirical, with declared values as priors.
//! 5. **Reconcile** — probe beats declaration, conflicts are recorded.

use std::collections::BTreeMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use ai_types::{Modality, ModelId, ProviderId};

use crate::declared;
use crate::declared::Concept;
use crate::errors::{ErrorClass, LimitKind, MinedLimit};
use crate::probe::{
    ContentShape, ThinkingSupport, Transport, TransportPolicy, probe_context_window,
    probe_max_output, probe_reachable_shapes, probe_streaming, probe_structured_output,
    probe_thinking, probe_tools, probe_vision,
};
use crate::provenance::{Fact, reconcile};
use crate::response::{EmptyAnswerCause, diagnose_empty};

/// What a model primarily does, determined by endpoint routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelRole {
    /// Answers chat/completions.
    Chat,
    /// Serves /embeddings.
    Embedding,
    /// Serves /rerank.
    Reranker,
    /// Serves /images/generations.
    ImageGeneration,
    /// Serves /audio/speech.
    Speech,
    /// Serves /audio/transcriptions.
    Transcription,
    /// Serves /videos.
    Video,
    /// Accepted by no probed endpoint.
    Unknown,
}

impl ModelRole {
    /// Whether the role is generative text (i.e. chat-compatible).
    pub fn is_chat_like(&self) -> bool {
        matches!(self, Self::Chat)
    }
}

impl std::fmt::Display for ModelRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Chat => "chat",
            Self::Embedding => "embedding",
            Self::Reranker => "reranker",
            Self::ImageGeneration => "image-generation",
            Self::Speech => "speech",
            Self::Transcription => "transcription",
            Self::Video => "video",
            Self::Unknown => "unknown",
        };
        f.write_str(s)
    }
}

/// Full discovery result for one model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredModel {
    /// Provider-scoped identifier.
    pub provider: String,
    /// Model id as the gateway spells it.
    pub id: String,
    /// Display name, when the gateway provides one.
    pub name: Option<String>,
    /// Everything the gateway said, flattened to `dotted.path -> value`.
    pub declared: BTreeMap<String, String>,
    /// Primary role, from endpoint routing.
    pub role: ModelRole,
    /// Every endpoint that accepted the model.
    pub accepted_endpoints: Vec<String>,
    /// Context window, with provenance.
    pub context_window: Fact<u64>,
    /// Maximum output tokens, with provenance.
    pub max_output_tokens: Fact<u64>,
    /// Input modalities, with provenance.
    pub input_modalities: Fact<Vec<Modality>>,
    /// Output modalities, with provenance.
    pub output_modalities: Fact<Vec<Modality>>,
    /// Whether tool/function calling works, with provenance.
    pub tools: Option<Fact<bool>>,
    /// Whether structured output works, with provenance.
    pub structured_output: Option<Fact<bool>>,
    /// Whether SSE streaming produced real frames.
    pub streaming: Option<Fact<bool>>,
    /// Reasoning/thinking control.
    pub thinking: Option<ThinkingSupport>,
    /// Whether the last reachability probe succeeded.
    pub reachable: bool,
    /// Why the model is unusable, when it is.
    pub blocker: Option<String>,
    /// Why an empty answer came back, when it did.
    pub empty_cause: Option<EmptyAnswerCause>,
    /// Latency of the reachability probe.
    pub latency: Option<Duration>,
    /// Anomalies detected during discovery, each traceable to its cause.
    pub anomalies: Vec<String>,
    /// Whether the model was listed by `/models` (false for extra probes).
    pub listed: bool,
}

impl DiscoveredModel {
    /// A short human-readable capability summary.
    pub fn summary(&self) -> String {
        let ctx = if self.context_window.value == 0 {
            "ctx=?".to_string()
        } else {
            format!("ctx={}", self.context_window.value)
        };
        let out = if self.max_output_tokens.value == 0 {
            "out=?".to_string()
        } else {
            format!("out={}", self.max_output_tokens.value)
        };
        let think = match &self.thinking {
            Some(t) if t.emits_reasoning => match &t.disable_spelling {
                Some(s) => format!("thinking(on,off via {s})"),
                None => "thinking(on,no-off-switch)".to_string(),
            },
            Some(_) => "thinking(off)".to_string(),
            None => "thinking(?)".to_string(),
        };
        let tools = match &self.tools {
            Some(f) if f.value => "tools=y",
            Some(_) => "tools=n",
            None => "tools=?",
        };
        let structured = match &self.structured_output {
            Some(f) if f.value => "json=y",
            Some(_) => "json=n",
            None => "json=?",
        };
        let mods = self
            .input_modalities
            .value
            .iter()
            .map(|m| match m {
                Modality::Text => "T",
                Modality::Image => "I",
                Modality::Audio => "A",
                Modality::Video => "V",
            })
            .collect::<String>();
        format!(
            "{} [{}] in={} {} {} {} {} {}",
            self.id, self.role, mods, ctx, out, think, tools, structured
        )
    }
}

/// Knobs controlling how aggressively to probe.
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Per-request timeout.
    pub timeout: Duration,
    /// Pacing and retry behaviour for the underlying transport.
    pub transport_policy: TransportPolicy,
    /// Attempt the image-input probe.
    pub probe_vision: bool,
    /// Attempt the tool-calling probe.
    pub probe_tools: bool,
    /// Attempt structured-output probes.
    pub probe_structured_output: bool,
    /// Attempt endpoint-routing probes (embeddings/rerank/images/audio).
    pub probe_endpoints: bool,
    /// Attempt the reasoning-toggle battery.
    pub probe_thinking: bool,
    /// Binary-search the context window.
    pub probe_context: bool,
    /// Upper bound for the context binary search.
    pub max_context_probe: usize,
    /// Binary-search rounds.
    pub context_rounds: usize,
    /// Max models to probe in one run (0 = all).
    pub limit: usize,
    /// Extra model ids to test even if not listed.
    pub extra_models: Vec<String>,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(60),
            transport_policy: TransportPolicy::default(),
            probe_vision: true,
            probe_tools: true,
            probe_structured_output: true,
            probe_endpoints: true,
            probe_thinking: true,
            probe_context: false,
            max_context_probe: 128_000,
            context_rounds: 6,
            limit: 0,
            extra_models: Vec::new(),
        }
    }
}

/// The discovery engine.
pub struct DiscoveryEngine {
    transport: Transport,
    provider: String,
}

impl DiscoveryEngine {
    pub fn new(
        provider: impl Into<String>,
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        timeout: Duration,
    ) -> Result<Self, crate::DiscoveryError> {
        let transport = Transport::new(base_url, api_key, timeout)?;
        Ok(Self {
            transport,
            provider: provider.into(),
        })
    }

    /// Builds an engine with an explicit pacing/retry policy.
    ///
    /// Required for gateways that throttle without advertising limits: use
    /// [`TransportPolicy::conservative`] there, or discovery will report its
    /// own throttling as model failure.
    pub fn with_policy(
        provider: impl Into<String>,
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        timeout: Duration,
        policy: TransportPolicy,
    ) -> Result<Self, crate::DiscoveryError> {
        let transport = Transport::with_policy(base_url, api_key, timeout, policy)?;
        Ok(Self {
            transport,
            provider: provider.into(),
        })
    }

    /// Enumerates the model ids exposed by the gateway.
    ///
    /// The `/models` response shape varies: `{data:[…]}` is the common case
    /// but a bare array and `{models:[…]}` both occur in the wild, so all
    /// three are accepted.
    pub async fn list_models(&self) -> Result<Vec<Value>, crate::DiscoveryError> {
        let raw = self.transport.get("models").await;
        if !raw.is_success() {
            let e = raw
                .error()
                .unwrap_or_else(|| crate::errors::classify(raw.status, &raw.body));
            return Err(crate::DiscoveryError::ListFailed {
                status: raw.status,
                message: e.message,
            });
        }
        let value: Value =
            serde_json::from_str(&raw.body).map_err(|e| crate::DiscoveryError::ListFailed {
                status: raw.status,
                message: format!("models response is not JSON: {e}"),
            })?;

        let entries = value
            .get("data")
            .and_then(|d| d.as_array())
            .or_else(|| value.get("models").and_then(|d| d.as_array()))
            .or_else(|| value.as_array())
            .cloned()
            .unwrap_or_default();

        Ok(entries)
    }

    /// Runs full discovery over every listed model (plus any extras).
    ///
    /// Returns [`crate::DiscoveryError::ListFailed`] when the gateway's
    /// `/models` cannot be read — a failed listing must never surface as
    /// "zero models", because that is an untraceable outcome: a wrong API
    /// key and an empty catalog would look identical.
    pub async fn discover(
        &self,
        config: &DiscoveryConfig,
    ) -> Result<Vec<DiscoveredModel>, crate::DiscoveryError> {
        let entries = self.list_models().await?;
        let mut ids: Vec<(String, Option<Value>, bool)> = Vec::new();

        for e in entries {
            if let Some(id) = e.get("id").and_then(|i| i.as_str()) {
                ids.push((id.to_string(), Some(e), true));
            }
        }
        for extra in &config.extra_models {
            if !ids.iter().any(|(i, _, _)| i == extra) {
                ids.push((extra.clone(), None, false));
            }
        }

        let total = match config.limit {
            0 => ids.len(),
            n => ids.len().min(n),
        };

        let mut out = Vec::with_capacity(total);
        for (id, entry, listed) in ids.into_iter().take(total) {
            out.push(self.discover_one(&id, entry.as_ref(), listed, config).await);
        }
        Ok(out)
    }

    /// Discovers one model.
    pub async fn discover_one(
        &self,
        id: &str,
        entry: Option<&Value>,
        listed: bool,
        config: &DiscoveryConfig,
    ) -> DiscoveredModel {
        let mut anomalies = Vec::new();

        // ---- Layer 1: declared metadata -------------------------------------
        let declared_map = entry.map(declared::flatten).unwrap_or_default();
        let declared_ctx = entry.and_then(|e| declared::first_u64(e, Concept::ContextWindow));
        let declared_out = entry.and_then(|e| declared::first_u64(e, Concept::MaxOutputTokens));
        let declared_in_mods =
            entry.and_then(|e| declared::all_strings(e, Concept::InputModalities));
        let declared_out_mods =
            entry.and_then(|e| declared::all_strings(e, Concept::OutputModalities));
        let declared_vision = entry.and_then(|e| {
            declared::first_bool(e, Concept::Vision)
                .or_else(|| declared::has_feature(e, "vision"))
                .or_else(|| declared::has_feature(e, "image"))
        });
        let declared_name = entry
            .and_then(|e| declared::first_str(e, Concept::Name))
            .map(|(s, _)| s);

        // ---- Layer 2: reachability -----------------------------------------
        // Shape-aware: a plain-text probe alone would report every
        // vision-only model as unreachable.
        let reach = probe_reachable_shapes(&self.transport, id, 64).await;

        // A 2xx with no usable message is a distinct failure: the gateway
        // accepted the request but the payload shape was unusable (missing
        // `choices`, an empty `choices` array, or no message object).
        if reach.reachable && reach.message.is_none() {
            anomalies
                .push("HTTP 200 but no usable message: choices array missing or empty".to_string());
        }

        if !reach.reachable {
            let err = reach.error.clone();
            let class = err.as_ref().map(|e| e.class);
            let mut accepted = Vec::new();
            let mut role = ModelRole::Unknown;

            // A chat rejection that says the model will not take plain text
            // is evidence of a *different input contract*, not of a broken
            // model: `nvidia/nemotron-parse` answers exactly so (J-014).
            // Recorded as an anomaly so the role stays honest (unknown)
            // while the reason is traceable.
            if let Some(e) = &err {
                if e.class == ErrorClass::BadRequest && is_non_text_rejection(&e.message) {
                    anomalies.push(format!(
                        "chat endpoint rejected plain-text input: {} — \
                         the model expects structured/content-part input; \
                         role unknown, text modality not asserted",
                        e.message
                    ));
                }
            }

            // A model unreachable on /chat/completions may still be served on
            // a modality-specific endpoint. Route-discover before declaring
            // it broken.
            if config.probe_endpoints && !matches!(class, Some(ErrorClass::RateLimited)) {
                role = self.route_discover(id, &mut accepted, &mut anomalies).await;
            }

            // When routing found an endpoint that *does* serve this model,
            // the model is not broken — it is simply not a chat model.
            // Emitting "not served by this gateway" here would flatly
            // contradict the `accepted_endpoints` we just populated:
            // `nvidia/nemotron-3-embed-1b` was reported unreachable while
            // `accepted_endpoints` read `["embeddings"]`.
            let blocker = if !accepted.is_empty() {
                Some(format!(
                    "not available on chat/completions; served on {} instead (role: {role:?})",
                    accepted.join(", ")
                ))
            } else {
                err.as_ref().map(|e| match e.class {
                    ErrorClass::ModelNotFound => {
                        format!("not served by this gateway ({})", e.message)
                    }
                    ErrorClass::NotEntitled => {
                        format!("model exists but this account is not entitled to it — request access ({})", e.message)
                    }
                    ErrorClass::Gone => format!("model retired by the provider ({})", e.message),
                    ErrorClass::Billing => {
                        format!("account not funded for this model ({})", e.message)
                    }
                    ErrorClass::PermissionDenied => format!("not permitted ({})", e.message),
                    ErrorClass::RateLimited => {
                        "throttled during discovery — result inconclusive".to_string()
                    }
                    ErrorClass::TemporarilyUnavailable => {
                        format!("temporarily unavailable ({})", e.message)
                    }
                    ErrorClass::Timeout => format!("no response within {:?}", config.timeout),
                    other => format!("{other}: {}", e.message),
                })
            };

            return DiscoveredModel {
                provider: self.provider.clone(),
                id: id.to_string(),
                name: declared_name,
                declared: declared_map,
                role,
                accepted_endpoints: accepted,
                context_window: declared_ctx
                    .unwrap_or_else(|| Fact::unknown(0, "model unreachable; not probed")),
                max_output_tokens: declared_out
                    .unwrap_or_else(|| Fact::unknown(0, "model unreachable; not probed")),
                input_modalities: declared_in_mods
                    .map(|(v, p)| Fact::declared(modalities_from_strings(&v), p))
                    .unwrap_or_else(|| Fact::unknown(vec![Modality::Text], "not probed")),
                output_modalities: declared_out_mods
                    .map(|(v, p)| Fact::declared(modalities_from_strings(&v), p))
                    .unwrap_or_else(|| Fact::unknown(vec![Modality::Text], "not probed")),
                thinking: None,
                tools: None,
                structured_output: None,
                streaming: None,
                reachable: false,
                blocker,
                empty_cause: None,
                latency: Some(reach.elapsed),
                anomalies,
                listed,
            };
        }

        // ---- Layer 3: endpoint routing -------------------------------------
        let accepted = vec!["chat/completions".to_string()];
        let role = ModelRole::Chat;

        // A model that answered only once we sent an image part rejected
        // plain text outright. That is a stronger statement about input
        // modalities than the vision probe can make (the vision probe
        // *adds* Image; it never removes Text), so record it here.
        let text_input_rejected = reach.accepted_shape == Some(ContentShape::Image);
        if text_input_rejected {
            anomalies.push(
                "plain-text input rejected: model answered only when sent an image part \
                 (vision-only — text input is not accepted)"
                    .to_string(),
            );
        }

        // ---- Layer 4: capability probes ------------------------------------
        let mut in_modalities = if text_input_rejected {
            vec![Modality::Image]
        } else {
            vec![Modality::Text]
        };
        let mut out_modalities = vec![Modality::Text];
        let mut vision_fact: Option<Fact<bool>> = None;

        if text_input_rejected {
            // Reachability already proved image acceptance: the only request
            // that succeeded was the one carrying an image part. Re-running
            // the vision probe would send a *mixed* text+image payload,
            // which this model rejects — producing a false "no vision"
            // verdict on a model that is nothing but vision.
            vision_fact = Some(Fact::probed(
                true,
                "accepted an image content part during the reachability probe",
                0.9,
            ));
        } else if config.probe_vision {
            match probe_vision(&self.transport, id).await {
                Ok(true) => {
                    if !in_modalities.contains(&Modality::Image) {
                        in_modalities.push(Modality::Image);
                    }
                    vision_fact = Some(Fact::probed(true, "image_url content part accepted", 0.9));
                }
                Ok(false) => {
                    vision_fact = Some(Fact::probed(false, "no tool call returned", 0.7));
                }
                Err(e) => {
                    // A server-side failure on the image request is not
                    // evidence of absence: it means the request never reached
                    // a verdict (observed: llama-3.2-11b-vision-instruct, a
                    // vision model, returns 500 on image parts). Same rule as
                    // endpoint routing (J-017): 5xx = routed-but-unhealthy =
                    // inconclusive, never negative.
                    let inconclusive = matches!(
                        e.class,
                        ErrorClass::ServerError | ErrorClass::Timeout | ErrorClass::Network
                    );
                    if inconclusive {
                        vision_fact = Some(Fact::probed(
                            false,
                            format!("image probe inconclusive ({}): {}", e.class, e.message),
                            0.3,
                        ));
                        anomalies.push(format!(
                            "vision probe inconclusive: got {} on image request ({}); \
                             not evidence that image input is unsupported",
                            e.class, e.message
                        ));
                    } else {
                        vision_fact = Some(Fact::probed(
                            false,
                            format!("image_url part rejected: {}", e.message),
                            0.85,
                        ));
                    }
                }
            }
        }

        let mut tools_fact: Option<Fact<bool>> = None;
        if config.probe_tools {
            match probe_tools(&self.transport, id).await {
                Ok(v) => {
                    tools_fact = Some(Fact::probed(
                        v.supported,
                        format!(
                            "tool_calls returned in {}/{} samples (temperature=0)",
                            v.positive, v.samples
                        ),
                        v.confidence(),
                    ))
                }
                Err(e) => {
                    tools_fact = Some(Fact::probed(
                        false,
                        format!("rejected: {}", e.message),
                        0.85,
                    ))
                }
            }
        }

        let mut structured_fact: Option<Fact<bool>> = None;
        if config.probe_structured_output {
            let (obj, schema) = probe_structured_output(&self.transport, id).await;
            structured_fact = Some(Fact::probed(
                obj,
                format!("json_object={obj}, json_schema={schema}"),
                if obj && schema { 0.9 } else { 0.8 },
            ));
            if obj && !schema {
                anomalies.push(
                    "json_object accepted but json_schema rejected: schema-constrained output unsupported"
                        .to_string(),
                );
            }
        }

        let streaming = probe_streaming(&self.transport, id).await;
        if !streaming {
            anomalies.push("stream=true did not produce SSE frames".to_string());
        }

        // ---- Reasoning ------------------------------------------------------
        let thinking = if config.probe_thinking {
            let t = probe_thinking(&self.transport, id, &reach).await;
            if t.emits_reasoning && t.disable_spelling.is_none() {
                anomalies.push(format!(
                    "model always emits reasoning and no tested toggle suppressed it (tried: {})",
                    t.observations
                        .iter()
                        .map(|(l, _)| l.clone())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            Some(t)
        } else {
            None
        };

        // ---- Empty-answer diagnosis ----------------------------------------
        let usage = reach.usage.clone().unwrap_or_default();
        let empty_cause = reach
            .message
            .as_ref()
            .map(|m| diagnose_empty(m, reach.finish_reason.as_deref(), &usage));
        if let Some(cause @ EmptyAnswerCause::BudgetConsumedByReasoning) = empty_cause {
            anomalies.push(format!(
                "HTTP 200 with no answer: {cause:?} — prompt_tokens={:?} completion_tokens={:?} reasoning_tokens={:?}",
                usage.prompt_tokens, usage.completion_tokens, usage.reasoning_tokens
            ));
        }

        // ---- Limits ---------------------------------------------------------
        let mut max_out = declared_out.clone();
        if let Some((v, ev)) = probe_max_output(&self.transport, id).await {
            let probed = Fact::inferred(v, format!("mined from rejection: {ev}"), 0.9);
            if let Some(d) = &declared_out {
                if d.value != v {
                    anomalies.push(format!(
                        "declared max_output={} at {} but gateway enforces {}",
                        d.value,
                        d.path.as_deref().unwrap_or("?"),
                        v
                    ));
                }
            }
            max_out = Some(reconcile(declared_out.clone(), probed));
        }

        let mut ctx = declared_ctx.clone();
        if config.probe_context {
            let (probed_ctx, ev) = probe_context_window(
                &self.transport,
                id,
                config.max_context_probe,
                config.context_rounds,
            )
            .await;
            if let Some(v) = probed_ctx {
                // Confidence must follow the quality of the search, not be a
                // constant: an aborted search is a lower bound dressed as a
                // measurement unless it is discounted here.
                let confidence = if ev.contains("SEARCH ABORTED") {
                    0.2
                } else if ev.contains("LOWER BOUND") {
                    0.4
                } else {
                    0.7
                };
                ctx = Some(Fact::probed(v, ev, confidence));
            }
        }

        // Reconcile vision: a declared claim contradicted by a probe is a
        // first-class anomaly, because it is the most common way an SDK
        // reports a capability the model does not have.
        let mut input_fact = Fact::probed(in_modalities.clone(), "probed modalities", 0.85);
        if let Some((declared_mods, path)) = &declared_in_mods {
            let dm = modalities_from_strings(declared_mods);
            let declares_image = dm.contains(&Modality::Image);
            let probed_image = in_modalities.contains(&Modality::Image);
            if declares_image && !probed_image {
                anomalies.push(format!(
                    "declared image input at {path} but image probe failed — declaration is wrong"
                ));
            }
            if !declares_image && probed_image {
                anomalies.push(format!("image input works but is not declared at {path}"));
            }
        }
        // A dedicated vision flag is a second, independent declaration; when
        // it disagrees with the probe the discrepancy is recorded too, since
        // some gateways set the flag while the modality list says otherwise.
        if let Some(dv) = &declared_vision {
            let probed_image = in_modalities.contains(&Modality::Image);
            if dv.value && !probed_image {
                anomalies.push(format!(
                    "declared supports_vision=true at {} but image probe failed",
                    dv.path.as_deref().unwrap_or("?")
                ));
            } else if !dv.value && probed_image {
                anomalies.push(format!(
                    "declared supports_vision=false at {} but image probe succeeded",
                    dv.path.as_deref().unwrap_or("?")
                ));
            }
        }
        if let Some((declared_mods, _)) = &declared_out_mods {
            out_modalities = modalities_from_strings(declared_mods);
        }
        if let Some(vf) = vision_fact {
            input_fact = input_fact.with_confidence(vf.confidence);
            input_fact.evidence = format!("{}; {}", vf.evidence, input_fact.evidence);
        }

        DiscoveredModel {
            provider: self.provider.clone(),
            id: id.to_string(),
            name: declared_name,
            declared: declared_map,
            role,
            accepted_endpoints: accepted,
            context_window: ctx.unwrap_or_else(|| {
                Fact::unknown(
                    0,
                    "gateway publishes no context metadata and probing was not enabled",
                )
            }),
            max_output_tokens: max_out.unwrap_or_else(|| {
                Fact::unknown(
                    0,
                    "gateway publishes no output-limit metadata and no rejection revealed a bound",
                )
            }),
            input_modalities: input_fact,
            output_modalities: Fact::probed(
                out_modalities,
                "probed/declared output modalities",
                0.7,
            ),
            thinking,
            tools: tools_fact,
            structured_output: structured_fact,
            streaming: Some(Fact::probed(
                streaming,
                if streaming {
                    "SSE data frames observed"
                } else {
                    "no SSE frames; stream=true silently ignored"
                },
                0.9,
            )),
            reachable: true,
            blocker: None,
            empty_cause,
            latency: Some(reach.elapsed),
            anomalies,
            listed,
        }
    }

    /// Determines a model's role by finding which endpoint routes to it.
    ///
    /// The key discriminator is **how** a minimal probe fails:
    ///
    /// * `404` / model-not-found — this endpoint does not serve the model.
    /// * `400` — the endpoint *did* route to the model and rejected our
    ///   deliberately minimal payload for missing/invalid parameters. That is
    ///   positive evidence the model lives on this endpoint: SenseNova's
    ///   image models answer `/images/generations` with a `400` enumerating
    ///   the valid sizes, while answering `/chat/completions` with a `404`.
    /// * `5xx` — routed, but the backend failed; inconclusive, so recorded
    ///   as an anomaly rather than silently treated as "not this endpoint".
    async fn route_discover(
        &self,
        id: &str,
        accepted: &mut Vec<String>,
        anomalies: &mut Vec<String>,
    ) -> ModelRole {
        let probes: Vec<(&str, ModelRole, serde_json::Value)> = vec![
            (
                "embeddings",
                ModelRole::Embedding,
                serde_json::json!({"model": id, "input": "hello"}),
            ),
            (
                "rerank",
                ModelRole::Reranker,
                serde_json::json!({"model": id, "query": "a", "documents": ["a", "b"]}),
            ),
            (
                "images/generations",
                ModelRole::ImageGeneration,
                serde_json::json!({"model": id, "prompt": "a red circle", "n": 1}),
            ),
            (
                "audio/speech",
                ModelRole::Speech,
                serde_json::json!({"model": id, "input": "hello", "voice": "alloy"}),
            ),
            (
                "videos",
                ModelRole::Video,
                serde_json::json!({"model": id, "prompt": "a red circle"}),
            ),
        ];

        for (path, role, body) in probes {
            let raw = self.transport.post(path, &body).await;
            if raw.is_success() {
                accepted.push(path.to_string());
                if role == ModelRole::Embedding {
                    let dim = raw
                        .json()
                        .and_then(|v| v.get("data").and_then(|d| d.get(0)).cloned())
                        .and_then(|d| {
                            d.get("embedding")
                                .and_then(|e| e.as_array())
                                .map(|a| a.len())
                        })
                        .unwrap_or(0);
                    if dim == 0 {
                        anomalies
                            .push("embeddings accepted but returned no vector data".to_string());
                    }
                }
                return role;
            }

            let err = match raw.error() {
                Some(e) => e,
                None => continue,
            };
            match err.class {
                // Routed to this model; our payload was merely incomplete.
                ErrorClass::BadRequest | ErrorClass::ContextTooLarge => {
                    accepted.push(format!("{path} (routed; minimal payload rejected)"));
                    return role;
                }
                // Routed, but the backend is unhealthy — remember it.
                ErrorClass::ServerError => {
                    anomalies.push(format!(
                        "{path} routes to this model but returned {} ({})",
                        err.status, err.message
                    ));
                }
                _ => {}
            }
        }

        ModelRole::Unknown
    }
}

/// Maps gateway modality strings onto the SDK's [`Modality`] type.
///
/// Matching is by **word token**, not raw substring. Substring matching read
/// `"context"` as `Text` (because "con|text" contains "text"), so a gateway
/// listing context metadata under a modality field would be credited with
/// text input. Splitting on non-alphanumerics keeps `image_url` and
/// `input_modalities`-style compound spellings working while rejecting
/// accidental substrings.
pub fn modalities_from_strings(values: &[String]) -> Vec<Modality> {
    let mut out = Vec::new();
    for v in values {
        let words: Vec<String> = v
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_ascii_lowercase())
            .collect();
        let has = |needles: &[&str]| words.iter().any(|w| needles.contains(&w.as_str()));
        let m = if has(&["image", "images", "vision", "picture", "img"]) {
            Modality::Image
        } else if has(&["audio", "speech", "sound", "voice"]) {
            Modality::Audio
        } else if has(&["video"]) {
            Modality::Video
        } else if has(&["text", "txt"]) {
            Modality::Text
        } else {
            continue;
        };
        if !out.contains(&m) {
            out.push(m);
        }
    }
    if out.is_empty() {
        out.push(Modality::Text);
    }
    out
}

/// Mines limits from arbitrary text, re-exported for callers.
pub fn mine(text: &str) -> Vec<MinedLimit> {
    crate::errors::mine_limits(text)
}

/// Whether a rejection message means "this model does not accept plain-text
/// input" rather than "this request is malformed".
///
/// Observed verbatim on NVIDIA's `nvidia/nemotron-parse` (J-014):
/// `Content cannot be a plain string. The model does not support text input.`
fn is_non_text_rejection(message: &str) -> bool {
    let m = message.to_ascii_lowercase();
    m.contains("does not support text input")
        || m.contains("text input not supported")
        || m.contains("cannot be a plain string")
}

/// Convenience alias.
pub type Limit = LimitKind;

pub use crate::probe::Reachability as ReachabilityProbe;
/// Re-exported for consumers building their own pipelines.
pub use crate::response::NormalizedMessage as Message;
pub use crate::response::NormalizedUsage as Usage;

/// Converts a discovered model into the registry's [`ai_models::ModelInfo`].
pub fn to_model_info(d: &DiscoveredModel) -> ai_models::ModelInfo {
    let provider = ProviderId::new(d.provider.clone());
    let mut info = ai_models::ModelInfo::new(
        provider,
        ModelId::new(d.id.clone()),
        d.context_window.value,
        d.max_output_tokens.value,
    )
    .with_name(d.name.clone().unwrap_or_else(|| d.id.clone()))
    .with_capabilities(ai_models::ModelCapabilities {
        input_modalities: d.input_modalities.value.clone(),
        output_modalities: d.output_modalities.value.clone(),
        // Every flag is read from a provenance fact; nothing defaults to
        // `true` on the assumption that a gateway "probably supports it".
        supports_streaming: d.streaming.as_ref().map(|f| f.value).unwrap_or(false),
        supports_tools: d.tools.as_ref().map(|f| f.value).unwrap_or(false),
        supports_structured_output: d
            .structured_output
            .as_ref()
            .map(|f| f.value)
            .unwrap_or(false),
        supports_embeddings: matches!(d.role, ModelRole::Embedding),
        supports_vision: d.input_modalities.value.contains(&Modality::Image),
        supports_fine_tuning: false,
    });
    // The discovered thinking toggle is the operational artifact: it is a
    // per-model value (J-026), so it must ride along with the model, not be
    // re-derived by each caller.
    if let Some(spelling) = d.thinking.as_ref().and_then(|t| t.disable_spelling.clone()) {
        info = info.with_thinking_control(spelling);
    }
    info
}
