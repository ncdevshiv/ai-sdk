//! # ai-discovery
//!
//! **Provider-agnostic model and capability discovery.**
//!
//! This crate answers, for any OpenAI-compatible gateway, the questions an
//! SDK needs answered before it can use a model:
//!
//! * Which models exist?
//! * What *kind* of model is each one (chat / embedding / reranker / image /
//!   audio / video)?
//! * How much context does it accept, and how much can it generate?
//! * Does it take images? Tools? JSON schemas?
//! * Does it reason, and if so **how do I turn that off?**
//!
//! ## Why discovery cannot be a lookup table
//!
//! Measured against three live gateways:
//!
//! | Gateway   | Models | Metadata in `/v1/models`                       |
//! |-----------|--------|------------------------------------------------|
//! | b.ai      | 44     | `id`, `object`, `created`, `owned_by` — **no capabilities** |
//! | NVIDIA    | 83     | `id`, `object`, `created`, `owned_by` — **no capabilities** |
//! | SenseNova | 4      | 18 fields including `context_length`, `input_modalities`, `supported_features` |
//!
//! Two of three publish nothing. The one that publishes metadata publishes
//! **incorrect** metadata: SenseNova declares image input and tool support
//! for models where both fail. So neither trusting a catalog nor trusting
//! the endpoint is sound.
//!
//! ## The layered model
//!
//! Discovery therefore combines three evidence sources, and every reported
//! fact records which one produced it ([`provenance::Fact`]):
//!
//! 1. **Declared** — [`declared`] scans arbitrary JSON for concepts using a
//!    synonym registry (`context_length` / `context_window` / `max_model_len`
//!    …) and a recursive path-recording walker. No provider names appear.
//! 2. **Inferred** — [`errors::mine_limits`] recovers limits from rejection
//!    text (`should be in [1, 65536]`), turning failures into facts.
//! 3. **Probed** — [`probe`] sends real requests and observes outcomes.
//!
//! When a probe contradicts a declaration the **probe wins** and the
//! conflict is recorded on the model as an [`engine::DiscoveredModel::anomalies`]
//! entry, so a wrong capability is traceable rather than invisible.
//!
//! ## Two failure modes this crate exists to prevent
//!
//! * **Null-echoed fields read as capabilities.** NVIDIA returns
//!   `reasoning`, `audio`, `tool_calls` and `annotations` as explicit
//!   `null`s on models supporting none of them. Key-presence checks therefore
//!   report capabilities that do not exist. [`response::normalize_message`]
//!   is value-driven for exactly this reason.
//! * **HTTP 200 with no answer.** Reasoning-first models can consume the
//!   entire completion budget on chain-of-thought and return `200 OK` with
//!   no `content` field at all. [`response::diagnose_empty`] classifies the
//!   cause so the caller can fix it instead of seeing a blank string.

pub mod declared;
pub mod engine;
pub mod errors;
pub mod probe;
pub mod provenance;
pub mod response;

pub use engine::{
    DiscoveredModel, DiscoveryConfig, DiscoveryEngine, ModelRole, modalities_from_strings,
    to_model_info,
};
pub use errors::{ClassifiedError, ErrorClass, LimitKind, MinedLimit, classify, mine_limits};
pub use probe::{Reachability, Transport, probe_reachable};
pub use provenance::{Fact, Source};
pub use response::{EmptyAnswerCause, NormalizedMessage, diagnose_empty, normalize_message};

use thiserror::Error;

/// Errors raised by the discovery process itself.
#[derive(Debug, Error)]
pub enum DiscoveryError {
    /// Listing models failed.
    #[error("model listing failed (HTTP {status}): {message}")]
    ListFailed {
        /// HTTP status.
        status: u16,
        /// Detail.
        message: String,
    },
    /// Transport construction failed.
    #[error("transport error: {0}")]
    Transport(#[from] reqwest::Error),
    /// The base URL was rejected before any credential was attached to it.
    #[error("invalid base url `{url}`: {reason}")]
    InvalidBaseUrl {
        /// The URL that was rejected.
        url: String,
        /// Why it was rejected.
        reason: String,
    },
}
