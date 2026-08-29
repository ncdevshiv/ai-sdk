//! Provenance-annotated facts.
//!
//! Every capability this crate reports is wrapped in a [`Fact`] that records
//! *where the value came from* and *what evidence produced it*. Nothing is
//! asserted without a source. This is what makes a wrong capability
//! traceable to root cause instead of appearing as an unexplained number.

use serde::{Deserialize, Serialize};
use std::fmt;

/// How a fact was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Source {
    /// The provider stated it explicitly (e.g. a `context_length` field).
    Declared,
    /// Derived from text the provider returned (e.g. a limit mined out of an
    /// error message like "should be in [1, 65536]").
    Inferred,
    /// Determined by sending a real request and observing the outcome.
    Probed,
    /// Not discoverable from any source; the value is genuinely unknown.
    Unknown,
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Declared => "declared",
            Self::Inferred => "inferred",
            Self::Probed => "probed",
            Self::Unknown => "unknown",
        };
        f.write_str(s)
    }
}

/// A value together with its origin, confidence and evidence.
///
/// `confidence` is a 0.0–1.0 heuristic: declared metadata starts high but is
/// discounted when a probe contradicts it; probes are high-confidence when
/// they produced an unambiguous response, lower when the signal was weak.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Fact<T> {
    /// The discovered value.
    pub value: T,
    /// How the value was obtained.
    pub source: Source,
    /// 0.0–1.0 confidence in the value.
    pub confidence: f32,
    /// Human-readable evidence: the JSON path, the error text, or the probe
    /// that produced this value.
    pub evidence: String,
    /// JSON path within the provider's response, when the value was declared.
    pub path: Option<String>,
}

impl<T> Fact<T> {
    /// A value the provider declared, with the JSON path it came from.
    pub fn declared(value: T, path: impl Into<String>) -> Self {
        let path = path.into();
        Self {
            value,
            source: Source::Declared,
            confidence: 0.8,
            evidence: format!("declared at {path}"),
            path: Some(path),
        }
    }

    /// A value derived from provider-supplied text.
    pub fn inferred(value: T, evidence: impl Into<String>, confidence: f32) -> Self {
        Self {
            value,
            source: Source::Inferred,
            confidence,
            evidence: evidence.into(),
            path: None,
        }
    }

    /// A value established by an empirical probe.
    pub fn probed(value: T, evidence: impl Into<String>, confidence: f32) -> Self {
        Self {
            value,
            source: Source::Probed,
            confidence,
            evidence: evidence.into(),
            path: None,
        }
    }

    /// A value that could not be discovered.
    pub fn unknown(value: T, reason: impl Into<String>) -> Self {
        Self {
            value,
            source: Source::Unknown,
            confidence: 0.0,
            evidence: reason.into(),
            path: None,
        }
    }

    /// Maps the inner value, preserving provenance.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Fact<U> {
        Fact {
            value: f(self.value),
            source: self.source,
            confidence: self.confidence,
            evidence: self.evidence,
            path: self.path,
        }
    }

    /// Overrides the confidence (used when corroborating/contradicting).
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }
}

impl<T: fmt::Display> fmt::Display for Fact<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}, {:.2})",
            self.value, self.source, self.confidence
        )
    }
}

/// Corroborates or contradicts a declared value with a probe result.
///
/// Root-cause rule: when an empirical probe disagrees with what the provider
/// declared, **the probe wins** and the conflict is recorded so the
/// discrepancy is visible rather than silently resolved.
pub fn reconcile<T: PartialEq + Clone + fmt::Debug>(
    declared: Option<Fact<T>>,
    probed: Fact<T>,
) -> Fact<T> {
    match declared {
        None => probed,
        Some(d) if d.value == probed.value => {
            // Agreement raises confidence above either source alone.
            let mut agreed = probed.clone();
            agreed.confidence = 0.95;
            agreed.evidence = format!(
                "probe agrees with declaration ({}); {}",
                d.path.as_deref().unwrap_or("unknown path"),
                probed.evidence
            );
            agreed
        }
        Some(d) => {
            let mut conflict = probed.clone();
            conflict.confidence = probed.confidence.max(0.85);
            conflict.evidence = format!(
                "CONFLICT: declared {:?} at {} but probe observed a different value; {}",
                d.value,
                d.path.as_deref().unwrap_or("unknown path"),
                probed.evidence
            );
            conflict
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declared_records_path() {
        let f = Fact::declared(128_000u64, "$.data[0].context_length");
        assert_eq!(f.source, Source::Declared);
        assert_eq!(f.path.as_deref(), Some("$.data[0].context_length"));
        assert!(f.evidence.contains("context_length"));
    }

    #[test]
    fn probe_beats_conflicting_declaration_and_records_it() {
        let d = Fact::declared(true, "$.supports_vision");
        let p = Fact::probed(false, "image_url part rejected with HTTP 400", 0.9);
        let r = reconcile(Some(d), p);
        assert!(!r.value);
        assert!(r.evidence.contains("CONFLICT"));
    }

    #[test]
    fn agreement_raises_confidence() {
        let d = Fact::declared(true, "$.supports_tools");
        let p = Fact::probed(true, "tool call returned", 0.9);
        let r = reconcile(Some(d), p);
        assert!(r.value);
        assert!(r.confidence > 0.9);
    }
}
