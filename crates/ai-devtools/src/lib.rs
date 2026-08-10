//! Developer tools (spec §20, PRD §4.3): execution inspector and trace
//! viewer over [`ai_observability`] events, with configurable redaction so
//! sensitive data is never exposed.

use std::collections::BTreeMap;

use ai_errors::{AiError, InternalError};
use ai_observability::{
    EventCollector, EventKind, EventStatus, ExecutionEvent, chronological_report,
};

/// A filtered view of one trace.
#[derive(Debug, Clone)]
pub struct TraceView {
    pub trace_id: String,
    pub events: Vec<ExecutionEvent>,
    pub duration_ms: u64,
    pub status: EventStatus,
}

/// Inspects collected execution events.
pub struct Inspector {
    collector: EventCollector,
    /// Fields to redact in textual views (e.g. `sk-...` keys).
    redactor: ai_security::Redactor,
}

impl Inspector {
    pub fn new(collector: EventCollector) -> Self {
        Self {
            collector,
            redactor: ai_security::Redactor::new(Vec::new()),
        }
    }

    pub fn with_redactor(mut self, redactor: ai_security::Redactor) -> Self {
        self.redactor = redactor;
        self
    }

    /// All traces (grouped by trace id), newest offset first.
    pub fn traces(&self) -> Vec<TraceView> {
        let events = self.collector.events();
        let mut by_trace: BTreeMap<String, Vec<ExecutionEvent>> = BTreeMap::new();
        for event in events {
            by_trace
                .entry(event.trace_id.clone())
                .or_default()
                .push(event);
        }

        by_trace
            .into_iter()
            .map(|(trace_id, mut events)| {
                events.sort_by_key(|e| e.offset_ms);
                let duration_ms = events
                    .last()
                    .map(|e| e.offset_ms)
                    .unwrap_or(0)
                    .saturating_sub(events.first().map(|e| e.offset_ms).unwrap_or(0));
                let status = events
                    .iter()
                    .find(|e| e.status == EventStatus::Failed)
                    .map(|_| EventStatus::Failed)
                    .unwrap_or(EventStatus::Succeeded);
                TraceView {
                    trace_id,
                    events,
                    duration_ms,
                    status,
                }
            })
            .collect()
    }

    /// Events for a specific trace.
    pub fn trace(&self, trace_id: &str) -> Vec<ExecutionEvent> {
        self.collector.trace(trace_id)
    }

    /// Operations matching a kind (e.g. `ModelCall`).
    pub fn operations(&self, kind: EventKind) -> Vec<ExecutionEvent> {
        self.collector
            .events()
            .into_iter()
            .filter(|e| e.kind == kind)
            .collect()
    }

    /// A redacted chronological report for a trace (spec §14).
    pub fn report(&self, trace_id: &str) -> String {
        let events = self.trace(trace_id);
        let report = chronological_report(&events);
        self.redactor.redact(&report)
    }

    /// The JSON-lines export of a trace with sensitive metadata redacted.
    pub fn export_json(&self, trace_id: &str) -> Result<String, AiError> {
        let events = self.trace(trace_id);
        let mut lines = Vec::new();
        for event in events {
            let json = serde_json::to_string(&event)
                .map_err(|e| AiError::Internal(InternalError::new(e.to_string())))?;
            lines.push(self.redactor.redact(&json));
        }
        Ok(lines.join("\n"))
    }

    pub fn collector(&self) -> &EventCollector {
        &self.collector
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_observability::{EventCollector, EventKind};

    fn sample_collector() -> EventCollector {
        let collector = EventCollector::new();
        // One execution shares a trace id (emitters use record_with_ids).
        for (index, (kind, operation)) in [
            (EventKind::RequestStarted, "request"),
            (EventKind::ModelCall, "openai:gpt-4o"),
            (EventKind::Completed, "request"),
        ]
        .into_iter()
        .enumerate()
        {
            collector.record_with_ids(
                kind,
                operation,
                EventStatus::Succeeded,
                BTreeMap::new(),
                "trace-1".to_string(),
                format!("span-{index}"),
                if index == 0 {
                    None
                } else {
                    Some("span-0".to_string())
                },
                Some(10),
            );
        }
        collector
    }

    #[test]
    fn traces_group_and_order() {
        let inspector = Inspector::new(sample_collector());
        let traces = inspector.traces();
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].events.len(), 3);
        assert_eq!(traces[0].status, EventStatus::Succeeded);
    }

    #[test]
    fn operations_filter_by_kind() {
        let inspector = Inspector::new(sample_collector());
        assert_eq!(inspector.operations(EventKind::ModelCall).len(), 1);
        assert_eq!(inspector.operations(EventKind::ToolCall).len(), 0);
    }

    #[test]
    fn report_is_redacted() {
        let collector = EventCollector::new();
        collector.record(
            EventKind::ModelCall,
            "auth with sk-abcdef1234567890xyz",
            EventStatus::Succeeded,
            BTreeMap::new(),
        );
        let inspector = Inspector::new(collector);
        let report = inspector.report(&inspector.traces()[0].trace_id);
        assert!(!report.contains("sk-abcdef1234567890xyz"), "{report}");
        assert!(report.contains("[REDACTED]"), "{report}");
    }

    #[test]
    fn export_json_is_parseable_and_redacted() {
        let collector = EventCollector::new();
        collector.record(
            EventKind::ModelCall,
            "Bearer abcdef1234567890xyz",
            EventStatus::Succeeded,
            BTreeMap::new(),
        );
        let inspector = Inspector::new(collector);
        let trace_id = inspector.traces()[0].trace_id.clone();
        let json = inspector.export_json(&trace_id).unwrap();
        assert!(!json.contains("abcdef1234567890xyz"), "{json}");
        let parsed: serde_json::Value = serde_json::from_str(json.lines().next().unwrap()).unwrap();
        assert_eq!(parsed["kind"], "model_call");
    }
}
