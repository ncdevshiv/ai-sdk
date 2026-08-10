//! Observability: structured events, chronological execution history,
//! span/trace correlation, and exporters.
//!
//! Every significant AI execution produces [`ExecutionEvent`]s with
//! timestamps, trace/span ids, durations, and typed metadata. Events are
//! collected in-process (bounded), exported to subscribers, and can be
//! rendered as a chronological report (spec §14) — as structured telemetry,
//! not expensive string logging.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// What kind of operation an event describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    RequestStarted,
    ProviderCall,
    ModelCall,
    ToolCall,
    AgentStep,
    AgentState,
    WorkflowStep,
    WebRequest,
    MemoryOperation,
    Retry,
    Fallback,
    Error,
    Completed,
    Metric,
}

/// Lifecycle status of the operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventStatus {
    Started,
    Succeeded,
    Failed,
    Retrying,
    Cancelled,
}

/// A single structured execution event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionEvent {
    /// Wall-clock time (RFC 3339) for persistence/export.
    pub wall_time: String,
    /// Monotonic offset from the collector's start (for chronological
    /// reports).
    pub offset_ms: u64,
    pub trace_id: String,
    pub span_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<String>,
    pub kind: EventKind,
    pub operation: String,
    pub status: EventStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl ExecutionEvent {
    pub fn started(&self) -> bool {
        self.status == EventStatus::Started
    }
}

/// Receives exported events (files, stdout, remote backends).
pub trait EventExporter: Send + Sync {
    fn export(&self, events: &[ExecutionEvent]);
}

/// Writes events as newline-delimited JSON.
pub struct JsonLinesExporter<W: std::io::Write + Send> {
    writer: parking_lot::Mutex<W>,
}

impl<W: std::io::Write + Send> JsonLinesExporter<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer: parking_lot::Mutex::new(writer),
        }
    }
}

impl<W: std::io::Write + Send> EventExporter for JsonLinesExporter<W> {
    fn export(&self, events: &[ExecutionEvent]) {
        let mut writer = self.writer.lock();
        for event in events {
            if let Ok(line) = serde_json::to_string(event) {
                let _ = writeln!(writer, "{line}");
            }
        }
    }
}

/// Collects execution events with bounded retention.
///
/// `capacity` bounds memory (spec: bounded resources). Events beyond the
/// capacity drop the oldest.
#[derive(Clone)]
pub struct EventCollector {
    inner: Arc<RwLock<Inner>>,
}

struct Inner {
    events: Vec<ExecutionEvent>,
    capacity: usize,
    start: Instant,
}

impl Default for EventCollector {
    fn default() -> Self {
        Self::with_capacity(10_000)
    }
}

impl EventCollector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(Inner {
                events: Vec::new(),
                capacity,
                start: Instant::now(),
            })),
        }
    }

    /// Records an event, returning its span id.
    pub fn record(
        &self,
        kind: EventKind,
        operation: impl Into<String>,
        status: EventStatus,
        metadata: BTreeMap<String, serde_json::Value>,
    ) -> String {
        let span_id = Uuid::new_v4().to_string();
        let trace_id = Uuid::new_v4().to_string();
        self.record_with_ids(
            kind,
            operation,
            status,
            metadata,
            trace_id,
            span_id.clone(),
            None,
            None,
        );
        span_id
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_with_ids(
        &self,
        kind: EventKind,
        operation: impl Into<String>,
        status: EventStatus,
        metadata: BTreeMap<String, serde_json::Value>,
        trace_id: String,
        span_id: String,
        parent_span_id: Option<String>,
        duration_ms: Option<u64>,
    ) {
        let mut inner = self.inner.write();
        let offset_ms = inner.start.elapsed().as_millis() as u64;
        let event = ExecutionEvent {
            wall_time: format!("{:?}", std::time::SystemTime::now()),
            offset_ms,
            trace_id,
            span_id,
            parent_span_id,
            kind,
            operation: operation.into(),
            status,
            duration_ms,
            metadata,
        };
        if inner.events.len() >= inner.capacity {
            inner.events.remove(0);
        }
        inner.events.push(event);
    }

    /// All events, oldest first.
    pub fn events(&self) -> Vec<ExecutionEvent> {
        self.inner.read().events.clone()
    }

    /// Events for a given trace.
    pub fn trace(&self, trace_id: &str) -> Vec<ExecutionEvent> {
        self.inner
            .read()
            .events
            .iter()
            .filter(|e| e.trace_id == trace_id)
            .cloned()
            .collect()
    }

    /// Exports all events through every exporter and clears the in-memory
    /// buffer (events are persisted/streamed downstream).
    pub fn flush(&self, exporters: &[Arc<dyn EventExporter>]) {
        let events = {
            let mut inner = self.inner.write();
            std::mem::take(&mut inner.events)
        };
        for exporter in exporters {
            exporter.export(&events);
        }
    }

    pub fn len(&self) -> usize {
        self.inner.read().events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Renders events as a chronological execution report (spec §14):
///
/// ```text
/// 00:00.000 Request received
/// 00:00.003 Agent initialized
/// ...
/// ```
pub fn chronological_report(events: &[ExecutionEvent]) -> String {
    let mut sorted = events.to_vec();
    sorted.sort_by_key(|e| e.offset_ms);

    let mut out = String::new();
    for event in sorted {
        let secs = event.offset_ms / 1000;
        let millis = event.offset_ms % 1000;
        out.push_str(&format!("{:02}:{:02}.{:03} ", secs / 60, secs % 60, millis));
        match event.status {
            EventStatus::Started => out.push_str("▶ "),
            EventStatus::Succeeded => out.push_str("✓ "),
            EventStatus::Failed => out.push_str("✗ "),
            EventStatus::Retrying => out.push_str("↻ "),
            EventStatus::Cancelled => out.push_str("⏹ "),
        }
        out.push_str(&format!("{:?} {}", event.kind, event.operation));
        if let Some(d) = event.duration_ms {
            out.push_str(&format!(" ({d} ms)"));
        }
        out.push('\n');
    }
    out
}

/// A timing guard that records a completed event on drop (structured, not
/// string-logged). Use via [`EventCollector::span`]-style helpers.
pub struct EventGuard {
    collector: EventCollector,
    kind: EventKind,
    operation: String,
    trace_id: String,
    span_id: String,
    parent_span_id: Option<String>,
    started: Instant,
    metadata: BTreeMap<String, serde_json::Value>,
    finished: bool,
}

impl EventGuard {
    pub fn new(
        collector: &EventCollector,
        kind: EventKind,
        operation: impl Into<String>,
        trace_id: String,
        span_id: String,
        parent_span_id: Option<String>,
    ) -> Self {
        Self {
            collector: collector.clone(),
            kind,
            operation: operation.into(),
            trace_id,
            span_id,
            parent_span_id,
            started: Instant::now(),
            metadata: BTreeMap::new(),
            finished: false,
        }
    }

    pub fn with_metadata(mut self, key: &str, value: serde_json::Value) -> Self {
        self.metadata.insert(key.to_string(), value);
        self
    }

    pub fn span_id(&self) -> &str {
        &self.span_id
    }

    /// Records the final event for this span. Idempotent: a dropped guard
    /// after an explicit `finish` records nothing.
    pub fn finish(mut self, status: EventStatus) {
        if !self.finished {
            self.finished = true;
            self.record(status);
        }
    }

    fn record(&self, status: EventStatus) {
        let duration_ms = self.started.elapsed().as_millis() as u64;
        self.collector.record_with_ids(
            self.kind,
            &self.operation,
            status,
            self.metadata.clone(),
            self.trace_id.clone(),
            self.span_id.clone(),
            self.parent_span_id.clone(),
            Some(duration_ms),
        );
    }
}

impl Drop for EventGuard {
    fn drop(&mut self) {
        // Record a completed event only if the guard was not explicitly
        // finished.
        if !self.finished {
            self.record(EventStatus::Succeeded);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn collector_records_and_bounds() {
        let collector = EventCollector::with_capacity(3);
        for i in 0..5 {
            collector.record(
                EventKind::ProviderCall,
                format!("call-{i}"),
                EventStatus::Succeeded,
                BTreeMap::new(),
            );
        }
        assert_eq!(collector.len(), 3, "capacity bounds memory");
        let events = collector.events();
        assert_eq!(events[0].operation, "call-2");
    }

    #[test]
    fn chronological_report_is_ordered() {
        let collector = EventCollector::new();
        collector.record(
            EventKind::RequestStarted,
            "request",
            EventStatus::Started,
            BTreeMap::new(),
        );
        collector.record(
            EventKind::ModelCall,
            "openai:gpt-4o",
            EventStatus::Succeeded,
            BTreeMap::new(),
        );
        collector.record(
            EventKind::Completed,
            "request",
            EventStatus::Succeeded,
            BTreeMap::new(),
        );

        let report = chronological_report(&collector.events());
        assert!(report.contains("▶ RequestStarted request"), "{report}");
        assert!(report.contains("✓ ModelCall"), "{report}");
        assert!(report.contains("00:00."), "{report}");
    }

    #[test]
    fn trace_filtering_works() {
        let collector = EventCollector::new();
        let span = collector.record(
            EventKind::AgentStep,
            "step-1",
            EventStatus::Started,
            BTreeMap::new(),
        );
        let events = collector.events();
        let trace_id = events[0].trace_id.clone();
        assert_eq!(collector.trace(&trace_id).len(), 1);
        assert_eq!(collector.trace("nope").len(), 0);
        assert!(!span.is_empty());
    }

    #[test]
    fn guard_records_duration() {
        let collector = EventCollector::new();
        {
            let guard = EventGuard::new(
                &collector,
                EventKind::ToolCall,
                "calculator",
                "t".into(),
                "s".into(),
                None,
            );
            std::thread::sleep(Duration::from_millis(5));
            guard.finish(EventStatus::Succeeded);
        }
        let events = collector.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, EventKind::ToolCall);
        assert!(events[0].duration_ms.unwrap_or(0) >= 5);
    }

    #[test]
    fn json_lines_exporter_writes_valid_lines() {
        let collector = EventCollector::new();
        collector.record(
            EventKind::Metric,
            "tokens",
            EventStatus::Succeeded,
            BTreeMap::new(),
        );

        let dir = std::env::temp_dir();
        let path = dir.join(format!("ai-sdk-events-{}.jsonl", std::process::id()));
        let file = std::fs::File::create(&path).expect("event file created");
        let exporter = Arc::new(JsonLinesExporter::new(file));
        collector.flush(&[exporter]);
        assert!(collector.is_empty(), "flush clears the buffer");

        let text = std::fs::read_to_string(&path).expect("event file readable");
        std::fs::remove_file(&path).ok();
        let parsed: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(parsed["kind"], "metric");
    }
}
