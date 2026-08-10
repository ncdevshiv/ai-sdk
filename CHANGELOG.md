# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Rust workspace scaffold (25 crates) per `ENGINEERING-SPEC.md` — 2026-08-09
- `ENGINEERING-SPEC.md`: Rust engineering specification adapted from the
  master template to the PRD v1.2 vision
- `ENGINEERING-LOG.md`: chronological execution report (what/assumed/passed/
  failed/root-cause/fix) for every stage, including live gateway testing
- `ai-types`, `ai-errors`, `ai-config`, `ai-models`, `ai-core`: domain types,
  typed error hierarchy, unified configuration, model registry with pricing,
  core traits + `AiClient` (26 unit tests)
- `ai-stream`: SSE parser (CRLF/CR/LF, multiline data, chunk boundaries,
  EOF flush, multi-event chunks), `collect_text`/`collect_completion`,
  `ToolCallAccumulator` with finalization (12 unit tests)
- `ai-runtime`: retry with exponential backoff+jitter, per-key concurrency
  limits, parallel fan-out/fan-in with deadlines and partial results,
  race/fallback, circuit breaker (22 unit tests)
- `ai-providers`: real OpenAI-compatible adapter (OpenAI wire protocol with
  DeepSeek extras: reasoning deltas, cache-aware usage) — works against the
  project gateway `opencode.ai/zen/go/v1` (16 unit tests)
- Live integration tests (`crates/ai-providers/tests/live_gateway.rs`):
  13 real-API tests covering discovery, generate, streaming, tool loop,
  structured output, vision, auth/unknown-model errors, and parallel calls
  across both models — **all 13 pass against the real gateway**
- ADR-011: Rust implementation language (supersedes ADR-001)
- ADR-012: workspace restructuring (removed dead `sdk` submodule gitlink)
- `.env.example` credential template; MIT license

### Added

- Full subsystem completion: `ai-memory` (four-tier + compaction),
  `ai-rag` (chunking/hybrid retrieval/reranking), `ai-workflows` (DAG
  engine with checkpoints), `ai-agents` (agent runtime, sub-agents,
  swarms, HITL, self-healing retries), `ai-analytics`, `ai-devtools`,
  `ai-edge`, `ai-voice` (VAD, real STT/TTS adapters)
- `ai-cli` binary: `doctor`, `providers`, `models`, `config`, `run`,
  `inspect`, `trace`, `benchmark` (all real, verified live)
- `ai-sdk` facade crate with `prelude`
- Live examples: `chat`, `agent`, `parallel`
- Live agent tool-loop test (`deepseek-v4-flash`): agent drives a real
  calculator call to "42"

### Fixed

- `ai-config` now reads the gateway env vars
  (`AI_SDK_GATEWAY_BASE_URL`/`AI_SDK_GATEWAY_API_KEY`) into the
  `opencode` provider (found by the CLI doctor smoke test)
- `ai-workflows` recursive executor rewritten to a boxed, state-by-value
  design (Send-provable; correct retry semantics)
- `ai-runtime::retry` accepts `FnMut` closures (needed for state capture)
- Model catalog pricing was per-1M while documented per-1K (1000× cost
  error) — corrected to per-1K semantics
- SSE parser treated `\r\n` as two line terminators (multiline `data:`
  broken) — CRLF consumed as one terminator
- Parallel executor dropped concurrency permits immediately (limits silently
  ignored) — permits held for task duration
- Parallel executor discarded completed results when a deadline fired —
  shared result store preserves partial results
- **SSE parser dropped all events after the first in multi-event network
  chunks** (found by live gateway testing; raw-probe verified the gateway
  stream was complete) — parser now consumes whole chunks before yielding

### Changed

- **MCP rewritten to the modern stateless 2026-07-28 revision** (ADR-013):
  per-request `_meta` protocol version + capabilities (no `initialize`
  handshake), `server/discover`, `resultType`, MRTR with elicitation,
  `subscriptions/listen` with `subscriptionId`, error codes
  `-32020/-32021/-32022`, Streamable HTTP transport with
  `MCP-Protocol-Version` header and HTTP 400 mapping. Legacy
  initialize-handshake era not implemented (documented; dual-era = roadmap).
- Repository restructured from docs-only to a Cargo workspace at repo root.
- README rewritten for the Rust implementation.

### Removed

- Dead `sdk/` submodule gitlink (no `.gitmodules` URL existed).
