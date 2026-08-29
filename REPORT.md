# Universal Model Discovery — Engineering Report

**Date:** 2026-08-28 · **Scope:** provider-agnostic model discovery + capability
detection for the `ai-sdk` workspace, validated live against three gateways
(b.ai, NVIDIA, SenseNova) — 133 models, plus a wire-level mock battery.

**Deliverables**

| Artifact | Where |
|----------|-------|
| Discovery engine (generic, no provider name anywhere) | `crates/ai-discovery/` |
| Edge-case battery (28 wire-level tests) | `crates/ai-discovery/tests/edge_harness.rs` |
| Chronological journal (every issue, timestamped, root-caused) | `DISCOVERY-JOURNAL.md` |
| Live probe runner | `crates/ai-discovery/examples/discover.rs`, `tools/discovery-probe/` |

---

## 1. Headline results

**All three providers were probed with a single generic engine — no provider
or model id appears in the discovery code.** Zero hardcoded context lengths,
zero provider-specific error parsing, zero model-type assumptions.

### Live sweep summary

| Gateway | Listed | Reachable | Breakdown of the rest |
|---------|--------|-----------|-----------------------|
| b.ai    | 46     | **6**      | 38 billing-blocked, 2 ids don't exist |
| NVIDIA  | 83     | **19**     | 55 not served for this account, 4 timeouts (>90 s), 2 server-error, 2 bad-request, 1 recovered-on-retry |
| SenseNova | 4    | **2**      | 2 route to image-generation endpoints |

Role map (endpoint routing, all 83 NVIDIA models): **20 chat, 2 embedding**
(`nemotron-3-embed-1b`, `llama-nemotron-embed-vl-1b-v2`), 61 unknown
(almost all account-not-served). A model that 404s on chat can still be an
embedding model — role comes from *which endpoint accepts it*, never its
name.

### Capability discovery — measured, with provenance (all three providers)

- **Context / output limits:** output ceilings mined from *rejections* where
  the gateway declares nothing — b.ai `qwen3.8-flash` → 131072,
  `glm-5.3-flash` → 131072, `deepseek-v4-flash-vision-exp` → 393216.
  Where the gateway accepts absurd bounds silently (`deepseek-v4-flash`,
  `hy3` accepted `max_tokens: 100_000_000`), the ceiling is honestly reported
  *unknown* rather than guessed at 8k/128k.
- **Thinking toggle:** the working spelling is **per-model**, not per-provider:
  `hy3` → `thinking.type=disabled`; `glm-5.3-flash` → `reasoning_effort=low`;
  `qwen3.8-flash` → `enable_thinking=false`; NVIDIA
  `nemotron-3-nano-omni-30b-a3b-reasoning` → `reasoning_effort=none`.
  All other spellings were **accepted with HTTP 200 and silently ignored**
  (one even raised tokens cost: reasoning stayed on). Acceptance is never
  treated as support. Bonus: `gpt-oss-20b/120b` and `muse-glimmer-30b` have
  **no working off-switch by any tested spelling** — recording that
  per model is (for these models) the difference between a 2× token bill and
  a normal one.
- **Modalities:** `hy3` (b.ai) is the only vision model of the six task
  models; SenseNova's `u1.5-lite` is an image model *despite* 404-ing on chat,
  and `u1-fast` *routes* to `/images/generations` but its backend returned 500
  (recorded as anomaly, role stays unknown). 404-on-chat ≠ broken.
- **Structured output:** `json_object` ≠ `json_schema`. Both SenseNova
  siblings accept the former and reject the latter (grammar error); b.ai's
  `deepseek-v4-flash-vision-exp` does the same while its sibling supports both.
- **Declared metadata is wrong:** SenseNova declares image input and `tools`
  for both chat siblings; the image probe fails and the tool probe flips
  between runs. Declarations are treated as a *prior*, never as the answer.

## 2. Bugs found and fixed today

| # | Bug | Root cause | Fix |
|---|-----|-----------|-----|
| J-019 | Tree didn't compile after prior session | last edit landed unfinished (stray `}` + stale initializers) | fixed; green tree restored (workspace check clean) |
| J-020 | Client timeouts classified `Network`, not `Timeout` | `reqwest::Error`'s `Display` hides the cause (it is in the source chain) | `describe_transport_error` walks the chain; regression test |
| J-021 | 404 with empty body classified `Other` | class derived from body text; empty body has no tokens | status-first: 404 + empty → `ModelNotFound` |
| J-022 | `probe_streaming` false-positive on `"data:"` inside content | substring scan instead of line-start frame check | require `data:` at line start; adversarial test |
| J-023 | HTTP 200 with no `choices` was silently "reachable" | reachability defined as status only | anomaly `no usable message` recorded |
| J-024 | `discover()` swallowed `/v1/models` failures as "0 models" | `unwrap_or_default()` at the only error boundary | `discover()` returns `Result`; example exits FATAL |
| J-025 | Runtime SDK defaulted `streaming/tools/structured` = true for *every* model under test (catalog has zero entries for all three providers) | guess-defaults in the no-catalog branch, contradicting its own doc | defaults now `false`; 72 provider tests still pass |
| J-026 | Thinking toggle spelling is model-local → any cached "provider toggle" is wrong | models route to different backends; gateways ignore unknown params silently | per-model battery (was already in place; now *verified* on 5 b.ai models simultaneously) |
| J-028 | Tool capability flipped between runs (SenseNova 6.8: `tools=n` → `tools=y`, identical requests) | tool-call *emission* is stochastic; a single sample is a coin flip | 3 samples at `temperature=0`, majority vote, `n/m` in evidence, confidence by agreement |
| J-029 | `mimo-v2.5` OK at 23:14, fail at 23:26/23:28 | account `***.only` provider preference (tencent) excludes all six providers serving the model — intermittent, account-config, not a model defect | documented; `retest()` sampling recommended (see §5) |
| J-030 | `nemotron-3.5-lightning` `DEGRADED function` at 23:20 → 404-empty at 23:53 | NVIDIA function deployment states cycle; states are not model health | `TemporarilyUnavailable` class recommended (see §5.7) |
| J-031 | `llama-3.2-11b-vision-instruct` probed `in=T` (no vision) despite being a vision model | image probe treated *any* non-2xx as negative; a 500 means routed-but-unhealthy (J-017 rule), not unsupported | 5xx/Timeout/Network on image probe → `inconclusive` anomaly, confidence 0.3; mock test pins it |

## 3. Root-cause taxonomy

Across J-001…J-029, a small set of causes recurs:

1. **The gateway is not the model.** `/v1/models` lists what the gateway is
   *configured* to sell; NVIDIA 404s the account on 55/83 today, and 73/83
   yesterday's run. Availability is account- and time-dependent.
2. **Metadata is authorship, not measurement.** SenseNova is the only
   publisher of capability fields and it is wrong on image input.
3. **HTTP status is the weakest signal.** Billing arrives as 400/403/429;
   "upstream no providers" as 400; degraded functions as 400; errors come in
   seven envelope shapes with empty bodies, HTML pages and nulls.
4. **200 is not a verdict.** Reasoning-only completions, missing `choices`,
   silently ignored parameters, stochastic tool calls — four different traps
   behind a single 200.
5. **Key presence ≠ capability.** NVIDIA echoes every optional field as
   `null`; value-driven normalization is the only safe reading.
6. **Latency is a distribution.** `gpt-oss-120b`: 22 s this hour, 57 s last
   hour. One fixed timeout is wrong in both directions.

## 4. Live probe runner

`cargo run -p ai-discovery --example discover -- --name <p> --base-url <u>
--key <k> [--policy conservative|default|none] [--concurrency N]
[--no-vision|--no-tools|--no-structured|--no-endpoints|--no-thinking]
[--probe-context] [--only a,b] [--extra a,b] --out out.json`

`--policy conservative` is required for b.ai: probing it with default pacing
produced 40/46 false "rate limited" failures (J-002) — the probe measured
its own interference. `--only` / `--extra` let you probe stale ids (the
capitalized names in the task → 404 → they get `listed: false` and a
`not served by this gateway` verdict, plus you can see *which* lowercase form
exists).

## 5. Architectural improvements

Legend: ✅ implemented in the follow-up session (2026-08-29; see the
journal's continuation section, J-032…J-035).

### 5.1 Replace the hardcoded catalog with a discover-then-cache bootstrap
`ai_models::default_catalog` (438 lines) has **zero** entries for any of the
three providers in scope, yet it runs first in `model_info_from_entry`.
Recommend: first use against a provider → run `ai-discovery` once → persist a
provider-scoped registry (`provider_models.json`) with a TTL; catalog drops to
a last-resort fallback for offline use. Wire `ai-providers::list_models` to go
through `engine::to_model_info` so CLI/sidecar (`commands.rs:41`, `sidecar
lib.rs:398`) gets probed capabilities instead of catalog guesses.
✅ *Half:* `model_info_from_entry` now parses declared fields through the
generic concept scanner (5.6b); the discover-then-cache bootstrap is still
open (registry-layout change).

### 5.2 Make the capability flags tri-state
A bool that the SDK sets `false` "because it wasn't measured" is read by UI
as "does not support". At `to_model_info`, promote the three probes to
`Option<bool>`-backed knowledge (`ModelCapabilities` gets
`capabilities_known: bool` or the fields become `Option`), so "unknown" is
renderable as `?`/gray rather than a confident `false` or `true`.

### 5.3 Flakiness-aware health (J-029, J-030)
A `retest()` API holding observation timestamps + sample counts; when a
single shot returns `BadRequest`/`ServerError` and the message contains
upstream-vocabulary (`No allowed providers`, `DEGRADED function`, `Upstream
request failed`), re-sample 2× and mark the model `intermittent` rather than
`broken`. Runtime then treats it as "try again" not "this id is dead".
✅ *Half:* the vocabulary now classifies to `TemporarilyUnavailable` (J-033)
instead of `BadRequest`; the `retest()` verdict-sampling API is still open.

### 5.4 Timeout is retryable — that multiplies dead-model cost ×4 ✅
`ErrorClass::Timeout` retryable means a model slower than the timeout costs
`attempts × timeout` (NVIDIA sweep: ~6.5 min per dead model). Implemented:
`TransportPolicy::max_timeout_attempts` — default 2 (one retry absorbs a
transient queue spike, J-005), `none()` = 1, capped by `max_attempts`.
Mock test pins 1 hit in sweep mode and exactly 2 at default.

### 5.5 ModelInfo should carry the discovered thinking toggle ✅
`ThinkingSupport.disable_spelling` is the operational artifact; cache it in
`ModelInfo` so an SDK caller can disable reasoning without re-running eight
probes. Implemented: `ModelInfo::thinking_control` (serde-default
`Option<String>`), populated by `to_model_info`.

### 5.6 Two more probe improvements
- `probe_max_output`: when the absurd value is accepted silently, retry with
  a ladder of large values (1 M → 4 M) to elicit a `[1, N]` range from
  gateways that cap only near the true bound. *(Open.)*
- ✅ `route_discover`: a chat 400 saying the model "does not support text
  input" now produces a traceable anomaly (J-035) instead of a bare
  `bad_request` with the reason lost.
- **5.6b (new) ✅** the scanner swap exposed `declared::first_*` resolving by
  JSON traversal order instead of synonym rank — fixed with `ranked_hits`
  (J-032); `extract_u64`/`extract_vision` in `ai-providers` deleted in favour
  of one shared synonym table.

### 5.7 Error vocabulary for transient infra states (J-030) ✅
`DEGRADED function cannot be invoked` and `Upstream request failed: [404]` are
*temporary* (deployment/rotation) but our classifier marks them
non-retryable. Implemented: `ErrorClass::TemporarilyUnavailable` (J-033),
retryable, vocabulary-driven, distinct from `RateLimited` (no Retry-After
promised) and from `NotEntitled`.

## 6. Residual risks (documented, not hidden)

- **Vision probe** accepts any 2xx as image support, even if the image part
  was silently dropped and the model answered only the text part. Confidence
  is 0.9; the safe check (send image-only prompt and look for a refusal) is a
  follow-up.
- **Retry-After** is parsed in seconds only; the HTTP-date form and
  `X-RateLimit-*` headers are ignored (none of the three gateways emitted
  them).
- **Context windows** on b.ai remain unknown (no metadata, binary search
  disabled by default in the runner because it costs ~10 requests per model
  on a throttle-prone gateway).
- **NVIDIA "not served"** is an account grant state; some of the 55 may be
  grantable — discovery cannot tell "not allowed" from "not deployed".
