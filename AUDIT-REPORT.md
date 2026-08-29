# Code Audit — working-tree changes

**Date:** 2026-08-29
**Scope:** all uncommitted changes on `main`

| Area | Files | Change |
|---|---|---|
| New crate `ai-discovery` | 10 files (~2,900 LOC) | provider-agnostic model/capability discovery |
| `ai-providers` | `openai_compat.rs` (+260/−33), `anthropic.rs`, `gemini.rs` | catalog-aware `ModelInfo`, remove fake limits |
| `ai-core` | `src/lib.rs` | new `ReasoningEffort::Max` variant |
| `ai-sidecar` | `src/lib.rs` | `health` RPC + `drain_streams` |
| Workspace | `Cargo.toml`, `Cargo.lock` | add crate + 6 deps |
| Untracked artifacts | `tools/discovery-probe/`, `DISCOVERY-JOURNAL.md`, `REPORT.md` | probe scripts + result captures |

**Verification performed:** `cargo check --workspace --all-targets` (clean), `cargo clippy -p ai-discovery --all-targets` (zero warnings), `cargo test --workspace` (all crates green except the two flaky `ai-discovery` tests below), plus a temporary harness of 9 targeted probes that empirically confirmed 8 defects (since removed).

**Overall assessment:** the design is genuinely strong — the provenance model, the "probe beats declaration" reconciliation rule, and the removal of the hardcoded `128_000/8_192` fake limits are real improvements over what they replaced. The defects below are concentrated in (a) secrets handling in the untracked artifacts, (b) the heuristics in `errors.rs`/`declared.rs` that are too loose and can assert false facts, and (c) one bad abort path in the context-window probe.

---

## Critical

### C-1 — Live API keys sit in the working tree, unignored

`tools/discovery-probe/cfg_bai.json`, `cfg_sn.json`, and `out/bai_stageA.log` contain **plaintext production API keys** (verified: one `sk-2bhb…` b.ai key, one `sk-WBu…` SenseNova key; the b.ai key is also echoed verbatim inside `out/bai_stageA.log`).

These files are currently untracked and `.gitignore` does **not** match them (`git check-ignore` returns nothing). Any `git add -A` / `git add .` commits them permanently — and the repo publishes releases from tags, so the keys would land in public history.

**Recommendation (do first):**
1. Rotate both keys now. They must be assumed compromised.
2. Delete the files or scrub the keys; re-run the probes reading keys from env vars.
3. Add to `.gitignore`:
   ```
   tools/discovery-probe/*.log
   tools/discovery-probe/cfg_*.json
   tools/discovery-probe/out/
   tools/discovery-probe/*.exe
   ```
4. Add a pre-commit secret scan (gitleaks or `cargo-audit`-style hook).

### C-2 — `Transport` leaks the API key through `Debug`

`crates/ai-discovery/src/probe.rs:137-146`

```rust
#[derive(Debug, Clone)]
pub struct Transport {
    client: reqwest::Client,
    base_url: String,
    api_key: String,      // <-- printed by the derived Debug
    ...
}
```

Verified: `format!("{:?}", transport)` contains the key in full. `Transport` is public, is cloned into spawned tasks, and the crate already uses `tracing::debug!`, so a single `{:?}` in a log statement or an error path exfiltrates the credential into logs.

**Recommendation:** implement `Debug` manually and redact:

```rust
impl std::fmt::Debug for Transport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transport")
            .field("base_url", &self.base_url)
            .field("api_key", &"<redacted>")
            .field("timeout", &self.timeout)
            .field("policy", &self.policy)
            .finish_non_exhaustive()
    }
}
```

Audit the rest of the workspace for the same pattern (`OpenAiCompatConfig` already has a hand-written `Debug`; confirm it redacts `api_key`).

---

## High

### H-1 — No scheme validation on `base_url`: bearer token can be sent in cleartext

`probe.rs:263-299`

```rust
let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));
...
.bearer_auth(&self.api_key)
```

`base_url` is never validated. A config typo, an `http://` redirect target, or a user-supplied gateway URL causes the API key to be transmitted in plaintext. There is also no SSRF guard if base URLs ever become externally influenced, and no redirect policy is set (a `https` → `http` downgrade redirect would carry the `Authorization` header with it).

Notably, `url` is declared in `Cargo.toml` but never used anywhere in the crate (see L-1) — the validation was evidently intended and never written.

**Recommendation:** parse with `url::Url` in `with_policy`, reject non-`https` (allow `http` only for loopback, to keep the test harness working), and pin redirects:

```rust
let parsed = url::Url::parse(&base_url).map_err(...)?;
if parsed.scheme() != "https" && !is_loopback(&parsed) {
    return Err(/* reject */);
}
let client = reqwest::Client::builder()
    .redirect(reqwest::redirect::Policy::limited(2))  // or none()
    ...
```

### H-2 — Context-window probe fabricates evidence and returns wrong values

`probe.rs:785-844` — verified empirically.

The `Fit::Other` branch records an abort note, then the function **overwrites** `evidence` unconditionally before returning:

```rust
(Fit::Other, _) => {
    evidence.push_str(&format!("; aborted at {mid} (non-context failure)"));
    break;
}
...
if saw_rejection {
    evidence = format!("binary search: largest accepted ≈ {lo} tokens");   // overwrite
    (Some(lo as u64), evidence)
} else {
    evidence = format!("no context rejection observed up to {lo} tokens; value is a LOWER BOUND"); // overwrite
    (best, evidence)
}
```

Observed with a mock that answers twice then returns 500:

```
value=Some(512)  evidence="no context rejection observed up to 4352 tokens; value is a LOWER BOUND"
```

Two separate defects: the abort note is discarded, and the message claims success up to **4352** — the size that actually *failed* — when the last successful size was **512**. The result is then wrapped as `Fact::probed(v, ev, 0.7)` and reported as a measured fact.

The same defect makes the value nondeterministic. The existing test `context_window_binary_searches_until_rejection` produced four different answers across runs against an identical mock:

```
value was 512   /   2432   /   3392   /   3872   (and occasionally passes ≈3992)
```

Any transient 429/5xx/network blip — which `TransportPolicy::none()` does not retry — silently truncates the search. Because `ai-discovery` is the component telling the SDK what a model can do, a silently wrong context window is worse than an unknown one.

**Recommendation:** use `push_str` instead of assignment, track an `aborted` flag, and refuse to report a confident value:

```rust
if aborted {
    let ev = format!("search ABORTED at {mid} ({}); largest confirmed acceptance is {lo} — LOWER BOUND, unreliable", err_class);
    return (None, ev);            // or Some(lo) with confidence ~0.2
}
```

Also: treat `ServerError`/`RateLimited` as "retry, then abort with `None`", not as a silent early exit.

---

## Medium

### M-1 — The "answer field fallback" in `normalize_message` is dead code

`response.rs:104-113` — verified.

```rust
out.answer = obj.iter()
    .filter(|(k, _)| classify_field(k, &Value::Null) == FieldRole::Other)
    .find_map(...)
```

`classify_field` returns `FieldRole::Empty` immediately when the value is null, so the filter is **never** true and `find_map` always yields `None`. Verified: `{"role":"assistant","answer":"the real reply","thinking":"hmm"}` normalizes to `answer = None`.

The documented contract — "fall back to any populated non-reasoning string field so gateways that omit `content` still yield their output" — is therefore not implemented. Such a model reports `answer_is_missing() == true`, gets diagnosed as `EmptyByStop`, and is flagged as broken when it returned a perfectly good reply.

**Recommendation:** filter on the *actual* value, not `Null`:

```rust
.filter(|(k, v)| classify_field(k, v) == FieldRole::Other)
```

Then add a regression test with a gateway that returns `answer`/`text`/`output` instead of `content`.

### M-2 — `mine_limits` mines arbitrary numbers as capability limits

`errors.rs:338-387` — verified.

The `"up to "` pattern matches any sentence. Verified: `mine_limits("You may send up to 40 requests per minute")` returns `MinedLimit { value: 40, kind: MaxOutputTokens }`. That value flows through `probe_max_output` into `Fact::inferred(40, …, 0.9)`, and `reconcile` lets it **override the curated catalog**. A rate-limit message silently becomes "this model can emit at most 40 tokens".

**Recommendation:** require a token-ish unit to follow the number (`tokens`, `characters`), and drop the bare `"up to "` pattern entirely. Bound the result plausibly (`1..=1_000_000`).

### M-3 — Thousands separators truncate mined numbers

`errors.rs:414-428` — verified. `number_after` collects digits and stops at the first non-digit, so `"at most 128,000 tokens"` yields **128** (verified), and `"32,768"` yields **32**.

**Recommendation:** strip `,` and `_` from the digit run before parsing, mirroring what `declared::Hit::as_u64` already does correctly at `declared.rs:244`.

### M-4 — `bracket_range` gives up on the first bracket pair

`errors.rs:390-411` — verified. It takes the first `[`, finds its `]`, and bails if that pair holds no numbers. Verified: `"invalid [size] value, should be in [1, 65536]"` returns `[]` — the real ceiling is missed.

**Recommendation:** iterate over all bracket pairs until one yields numbers.

### M-5 — `has_feature` reports "not mentioned" as "declared false"

`declared.rs:342-351` — verified.

`has_feature` returns `Some(Fact::declared(hit, path))` whenever a feature list exists, so a list without the token yields `Some(false)`. Verified: `{"supported_features": ["tools","json_mode"]}` → `has_feature("vision") == false`.

In `engine.rs:342-346` that false value is treated as a positive declaration, and `engine.rs:589-593` then emits the anomaly *"declared supports_vision=false at … but image probe succeeded"*. The gateway never claimed vision was unsupported — it just didn't list it. This generates false anomalies on every gateway with a feature list. It also downgrades catalog entries in `openai_compat::model_info_from_entry`, where `extract_vision` has the same `has_text → Some(false)` conflation at `openai_compat.rs:145-175`.

The bidirectional match is also too loose: `normalize_key(f).contains(needle) || needle.contains(normalize_key(f))` will match any single-letter feature token against `needle`.

**Recommendation:** return `None` when the token is absent (`has_feature` → `Option<Fact<bool>>` with `Some` only on a positive hit, or add a tri-state `Declared(Option<bool>)`). Drop the reverse `needle.contains(...)` direction.

### M-6 — Two of 29 integration tests are flaky

Measured over repeat runs of `cargo test -p ai-discovery --test edge_harness`:

- serial (`--test-threads=1`): **passes** consistently
- default (parallel): `context_window_binary_searches_until_rejection` and `vision_probe_2xx_marks_image_input` fail intermittently (2 failures observed in 3 parallel runs)

Root cause is twofold. The mock at `edge_harness.rs:114-188` serves **one request per TCP connection** and then closes, so reqwest's pool intermittently hands out a dead socket; with `TransportPolicy::none()` (`max_attempts: 1`) there is no retry, so the failure is fatal. The production code then amplifies it via H-2 instead of degrading gracefully.

**Recommendation:** add `connection: close` to the mock response (or loop reading multiple requests per connection) so the harness is deterministic; separately fix H-2 so a transient failure cannot silently corrupt a measurement. Then run CI with `--test-threads=1` for this target as a backstop, and add `--fail-fast` off so one flake doesn't mask others.

### M-7 — Unknown models get contradictory capabilities from `list_models()` vs `model()`

`openai_compat.rs` — `model_info_from_entry` (used by `list_models`) vs `model_info_for_id` (used by `model`):

| | `list_models()` path | `model()` path |
|---|---|---|
| `supports_streaming` | `false` | **`true`** |
| `supports_tools` | `false` | **`true`** |

The same unknown model yields different capability sets depending on which entry point the caller used. Both paths also disagree with the struct field `capabilities` set in `OpenAiCompatProvider::new` (all-`false`), which appears to be dead after this refactor — worth confirming and removing.

**Recommendation:** one function, one answer. Have `model_info_for_id` call `model_info_from_entry` with an empty/`Null` entry so there is a single code path.

### M-8 — `ReasoningEffort::Max` is a no-op on every provider

- `openai_compat.rs:895-904` silently downgrades `Max` → `High`.
- `anthropic.rs` and `gemini.rs` never read `reasoning_effort` at all (verified: the only non-discovery reference in the workspace is the openai_compat match arm).

So `.with_reasoning_effort(ReasoningEffort::Max)` either silently degrades or is silently dropped, with no warning. Additionally, adding a variant to a public enum without `#[non_exhaustive]` is a breaking change for downstream exhaustive matches, and `CHANGELOG.md` has no entry for it.

**Recommendation:** either implement `Max` per-provider (or fall through unmapped), or remove the variant. Add `#[non_exhaustive]` to `ReasoningEffort`. Log a `tracing::warn!` on any normalization. Add a CHANGELOG entry.

### M-9 — Response bodies are read unbounded and read errors are swallowed

`probe.rs:278` and `probe.rs:311`:

```rust
let body = resp.text().await.unwrap_or_default();
```

`text()` buffers the whole body with no size cap, and `unwrap_or_default()` converts a mid-stream read failure into an empty string. An oversized or truncated response therefore becomes `status == 200` with `body == ""`, which downstream reads as "success with unusable payload" rather than a transport error — exactly the failure class this crate exists to detect.

**Recommendation:** cap with `response.chunk()` in a loop up to e.g. 8 MiB, and propagate read errors into `transport_error` instead of defaulting.

---

## Low

- **L-1 — Six unused dependencies.** `ai-errors`, `async-trait`, `regex`, `url`, `bytes`, `futures` are declared in `crates/ai-discovery/Cargo.toml` with zero references (verified by grep across `src/`, `examples/`, `tests/`). They are already baked into `Cargo.lock`. Remove them; keep `url` only if you adopt the H-1 fix.
- **L-2 — Synonym "priority" does not work as documented.** `declared.rs:65` claims "earlier entries are preferred", but `serde_json::Value::Object` iterates keys in sorted order, so preference is decided alphabetically. Verified: `{"id":"gpt-9","name":"GPT Nine"}` resolves `Concept::Name` to **`gpt-9`** at path `$.id`. Display names silently fall back to the raw id. Sort `scan_concept`'s hits by synonym index before taking the first.
- **L-3 — Duplicate synonym.** `"supports_tools"` appears twice in `declared.rs:131-139`.
- **L-4 — Substring collision in `modalities_from_strings`.** `engine.rs:735-758` checks `n.contains("text")`; `"context"` contains `"text"`. Verified: `modalities_from_strings(["context"])` → `[Text]`. Match on word boundaries or exact tokens.
- **L-5 — `can_enable` is hardcoded.** `probe.rs:708` sets `can_enable: true` whenever a model emits reasoning, without ever testing it. Should be measured like `disable_spelling` is.
- **L-6 — Vision probe uses a weaker standard than the tools probe.** `probe_vision` (`probe.rs:421-437`) accepts any HTTP 200 as proof of image support, while `probe_tools` correctly requires an actual `tool_calls` entry. Many gateways accept `image_url` and ignore it. Apply the same "observe, don't assume" standard.
- **L-7 — `probe_streaming` ignores config gating.** `engine.rs:500` calls it unconditionally; every other probe is behind a `DiscoveryConfig` flag.
- **L-8 — Output modalities are always labelled `Probed`.** `engine.rs:618` uses `Fact::probed(...)` even when the value came purely from a declaration, corrupting the provenance this crate's design depends on.
- **L-9 — `flatten_into` has no depth cap** (`declared.rs:363`), unlike `walk`'s `MAX_DEPTH = 8`. Currently bounded only by serde_json's 128-level parse limit.
- **L-10 — Key passed on the command line.** `examples/discover.rs:33` takes `--key` as an argv value, exposing it via the process list and shell history. Prefer `--key-env VAR` or `--key-file`.
- **L-11 — Example duplicates engine logic.** `examples/discover.rs:137-156` re-implements listing, `--only` filtering and `--limit` truncation that `DiscoveryEngine::discover` already owns. This will drift.
- **L-12 — `drain_streams` busy-waits and can panic.** `ai-sidecar/src/lib.rs:575-584` uses `std::thread::sleep` in a poll loop and `expect("streams lock not poisoned")`. Both are hostile if called from async context. Use a `Condvar`/`tokio::time::timeout` and return a `Result`.
- **L-13 — `classify` misses 402 and 408.** `errors.rs:227-261` maps 401 and 429 but lets `402 Payment Required` and `408 Request Timeout` fall through to `Other`, so 408 is not retryable.
- **L-14 — Anthropic keeps the blanket vision claim.** `anthropic.rs` still hardcodes `supports_vision: true` for unknown models, contradicting the "no capability without evidence" rule applied in `openai_compat.rs` and `gemini.rs`. Also note the unknown-model output ceiling dropped from `64_000` to `MODEL_MAX_OUTPUT_TOKENS = 8_192`; if anything derives request limits from `ModelInfo`, that is a regression.
- **L-15 — Timeout raised 30 s → 90 s** (`openai_compat.rs:69`) with no validation or changelog note.
- **L-16 — `Instant::now() - Duration::from_secs(60)`** (`probe.rs:174-176`) can theoretically panic on underflow. Use `checked_sub(...).unwrap_or_else(Instant::now)` or `Option<Instant>`.
- **L-17 — Test name typo:** `allen_null_fields_are_not_capabilities` (`response.rs:310`).
- **L-18 — Internal host in a source comment:** `openai_compat.rs:896` documents `opencode/ncnio zen at 127.0.0.1:5664`. Keep environment specifics out of committed code comments.
- **L-19 — No CHANGELOG entry** for a new public crate, a new public enum variant, or a significant provider behavior change, against this repo's established convention.
- **L-20 — 11 MB binary in the tree:** `tools/discovery-probe/discover_old.exe`. Should be removed, not committed.

---

## Recommended order of work

1. **Rotate the two exposed API keys**, scrub the files, extend `.gitignore`. *(C-1)*
2. Hand-write `Debug` for `Transport`. *(C-2)*
3. Validate `base_url` scheme and pin redirects. *(H-1)*
4. Fix the `probe_context_window` abort path — stop overwriting evidence, return `None` on abort. *(H-2)*
5. Fix the dead fallback filter in `normalize_message`. *(M-1)*
6. Tighten `mine_limits` (M-2, M-3, M-4) and `has_feature`/`extract_vision` (M-5).
7. Make the harness deterministic and unify the capability code paths (M-6, M-7).
8. Resolve `ReasoningEffort::Max` (M-8) and work through the Low list.
