# Live Discovery Campaign — Chronological Journal

**Campaign:** prove the SDK is fully general across three live gateways
**Started:** 2026-08-29 01:48 IST
**Providers:** b.ai (44 models), NVIDIA (83 models), SenseNova (4 models)

Keys were supplied by the user for this campaign and are referenced only via
shell environment in the run commands. They are **never** written into this
journal or any committed file. (See finding C-1 in `AUDIT-REPORT.md`: the same
keys already sit in plaintext in `tools/discovery-probe/*.json` and must be
rotated.)

---

## 2026-08-29 01:48 — Campaign start

Decision: fix the confirmed static-audit defects **before** the live campaign.
Rationale: running known-broken code would produce measurements I would then
have to discard and re-collect, and the live gateways are rate-limited.

## 2026-08-29 01:49–01:52 — Defects fixed before live testing

| ID | Fix | File |
|---|---|---|
| C-2 | Hand-written `Debug` for `Transport`, redacts `api_key` | `probe.rs` |
| H-1 | `validate_base_url`: reject non-`https` (loopback excepted); `redirect::Policy::none()` | `probe.rs` |
| H-2 | `probe_context_window` evidence now **appends**; abort is stated explicitly; engine confidence now 0.2/0.4/0.7 by evidence quality | `probe.rs`, `engine.rs` |
| M-1 | `normalize_message` fallback filters on the real value, not `Value::Null` | `response.rs` |
| M-2 | `mine_limits` requires a token/char unit; bare `"up to "` pattern removed | `errors.rs` |
| M-3 | `number_after` tolerates thousands separators | `errors.rs` |
| M-4 | `bracket_range` scans every bracket pair | `errors.rs` |
| M-5 | `has_feature` returns `Some` only on a positive hit | `declared.rs` |
| L-4 | `modalities_from_strings` matches word tokens, not substrings | `engine.rs` |
| L-16 | `Instant - Duration` → `checked_sub` | `probe.rs` |
| M-6 | Mock advertises `Connection: close` | `edge_harness.rs` |

Verification: 68/68 tests pass; 3 consecutive parallel runs of the previously
flaky harness are clean; `clippy --all-targets` silent.

## 2026-08-29 01:53 — Enumeration: all three gateways reachable

| Gateway | HTTP | Models | Metadata published |
|---|---|---|---|
| b.ai | 200 | 44 | `id`, `object`, `created`, `owned_by`, **`supported_endpoint_types`** |
| NVIDIA | 200 | 83 | `id`, `object`, `created`, `owned_by` — **nothing else** |
| SenseNova | 200 | 4 | 18 fields incl. `context_length`, `input_modalities`, `max_output_length`, `supported_features` |

### Issue L-01 — Requested b.ai model IDs do not exist as specified (severity: **High**)

You specified `DeepSeek-V4-Flash` and `DeepSeek-V4-Flash-Vision-Exp`.
`GET /v1/models` returns **`deepseek-v4-flash`** and
**`deepseek-v4-flash-vision-exp`** — lowercase. The other four (`hy3`,
`mimo-v2.5`, `glm-5.3-flash`, `qwen3.8-flash`) matched exactly.

**Root cause:** b.ai's catalog is lowercase and the gateway does no
case-folding. Any consumer that hardcodes the catalog spelling the *vendor
documentation* uses will miss. Worse, `ai_models::default_catalog().get(...)`
and `DiscoveryConfig.extra_models` both do **exact string matching**, so
`provider.model("DeepSeek-V4-Flash")` silently returns an "unknown model"
fallback rather than an error.

**This is the single most user-visible generality gap found so far:** the SDK
cannot resolve a model id that differs only in case, and it fails *silently*
rather than reporting "no such model (closest match: …)".

**Recommendation:** normalise on a case-insensitive key for lookup, keep the
gateway's canonical spelling for requests, and surface a "did you mean" error
instead of a silent fallback.

### Issue L-02 — b.ai publishes `supported_endpoint_types`, which the SDK ignores (severity: **Medium**, architectural)

All 44 b.ai entries carry `"supported_endpoint_types": ["openai","anthropic"]`.
This is a *declared* fact stating which wire protocols the model speaks —
exactly the kind of thing the `declared` layer exists to harvest.

**Root cause:** `declared::Concept` has no endpoint/protocol concept. The
synonym registry covers context, output, modalities, features, vision, tools,
streaming, structured output, embeddings, reasoning, fine-tuning, name,
description, created, owner, pricing — but not protocol. The SDK therefore
*probes* for what the gateway already *told* it.

**Impact:** an SDK aiming to be "fully general" must negotiate protocol
(OpenAI-chat vs Anthropic-messages vs Gemini) per model. b.ai hands that over
for free and it is being discarded.

**Recommendation:** add `Concept::EndpointTypes` (synonyms:
`supported_endpoint_types`, `endpoint_types`, `supported_endpoints`,
`protocols`, `api_types`) and surface it on `DiscoveredModel` as
`wire_protocols: Fact<Vec<String>>`.

### Issue L-03 — SenseNova list mixes chat and image-generation models undifferentiated (severity: **Medium**)

| Model | `input_modalities` | `output_modalities` | `supported_features` |
|---|---|---|---|
| `sensenova-6.7-flash-lite` | `["text","image"]` | `["text"]` | tools, json_mode, reasoning |
| `sensenova-6.8-flash-lite` | `["text","image"]` | `["text"]` | tools, json_mode, reasoning |
| `sensenova-u1-fast` | `["text"]` | **`["image"]`** | tools, json_mode, reasoning |
| `sensenova-u1.5-lite` | `["text"]` | **`["image"]`** | tools, json_mode, reasoning |

Two of the four models in a `/v1/models` list are **image generators**
(declared `output_modalities: ["image"]`), yet all four declare the identical
`supported_features` set including `tools` and `reasoning`.

**Root cause (gateway side):** SenseNova emits one feature list for every model
rather than per-model truth. **Root cause (SDK side):** `ModelRole` currently
has no `ImageGeneration` path for a model that *is listed* on `/models` —
`discover_one` hardcodes `role = ModelRole::Chat` the moment
`/chat/completions` returns 2xx, and only falls back to `route_discover` when
chat *fails*. A model that answers chat with a 2xx but whose declared output
modality is `image` will be typed as Chat.

**Impact:** the SDK's declared-vs-probed reconciliation produces a *false
positive*: it will accept `output_modalities` at face value while typing the
model as Chat. The declared output modality is never checked against the
routed role.

### Issue L-04 — SenseNova declares a uniform 262144/65536 for all four models (severity: **Medium**, gateway-side)

All four — including the two image generators — declare
`context_length: 262144` and `max_output_length: 65536`.

**Assessment:** this is precisely the "declared metadata is present but not
trustworthy" case the crate documents. It is only detectable empirically. This
campaign should confirm it by probing `max_output_tokens`.

---

## Phase 2 — triage results and role discovery

### Issue L-05 — a transient 429 permanently poisons a working model (severity: **High**)

**Observed 02:05.** b.ai triage reported `mimo-v2.5` and
`deepseek-v4-flash-vision-exp` as
`throttled during discovery — result inconclusive`, with `latency = 125 ms`
and `accepted_endpoints = []`.

Three checks prove the verdict was wrong:

1. `curl` against both models, sequentially → **HTTP 200** (1.7 s, 2.9 s).
2. `curl` × **6 concurrent** requests to `mimo-v2.5` → **all six 200**.
   b.ai was not rate-limiting at all.
3. Serial re-run through the SDK → **2/2 reachable, role Chat**.

**Root cause.** Two independent design decisions compound:

- `engine.rs:371` — `probe_reachable` uses `Transport::post`, which retries
  only `max_attempts` times with a short backoff. Two attempts were consumed
  in 125 ms, i.e. both were rejected at the edge, well before b.ai's ~2 s
  model latency. A rejection that fast is not an invocation.
- `engine.rs:363` — `if config.probe_endpoints && !matches!(class, Some(ErrorClass::RateLimited))`
  **skips route discovery entirely** for `RateLimited`. The intent (do not add
  load to a gateway that is already throttling) is sound, but the consequence
  is that `role` stays `Unknown` and `accepted_endpoints` stays empty.
- There is **no deferred retry pass**. Once a model is finalized as
  `reachable: false`, nothing ever revisits it.

**Impact.** Any gateway that emits even one spurious 429 loses that model for
the whole run, silently. On a large catalog this is not a rare event.

**Fix.** Rate-limited models must be **re-queued**, not finalized. Record them
as `reachable: false` with `blocker = RateLimited` *and* a `retry_hint`, then
run a low-concurrency sweep at the end of the run to settle them.

### Issue L-06 — the reachability probe is text-only, so modality-restricted models are reported as dead (severity: **Critical**)

**Observed 02:17.** `nvidia/nemotron-parse` was reported as
`bad_request: Content cannot be a plain string. The model does not support
text input.`, `role: Unknown`, `accepted_endpoints: []`.

It is not dead. It works. Empirically:

| Content shape | Result |
|---|---|
| `"content": "hi"` (plain string) | 400 — *does not support text input* |
| `[{"type":"text","text":"hi"}]` | 400 — *does not support text input* |
| `[{"type":"image_url",...}]` (image only) | **200** — real completion |
| `[{"type":"image_url",...},{"type":"text","text":" "}]` | **200** — real completion |

The model is **vision-only**: it accepts an image part and no meaningful text
part. Its answer arrives as a **tool call** (`markdown_bbox`), with
`content: null` — it is a document-parsing model that emits structured output
rather than prose.

**Root cause.** `probe_reachable` (`probe.rs:441`) hardcodes

```rust
"messages": [{"role": "user", "content": "Reply with the single word: OK"}]
```

A single content shape is treated as a proxy for "is this model served here".
`route_discover` (`engine.rs:662`) then probes `embeddings`, `rerank`,
`images/generations`, `audio/speech`, `videos` — but **also with text-only
payloads**, so a modality-restricted chat model is rejected everywhere and
falls through to `ModelRole::Unknown`.

**This is the single largest generality gap in the SDK.** A "no
provider-specific knowledge" SDK cannot assume that every model accepts a
plain-text user message. Vision-only, audio-only, and image-generation models
all violate that assumption, and every one of them is currently reported as
broken.

**Fix.** Reachability must be probed across **content shapes**, not just
endpoints. On a rejection whose message indicates a content/modality
constraint, retry with an image part, then an audio part, before concluding.
The first shape that succeeds defines `input_modalities`.

### Issue L-07 — NVIDIA declares nothing; every capability must be empirical (severity: **Medium**, informational but load-bearing)

Across all 83 NVIDIA models, `declared::flatten` yields exactly four keys,
present on **every** model:

```
83  $.created
83  $.id
83  $.object
83  $.owned_by
```

No context window, no max output, no modalities, no feature flags. Compare
SenseNova (rich but untrustworthy, L-04) and b.ai (sparse).

**Consequence:** the declared layer contributes nothing for NVIDIA, so
`context_window` / `max_output_tokens` can only come from probing. A triage
pass that disables probes therefore reports `ctx=0 out=0` for every model —
which is indistinguishable from "unknown" in the current output schema unless
the reader inspects `source`. **The JSON should make `Unknown`-sourced zero
visibly different from a measured zero.**

### Issue L-08 — a fixed 30 s timeout misclassifies slow models as dead (severity: **High**)

**Observed 02:25.** Six NVIDIA models failed triage with
`no response within 30s`. Re-running the same six with `--timeout 120`:

| Model | 30 s | 120 s |
|---|---|---|
| `deepseek-ai/deepseek-v4-flash-0731` | timeout | **OK** |
| `meta/llama-3.2-90b-vision-instruct` | timeout | **OK** |
| `mistralai/mistral-nemotron` | timeout | **OK** |
| `meta/llama-guard-4-12b` | timeout | timeout |
| `moonshotai/kimi-k3` | timeout | timeout |
| `nvidia/nemotron-3.5-lightning-30b-a3b` | timeout | network error |

Three of six — **half** — were live. Also observed in the triage run:
`poolside/laguna-xs-2.1` answered in **28 s**, one second inside the limit.

**Root cause.** The timeout is a constant with no relation to observed
latency. A cold NIM container takes tens of seconds to serve its first
request; 30 s is below the cold-start latency of this gateway.

**Fix.** Two-tier: a short *first-byte* deadline to detect a live socket, and
a much longer *completion* deadline. A model that has already produced tokens
must never be abandoned. Additionally, keep a running p95 of observed latency
per provider and derive the deadline from it instead of a constant.

### Issue L-09 — HTTP 410 Gone (model retired) is unclassified (severity: **Medium**)

`nvidia/llama-3.3-nemotron-super-49b-v1` returns:

```json
{"type":"about:blank","title":"Gone","status":410,
 "detail":"The model '...' has reached its end of life on 2026-08-26T09:00:00Z and is no longer available."}
```

`classify` has no 410 branch. The chain ends at `status >= 500 → ServerError`
else `Other`, so a **permanent, non-retryable** retirement is bucketed as
`Other`. `is_retryable()` may then burn attempts on a model that can never
come back.

**Fix.** Add `ErrorClass::Gone` (or `Retired`) for 410. It is the strongest
possible negative signal — stronger than 404 — and must be surfaced verbatim,
since the body carries the retirement date.

### Issue L-10 — `--triage` silently disables endpoint routing (severity: **Medium**, tooling defect, self-inflicted)

`discover.rs:145` sets `probe_endpoints: !args.no_endpoints && !args.triage`.
Combined with `engine.rs:363`, this means **no triage run ever performs role
discovery**, so every non-chat model in the triage output has
`role: Unknown` and `accepted_endpoints: []` — which reads as "broken" but
actually means "never asked".

**This invalidated my own first-pass conclusion** that NVIDIA's nine embedding
models were dead. They were never probed on `/embeddings`.

**Fix.** Triage should disable only the *capability battery*
(vision/tools/structured/thinking/context), never role routing — routing is
what makes triage meaningful in the first place.

### Issue L-11 — embedding models: real, served, and missed (severity: **High**)

Nine NVIDIA models returned the plain-text Go default `404 page not found`
from `/chat/completions` — a correct ModelNotFound. Listing them reveals what
they are:

```
bigcode/starcoder2-15b
nvidia/embed-qa-4
nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1
nvidia/llama-3.2-nv-embedqa-1b-v1
nvidia/llama-nemotron-embed-vl-1b-v2
nvidia/nemotron-3-embed-1b
nvidia/nv-embedqa-mistral-7b-v2
nvidia/nvclip
snowflake/arctic-embed-l
```

**All nine are embedding models.** They are listed in `/v1/models`, served by
the gateway, and simply not available on the chat endpoint. Direct test:

| Model | `/v1/embeddings` |
|---|---|
| `nvidia/nemotron-3-embed-1b` | **200** — real float vector |
| `snowflake/arctic-embed-l` | 404 (account entitlement) |
| `nvidia/nvclip` | 404 (account entitlement) |

So `/v1/embeddings` demonstrably works for `nemotron-3-embed-1b`, and the SDK
would have found it — had routing been enabled (L-10). The dual failure of
L-10 + a text-only probe (L-06) means the SDK currently reports **every
non-chat model in a catalog as broken**.

**Fix.** Routing must never be optional by default, and the router must treat
a 404 on chat as *positive evidence* that the model lives elsewhere, not as a
verdict.

### Issue L-12 — raw rustls diagnostics leak into user-facing output (severity: **Low**)

`nvidia/nemotron-3.5-lightning-30b-a3b` produced:

```
network: error sending request for url (https://integrate.api.nvidia.com/v1/chat/completions):
client error (SendRequest): connection error: peer closed connection without
sending TLS close_notify: https://docs.rs/rustls/latest/rustls/manual/_03_howto/index.html#unexpected-eof
```

A `docs.rs` hyperlink for a Rust TLS library is meaningless to an SDK
consumer, and the string is unstable across reqwest versions.

**Fix.** Map transport errors to a stable taxonomy
(`TlsUnexpectedEof`, `DnsFailure`, `ConnectionReset`, …) and keep the raw
string in a separate `debug` field.

### Issue L-13 — entitlement 404s are not absence (severity: **Medium**)

45 of 83 NVIDIA models return:

```json
{"status":404,"title":"Not Found",
 "detail":"Function '<uuid>': Not found for account '<account-id>'"}
```

Verified by direct `curl` — genuinely 404, not an SDK artifact. But the
semantics are **entitlement**, not absence: the model exists on the gateway
and is not enabled for this key. Reporting it as
`not served by this gateway` conflates "this provider has no such model" with
"you have not been granted this model" — and the latter is actionable.

**Fix.** Add `ErrorClass::NotEntitled` (or a boolean on `ModelNotFound`)
triggered by an account/entitlement mention, so the SDK can tell the user to
request access rather than to pick a different model.

---

## Phase 3 — hidden defects found while fixing

### Issue L-14 — capability probes ignore the accepted content shape (severity: **High**)

After the L-06 fix, `nvidia/nemotron-parse` is correctly discovered as
`[chat] in=I`. But the same run reports **`tools=n`**.

That is false. Direct observation of the model's actual response:

```json
"message":{"role":"assistant","content":null,
 "tool_calls":[{"id":"chatcmpl-tool-…","type":"function",
 "function":{"name":"markdown_bbox","arguments":"…"}}]}
```

The model **emits a tool call** — it is a document parser whose entire
output mechanism is a tool call. The probe says it does not support tools.

**Root cause.** `probe_tools`, `probe_structured_output`, `probe_thinking`,
`probe_max_output` and `probe_context_window` all build their own payload
with a plain-text user message. Only *reachability* was made shape-aware.
For a vision-only model every one of those probes is rejected on the same
grounds as the original reachability probe, and the rejection is recorded
as "capability absent" rather than "probe inapplicable".

**This is the same defect as L-06, one layer up.** L-06 fixed *whether the
model exists*; L-14 is about *what the model can do*. A capability verdict
produced by a payload the model structurally cannot accept is not a
negative result — it is a failed experiment, and must be reported as such
(`None` + an anomaly), not as `false`.

**Fix.** Thread the accepted `ContentShape` through every capability probe.
Where a probe cannot be expressed in the accepted shape, record the
capability as *unknown* with the reason, never as *absent*.

### Issue L-15 — `reachable` and `blocker` contradict `accepted_endpoints` (severity: **High**)

After enabling routing, `nvidia/nemotron-3-embed-1b` produced this record:

```
reachable         : False
role              : Embedding
accepted_endpoints: ['embeddings']
blocker           : not served by this gateway (404 page not found)
```

The model **is** served — `accepted_endpoints` says so, and a direct
`POST /v1/embeddings` returns a real float vector. Yet the report says
"not served by this gateway" and `reachable: false`.

**Root cause.** `reachable` is defined purely as "the chat probe returned
2xx", and `blocker` is derived from the chat error without consulting the
routing outcome. When routing succeeds, the two fields are computed
independently and contradict each other.

**Consequence.** Any consumer filtering on `reachable == true` silently
discards every embedding, reranker, TTS and image model in the catalog —
i.e. precisely the models L-06/L-11 were trying to rescue.

**Fix (applied).** When `accepted_endpoints` is non-empty the model is
served; the blocker now reads
`not available on chat/completions; served on embeddings instead (role: Embedding)`.
The deeper fix is to split the flag into `chat_reachable` and `served`.

### Issue L-16 — the embedded probe image is a corrupt PNG (severity: **Critical**, hidden)

Chasing why the image-shape fallback still failed, I replayed the crate's
exact payload against NVIDIA and got:

```json
{"object":"error","message":"broken data stream when reading image file",
 "type":"InternalServerError","code":500}
```

`TINY_PNG_B64` was **not a valid PNG**. Chunk-level analysis:

```
--- crate TINY_PNG_B64 (70 bytes) ---
  IHDR  len=13   crc=OK
  IDAT  len=13   crc=MISMATCH zlib=FAIL(error)
  IEND  len=0    crc=OK
  VERDICT: CORRUPT
```

Valid signature, valid base64, **bad IDAT CRC and an undecodable zlib
stream**. The byte sequence had been mistyped at some point, and the only
test guarding it was:

```rust
fn tiny_png_is_valid_base64() {
    let bytes = decode(TINY_PNG_B64).expect("valid base64");
    assert_eq!(&bytes[1..4], b"PNG");
}
```

which checks that the bytes decode and contain "PNG" — both true of the
corrupt constant. The test could never fail.

**Blast radius.** `probe_vision` uses this constant for **every** vision
check on **every** provider. Behaviour diverges by gateway:

- Gateways that validate image bytes (NVIDIA) → 500 → currently treated as
  *inconclusive* (5xx), so vision is silently **unknown**, never reported.
- Gateways that reject with 400 → a **false negative** on a vision model.

No provider has ever had its vision capability correctly confirmed by this
crate. Every `vision=false` in every prior report is untrustworthy.

**Fix (applied).** Replaced with a verified 1×1 PNG (all chunk CRCs valid,
IDAT inflates). Replaced the test with `tiny_png_is_a_decodable_png`, which
walks the chunk structure, verifies every CRC, checks the zlib header, and
asserts 1×1 dimensions.

**Note:** the copy of `TINY_PNG_B64` in `crates/ai-computer` is a different
constant and **is valid** — only `ai-discovery`'s was corrupt.

**Lesson.** A test that asserts a weaker property than the one you depend
on is worse than no test: it certifies the defect.

---

### Issue L-17 — b.ai enforces 10 requests per 60 s, and the SDK has no budget awareness (severity: **Critical**)

**Observed 02:37.** The full battery over b.ai's six models returned **1/6
reachable**. `hy3` succeeded; every subsequent model failed in ~1 s with
`throttled during discovery — result inconclusive`.

A 24-request burst at concurrency 3:

```
req1..req10  = 200   (all ~2-3 s)
req11..req24 = 429   (all ~1-2 s)
```

The cutover is exactly at **10 requests**. The window then resets: after a
70 s wait, five consecutive requests all returned 200.

The 429 itself carries **no information at all**:

```
HTTP/1.1 429 Too Many Requests
Content-Length: 0
X-Oneapi-Request-Id: 20260828211231613413409c955d568JCBK6Xt2
```

Empty body, no `Retry-After`, no `X-RateLimit-*` headers. The SDK's
empty-envelope handling correctly turns this into `RateLimited` — that part
is right — but there is nothing to adapt *to*, so the only available
strategy is to infer the quota from observation.

**Why this matters architecturally.** The full capability battery costs
~17 requests per model. b.ai's quota is 10 per minute. Those two numbers
are irreconcilable: **a full discovery pass is mathematically impossible on
b.ai within its quota.** No amount of retry tuning fixes an arithmetic
contradiction — and per L-05, every model that hits it is silently
recorded as unreachable.

**Fix required.** The SDK must treat requests as a budget and derive the
per-provider rate from observation, not configuration:

1. Maintain a per-provider token bucket seeded conservatively.
2. On the first 429, **halve the inferred rate** and back off for a full
   inferred window rather than a fixed sleep.
3. **Defer, do not finalize** — queue throttled models for a later pass
   (this is the same mechanism L-05 needs).
4. Order probes by value-per-request: reachability → role routing →
   streaming → output ceiling → context window, and skip the rest when the
   budget is short. A model with role + reachability known is far more
   useful than a model marked dead.

### Issue L-18 — the context probe mixes two token scales in one record (severity: **Medium**)

After the SenseNova full run, `sensenova-6.7-flash-lite` reported:

```
ctx value    = 350
ctx evidence = "512 ok; SEARCH ABORTED at 16640 (non-context failure);
                512 is the last size confirmed accepted — LOWER BOUND"
```

The value and the evidence disagree, and neither is wrong — they are in
**different units**.

**Root cause.** `probe_context_window` tracks two different quantities and
reports one while describing the other:

- `lo` — the *nominal* token size we asked `filler_prompt(n)` to generate.
- `best` — the *measured* `usage.prompt_tokens` the gateway actually
  reported.

On abort the function returns `best` (350 measured) but the evidence string
described `lo` (512 nominal). The gap is the error in the filler's
chars-per-token estimate, observed on SenseNova at **1.46x** (512→350) and
**1.99x** (28736→14462).

**Fix (applied).** The abort branch now names both scales explicitly:

```
largest accepted request was 512 nominal tokens, which the gateway counted
as 350 prompt_tokens — LOWER BOUND of 350, not a measurement
```

**Wider point.** The nominal/measured divergence means the binary search is
stepping through sizes that do not mean what the code thinks they mean. The
search should be defined over *measured* tokens throughout, using each
response's `prompt_tokens` to recalibrate the filler before the next step.

---

## SenseNova results (full battery)

| Model | Role | Reach | Input | ctx | out | thinking | tools | JSON |
|---|---|---|---|---|---|---|---|---|
| `sensenova-6.7-flash-lite` | Chat | yes | **T+I** | 350 ¹ | 65536 ² | on; off via `thinking.type=disabled` | n | y |
| `sensenova-6.8-flash-lite` | Chat | yes | **T+I** | 14462 ¹ | 65536 ² | on; off via `thinking.type=disabled` | y | y |
| `sensenova-u1-fast` | Unknown | no | T | 262144 ³ | 65536 ³ | — | — | — |
| `sensenova-u1.5-lite` | **ImageGeneration** | no | T | 262144 ³ | 65536 ³ | — | — | — |

¹ `Probed`, confidence **0.2** — search aborted on a non-context failure; a
lower bound, not a measurement (see L-18). Compare the declared **262144**:
L-04's "declared metadata is not trustworthy" is now empirically confirmed,
though these bounds are too weak to be actionable on their own.

² `Inferred`, confidence 0.95 — probe agrees with the declaration and the
bracket `[1, 65536]` was mined from a rejection. This path works well.

³ `Declared` only — the model was never reached, so nothing was probed.

**Positive results.** Thinking-toggle detection is genuinely working: both
live models were found to emit reasoning and to honour
`thinking.type=disabled` — discovered by observation, with no
provider-specific knowledge (`reasoning_effort=none` and
`enable_thinking=false` were tried and rejected). Endpoint routing also
works: `sensenova-u1.5-lite` was correctly typed **ImageGeneration**.

**Two remaining anomalies on SenseNova:**

- Both live models report
  `HTTP 200 with no answer: BudgetConsumedByReasoning` — with 64 of 64
  completion tokens spent on reasoning. The model returned success while
  producing nothing usable. Correctly diagnosed, but the SDK should treat
  "every completion token went to reasoning" as a *hint about default
  reasoning budget*, not merely an anomaly.
- `sensenova-u1-fast` records
  `images/generations routes to this model but returned 500 (internal error)`
  — routing found it, the backend is unhealthy. Per J-017 this is correctly
  recorded as inconclusive rather than negative.

---

### Issue L-19 — a throttled probe silently flips a capability verdict (severity: **Critical**)

**Observed 03:00.** Running the six b.ai models in quota-sized pairs, the
first model of each pair succeeded and the second always throttled — the
battery on one model consumes the whole 10-request quota. Comparing `hy3`
across two runs:

| Run | `hy3` thinking verdict |
|---|---|
| full sweep, quota already exhausted afterwards | `thinking(on, off via thinking.type=disabled)` |
| staggered sweep, first in its pair | `thinking(on, no-off-switch)` |

The same model, same gateway, two different answers.

**Root cause.** A probe that fails with a 429 is *not* distinguishable, at
the call site, from a probe that succeeded and returned no reasoning field.
The thinking probe votes across several spellings; when some of those
requests are throttled, the vote changes. The result is a **confident,
wrong capability verdict with no indication that the evidence was
incomplete.**

**This is the most dangerous failure mode in the whole system.** It is
worse than reporting "unknown": the record looks authoritative, has
`source: Probed`, and is simply false. A consumer will route on it.

**Fix.** Any capability verdict built from probes where **any** constituent
request failed for a retryable reason (429/5xx/timeout/network) must be
demoted: record the capability as `None` with
`evidence = "inconclusive: N of M probes were throttled"`. The provenance
layer must carry a `degraded: bool` alongside `source`, and confidence must
be scaled by the fraction of probes that actually completed.

---

## b.ai results (staggered, quota-respecting)

| Model | Role | Reach | Input | thinking | tools | JSON |
|---|---|---|---|---|---|---|
| `hy3` | Chat | yes | T+I | on, no off-switch | y | n |
| `glm-5.3-flash` | Chat | yes | T+I | on, no off-switch | y | y |
| `deepseek-v4-flash` | Chat | yes | T+I | on; **off via `thinking.type=disabled`** | y | y |
| `mimo-v2.5` | — | throttled | — | — | — | — |
| `qwen3.8-flash` | — | throttled | — | — | — | — |
| `deepseek-v4-flash-vision-exp` | — | throttled | — | — | — | — |

Three of six were characterised. The other three were never throttled by
*their* behaviour — they were throttled by the cost of the models probed
before them. Every one of them was confirmed working by direct `curl`
earlier in this campaign (02:11: six concurrent requests to `mimo-v2.5`,
all 200).

Note that `deepseek-v4-flash` **was** found to honour
`thinking.type=disabled` while `hy3` and `glm-5.3-flash` were not — a
genuine per-model difference discovered empirically, with no
provider-specific knowledge. That is the design working as intended.

---

## Architectural recommendations

Ordered by how much generality each one buys.

1. **Budget-aware scheduling (L-05, L-17).** Treat requests as a currency.
   Infer each provider's rate from the first 429, halve it, and back off
   for a full inferred window. Order probes by value-per-request and skip
   the rest when the budget is short. Without this, any gateway with a
   quota smaller than the battery simply cannot be discovered.

2. **Defer, never finalize (L-05, L-19).** A model that failed for a
   *retryable* reason must be re-queued, not written off. One
   low-concurrency settlement pass at the end of a run converts most
   "unreachable" verdicts into real results.

3. **Shape-aware probing everywhere (L-06, L-14).** Reachability is now
   shape-aware; every *capability* probe must be too. A verdict produced by
   a payload the model structurally cannot accept is a failed experiment,
   not a negative result.

4. **Degraded provenance (L-19).** `source: Probed` must not be emitted
   when some constituent probes failed. Add `degraded` and scale confidence
   by the completion fraction.

5. **Separate `served` from `chat_reachable` (L-15).** One boolean cannot
   express "not a chat model, but served on /embeddings". Partially fixed;
   the flag itself should be split.

6. **Latency-derived deadlines (L-08).** Keep a per-provider p95 and derive
   timeouts from it. A constant cannot serve both a 200 ms gateway and a
   28 s cold-starting NIM container. Distinguish a first-byte deadline from
   a completion deadline — never abandon a model that is already streaming.

7. **Verify test fixtures (L-16).** A test asserting a weaker property than
   the one depended on certifies the defect. Any embedded binary fixture
   needs a test that actually parses it.

---

## Run log

| Time | Event |
|---|---|
| 01:54 | NVIDIA full sweep launched (83 models, concurrency 3, default pacing) |
| 01:57 | b.ai sweep launched (6 requested models, concurrency 2) — first attempt at 01:54 with `--policy conservative` exceeded a 15-minute cap before writing output; relaunched with default pacing |
| 01:57 | SenseNova full sweep launched (4 models, concurrency 2) |
| 02:05 | SenseNova triage: **2/4 reachable**. Both image generators (`sensenova-u1-fast`, `sensenova-u1.5-lite`) reported `not served by this gateway` |
| 02:05 | b.ai triage: **4/6 reachable**. `mimo-v2.5` and `deepseek-v4-flash-vision-exp` reported *throttled* (→ L-05) |
| 02:11 | b.ai concurrency test: 6 parallel requests → all 200. Throttle verdict disproved |
| 02:16 | b.ai serial re-run → **2/2 reachable**. L-05 confirmed as a false negative |
| 02:17 | `nvidia/nemotron-parse` confirmed live via image-only content part (→ L-06) |
| 02:22 | NVIDIA triage: **19/83 reachable**. Failure breakdown: 45 entitlement 404s, 9 plain-text 404s (all embeddings), 6 timeouts, 2 server errors, 1 bad-request |
| 02:25 | Slow-model retest at 120 s: **3 of 6 timeouts were live** (→ L-08) |
| 02:28 | Route-discovery retest launched across 58 non-chat NVIDIA models with routing enabled |
| 02:32 | **L-16 found:** `TINY_PNG_B64` is a corrupt PNG (bad IDAT CRC, zlib fails). Every vision probe in the crate was sending an unopenable file; the guarding test could not fail |
| 02:33 | SenseNova full battery: **2/4 reachable**, both `in=TI`, thinking toggle detected empirically. `u1.5-lite` correctly typed ImageGeneration |
| 02:33 | b.ai full battery: **1/6** — `hy3` only; the other five throttled. Traced to a hard 10-request quota (→ L-17) |
| 02:41 | **L-06 fix verified live:** `nvidia/nemotron-parse` now reports `[chat] in=I` and `reachable: true` (was: unreachable, bad_request) |
| 02:42 | **L-14 found:** the same run reports `tools=n` for `nemotron-parse`, which demonstrably emits a `markdown_bbox` tool call — capability probes are not shape-aware |
| 02:43 | **L-15 found and fixed:** `nemotron-3-embed-1b` reported `reachable: false` + "not served" while `accepted_endpoints` read `["embeddings"]` |
| 02:52 | b.ai quota measured precisely: 10 requests → 429, full recovery after 70 s; 429 body is empty with no Retry-After |
| 02:55 | **L-18 found and fixed:** context probe reported `ctx=350` with evidence "512 accepted" — nominal vs measured token scales mixed |
| 02:58 | Full suite green: **68 tests pass** (39 + 29) |
| 03:00 | b.ai staggered sweep (2 models per quota window): **3/6 characterised**. First model of each pair succeeds, second always throttles — one model's battery consumes the whole quota |
| 03:00 | **L-19 found:** `hy3` reported `thinking.type=disabled` in one run and `no-off-switch` in the next. Throttled probes silently flip capability verdicts |
