# Universal Model Discovery — Engineering Journal

**Goal:** make the SDK fully general — auto-fetch models from any provider,
handle any model type dynamically, and auto-detect capabilities (context
length, input/output token limits, thinking toggle) with **no provider or
model specifics hardcoded**, and with every result traceable to root cause.

**Test subjects (all live, real credentials):**

| # | Provider  | Base URL                              | Models listed |
|---|-----------|---------------------------------------|---------------|
| 1 | b.ai      | `https://api.b.ai/v1`                 | 44            |
| 2 | NVIDIA    | `https://integrate.api.nvidia.com/v1` | 83            |
| 3 | SenseNova | `https://token.sensenova.ai/v1`       | 4             |

**Convention:** times are local (IST, UTC+5:30), 2026-08-28. Every entry
records *symptom → root cause → fix*, because a bug that is only described is
a bug that will come back.

---

## 21:23 — Workspace survey

Existing Rust workspace, 28 crates, `cargo check --workspace` green. Prior
uncommitted work had already replaced the hardcoded `128_000 / 8_192` context
constants in `ai-providers::openai_compat` with per-entry parsing, and had
added a `ReasoningEffort::Max` variant.

**Observation:** the prior fix parses `/v1/models` entries for fields named
`context_window`, `max_output_tokens`, `supports_vision` … That is a good step
but it assumes the gateway *publishes* those fields. Verified against the
three providers: 2 of 3 publish none. Parsing alone cannot be the answer.

---

## 21:31 — Ground truth: what each provider actually publishes

Fetched all three `/v1/models` endpoints and unioned the keys.

| Provider  | Field union                                                                 |
|-----------|-----------------------------------------------------------------------------|
| b.ai      | `id`, `object`, `created`, `owned_by`, `supported_endpoint_types`            |
| NVIDIA    | `id`, `object`, `created`, `owned_by`                                        |
| SenseNova | `id`, `name`, `context_length`, `max_output_length`, `input_modalities`, `output_modalities`, `supported_features`, `supported_sampling_parameters`, `pricing`, `quantization`, `description`, `hugging_face_id`, `openrouter`, `datacenters`, … |

### Finding J-001 — `/v1/models` is not a capability source

**Symptom:** b.ai and NVIDIA publish 44 and 83 models respectively with
**zero** capability fields. Any SDK that builds `ModelInfo` from the list
endpoint has nothing to populate.

**Root cause:** the OpenAI `/v1/models` schema only *requires*
`id`/`object`/`created`/`owned_by`. Capability fields are extensions; whether
a gateway emits them is entirely optional.

**Consequence:** capability discovery must be **empirical**, with declared
metadata treated as an *optional prior*, never as the source of truth.

---

## 21:37 — First b.ai sweep (46 ids, 10 concurrent)

Result: `{'http_error': 46}`, i.e. **everything failed**, contradicting a
direct `curl` seconds earlier that returned 200.

### Finding J-002 — b.ai rate-limits with an empty 429 body and no headers

**Symptom:** 40 of 46 requests returned HTTP 429. The response body was
**completely empty** (`Content-Length: 0`). There is no `Retry-After` and no
`X-RateLimit-*` header on any response, successful or not.

**Root cause:** gateway-level throttle that signals purely by status code.

**Why this matters as a bug class:** an error parser that does
`body["error"]["message"]` on an empty body produces either a crash or a
meaningless `None` — which is precisely what my first harness printed
(`{'raw': ''}`). The *status* carried the whole meaning and was being thrown
away in favour of a body that could not be parsed.

**Fix:** classify on status first, body second; an empty body yields a
synthesised message (`HTTP 429 with empty body`) rather than a parse failure.
`errors::classify` + test `empty_body_429_is_rate_limit_not_parse_error`.

**Secondary:** concurrency had to be reduced (10 → 3) and a backoff added
before b.ai could be measured at all. Any discovery run against a
per-account-throttled gateway is unreliable above low concurrency — the
*measurement itself* perturbs the result.

---

## 21:44 — Second b.ai sweep (concurrency 3, backoff)

Result: 4 reachable, 20 × HTTP 403, 1 × HTTP 400, 21 × HTTP 429.

### Finding J-003 — billing failures have no stable status code

**Symptom:** "account has no money" arrives as **three different statuses**:

| Status | `code`                     | Message                                             |
|--------|----------------------------|-----------------------------------------------------|
| 403    | `access_denied`            | `Access restricted. Deposit required to unlock premium models.` |
| 400    | `insufficient_user_quota`  | `credit insufficient balance: balance=0 required=2404` |
| 429    | *(empty body)*             | *(throttle)*                                         |

**Root cause:** billing state is provider-modelled; HTTP has no dedicated
"you owe us money" status that is consistently applied (402 is defined but
almost never used).

**Consequence:** classifying by status alone reports a billing failure as a
bad request or a rate limit, and an SDK that retries "400" or back-offs on
"429" will retry something that can never succeed.

**Fix:** status **and** body vocabulary both feed
`errors::classify`; `insufficient_*quota*`, `balance=0`, `deposit required`
resolve to `ErrorClass::Billing`, which is **not retryable**.

---

## 21:56 — The six user-specified b.ai models

The task named six models: `DeepSeek-V4-Flash`, `DeepSeek-V4-Flash-Vision-Exp`,
`hy3`, `mimo-v2.5`, `glm-5.3-flash`, `qwen3.8-flash`.

### Finding J-004 — model ids are case-sensitive; two of the six do not exist

**Symptom:** `DeepSeek-V4-Flash` and `DeepSeek-V4-Flash-Vision-Exp` → HTTP 404
`The model '…' does not exist (distributor)`. The lowercase
`deepseek-v4-flash` and `deepseek-v4-flash-vision-exp` **do** exist and work.

**Root cause:** the gateway resolves model ids by exact string match. The
capitalised forms are stale/never-existed aliases.

**Consequence for a "general" SDK:** a model id supplied by a user must be
validated against the live list, and the SDK must say *"this id does not
exist, did you mean …?"* rather than surfacing an opaque 404. An SDK that
hardcodes or trusts user-supplied ids fails silently at runtime.

**Fix:** discovery accepts `extra_models` and probes them *as well as* the
listed set, so a stale id is reported as such — with `listed: false` — instead
of being assumed valid.

### Finding J-005 — `hy3` and `glm-5.3-flash` appear broken under load but are not

**Symptom:** both timed out at 45 s in the first sweep; re-tested
sequentially they answered in 1.74 s and 3.84 s.

**Root cause:** not a model fault — queueing behind the throttle from J-002.
Timeout and "model is slow/broken" are indistinguishable in a burst sweep.

**Consequence:** **a burst sweep conflates rate limiting with model failure.**
Any conclusion about a model's health drawn from a parallel sweep against a
throttled gateway is unsound.

**Fix:** re-test every throttled/timeouts model sequentially
(`retest.py`). Final b.ai tally: **6 reachable**, 34 billing-locked,
3 quota-exhausted, 2 nonexistent, 1 still throttled.

---

## 22:00 — SenseNova deep probe

### Finding J-006 — HTTP 200 with no answer at all (the worst failure mode)

**Symptom:** `sensenova-6.7-flash-lite` returns **HTTP 200** with
`finish_reason: "length"`, `usage.completion_tokens: 64`,
`completion_tokens_details.reasoning_tokens: 64`, and a message object
containing **only** `role` and `reasoning`. There is **no `content` key at
all** — not an empty string, the key is absent.

**Root cause:** reasoning-first architecture. The model spends the entire
completion budget on chain-of-thought and is cut off before emitting an
answer. Because `max_tokens` was 64, reasoning consumed 64/64.

**Why this is the most damaging bug in the set:** every layer of a naive SDK
reports success. Status is 200, no exception is raised, and
`choices[0].message.content` is `None`/absent → the application renders an
empty string. There is no error to catch.

**Fix:** `response::normalize_message` + `response::diagnose_empty`, which
distinguish three distinct causes with three distinct remedies:

| Cause                       | Signal                                            | Remedy                    |
|-----------------------------|---------------------------------------------------|---------------------------|
| `BudgetConsumedByReasoning` | reasoning present, `reasoning_tokens >= completion_tokens`, no answer | raise `max_tokens` **or** disable thinking |
| `BudgetTooSmall`            | `finish_reason == "length"`, no reasoning         | raise `max_tokens`        |
| `EmptyByStop`               | `finish_reason == "stop"`, no content             | genuinely empty response  |

Recorded on the model as an anomaly with the token counts attached.

### Finding J-007 — the reasoning field has no standard name

**Symptom:** the chain-of-thought field differs **per provider and per model
on the same provider**:

| Model                       | Reasoning field(s)                        | Answer field |
|-----------------------------|-------------------------------------------|--------------|
| b.ai `deepseek-v4-flash`    | `reasoning_content`                       | `content`    |
| b.ai `mimo-v2.5`            | `reasoning`, `reasoning_details`, `refusal` | `content`  |
| SenseNova `6.7-flash-lite`  | `reasoning`                               | **absent**   |
| NVIDIA `gpt-oss-*`          | `reasoning`, `reasoning_content`          | `content`    |

**Root cause:** no field is standardised; the OpenAI spec defines neither
`reasoning_content` nor `reasoning`.

**Consequence:** an SDK that reads `message["reasoning_content"]` sees nothing
on SenseNova and on b.ai's `mimo-v2.5`, and therefore cannot detect that
those models are reasoning — so it cannot detect J-006 either.

**Fix:** classification by **name substring** (`reason`/`think`), not by exact
key, in `response::is_reasoning_key`. All matching populated fields are
concatenated and the contributing field names are recorded.

### Finding J-008 — declared metadata is wrong

**Symptom:** SenseNova declares, for `6.7-flash-lite`,
`input_modalities: ["text","image"]` and
`supported_features: ["tools","json_mode","reasoning"]`. Measured reality:

| Declared          | Actual                                                                 |
|-------------------|------------------------------------------------------------------------|
| image input       | **fails** — HTTP 422 `Derivation source is unprocessable`               |
| `tools`           | **fails** on `6.7-flash-lite` (empty error body again)                  |
| `json_mode`       | works (`response_format: json_object` accepted)                         |
| `reasoning`       | correct                                                                 |

`json_schema` is **not** declared but is also **not** supported: it fails with
`guided_grammar … has compile_grammar_error: No module named 'xgrammar'`.

**Root cause:** the metadata is authored by the gateway operator and is not
validated against the serving runtime. It drifts.

**Consequence:** *the one provider that publishes capabilities publishes
incorrect capabilities.* Trusting declarations is strictly worse than
ignoring them, because it produces confident wrong answers.

**Fix:** three-layer evidence with provenance (`provenance::Fact`) and
`provenance::reconcile` — when a probe contradicts a declaration the **probe
wins** and the conflict is recorded on the model as an anomaly
(`declared image input at $.input_modalities but image probe failed`).

### Finding J-009 — the reasoning toggle: only one of eight spellings works, and six fail silently

**Symptom:** baseline response always carries `reasoning`. Of eight candidate
spellings, only `thinking: {"type":"disabled"}` actually removed it. The other
seven — including `enable_thinking: false`, `reasoning_effort: "low"`, and
`chat_template_kwargs.enable_thinking: false` — were **accepted with HTTP 200
and had no effect whatsoever.**

**Root cause:** unknown parameters are ignored by the gateway rather than
rejected. Acceptance is therefore **not** evidence of support.

**Consequence:** this is the single most dangerous signal in capability
detection. An SDK that sends `enable_thinking: false`, gets a 200, and reports
"thinking disabled" is wrong — and wrong in the direction that silently
doubles token cost and latency.

**Fix:** `probe::probe_thinking` determines support by **observing the
response**, never the status: send the toggle, then check whether the
reasoning text actually disappeared. Discovered result for SenseNova:
`thinking(on, off via thinking.type=disabled)`.

### Finding J-010 — output ceiling is discoverable from the rejection

**Symptom:** `max_tokens: 100000` → HTTP 400
`field MaxTokens invalid, should be in [1, 65536]`. The declared
`max_output_length` was also 65536.

**Root cause:** the validation error quotes the bound it enforces.

**Fix:** `errors::mine_limits` parses `should be in [1, N]` / `at most N` /
`maximum context length is N` and turns rejections into facts with
`Source::Inferred`. This is the *only* way to learn limits from gateways that
publish no metadata — which is 2 of our 3 providers.

### Finding J-011 — non-JSON error bodies

**Symptom:** `/v1/embeddings` and `/v1/rerank` on SenseNova return an **nginx
HTML page** (`<html>…404 Not Found…</html>`), not JSON.

**Root cause:** the endpoint is fronted by nginx and does not exist; the
error never reaches the application layer.

**Consequence:** `serde_json::from_str` fails and a naive client reports a
deserialisation error, losing the fact that this is simply a 404.

**Fix:** `errors::extract_message` falls back to text and strips HTML tags
(`strip_html`), so the message surfaces as `404 Not Found`.

---

## 22:10 — NVIDIA sweep (83 models, concurrency 6)

Result: **18 reachable, 59 HTTP error, 6 timeout.**

### Finding J-012 — NVIDIA echoes every optional field as `null`

**Symptom:** responses include the **full** optional field set —
`annotations`, `audio`, `function_call`, `reasoning`, `refusal`, `tool_calls`
— as explicit `null`s, on models supporting none of them. The key set even
varies between models on the *same* provider.

**Root cause:** the gateway serialises a fixed response struct rather than
omitting unset fields.

**Consequence (hidden bug):** **key presence does not imply capability.** Any
inference of the form `if "reasoning" in message` reports reasoning support
for every NVIDIA model. Worse, this is invisible — the code looks correct and
the data looks correct.

**Fix:** `response::classify_field` is **value-driven**: `null`, `[]`, `{}`
and empty/whitespace-only strings all classify as `FieldRole::Empty`
regardless of key name. Test: `allen_null_fields_are_not_capabilities`.

### Finding J-013 — five distinct error envelopes on one provider

NVIDIA alone returned:

| Envelope                              | Example                                                        |
|---------------------------------------|----------------------------------------------------------------|
| OpenAI `{"error":{"message":…}}`      | `{"error":{"message":"Model not found"}}`                      |
| RFC 7807 `{"status","title","detail"}`| `{"status":404,"title":"Not Found","detail":"Function '…': Not found for account"}` |
| Bare JSON `{"object":"error",…}`      | `{"object":"error","message":"Content cannot be a plain string…"}` |
| Plain text                            | `404 page not found`                                           |
| Internal 500                          | `Error during inference of request chat-…`                     |

Plus SenseNova's nginx HTML and b.ai's empty body → **seven shapes across
three providers.**

**Root cause:** no error envelope is standardised for OpenAI-compatible
gateways; RFC 7807 is a *different* standard that some adopt.

**Fix:** `errors::extract_message` tries each envelope in order and returns
which one matched (`envelope` field) so the parse is traceable. A test asserts
every declared envelope is actually reachable.

### Finding J-014 — a model that rejects text input

**Symptom:** `nvidia/nemotron-parse` → HTTP 400
`Content cannot be a plain string. The model does not support text input.`

**Root cause:** it is a document-parsing model expecting structured content
parts, not a chat model.

**Consequence:** "listed in `/v1/models`" does not mean "usable as chat".
Model *type* must be discovered, not assumed.

**Fix:** role is determined by **endpoint routing**
(`engine::route_discover`), never by name pattern-matching.

### Finding J-015 — embedding models 404 on the chat endpoint

**Symptom:** seven NVIDIA `*embed*` / `nvclip` / `arctic-embed` models return
plain-text `404 page not found` on `/v1/chat/completions`.

**Root cause:** they are served on a different endpoint family.

**Fix:** `route_discover` probes `/embeddings`, `/rerank`,
`/images/generations`, `/audio/speech`, `/videos` and assigns
`ModelRole` from whichever routes.

### Finding J-016 — latency spans two orders of magnitude

Measured on NVIDIA:

| Bucket | Models                                                                 |
|--------|------------------------------------------------------------------------|
| < 1 s  | `riva-translate-*-v2` 0.36 s, `nemotron-3.5-content-safety` 0.37 s, …   |
| 1–10 s | `nemotron-3-super-120b` 1.27 s, `gemma-4-31b-it` 7.66 s, `muse-glimmer-30b` 9.16 s |
| 30 s+  | `mistral-nemotron` 32.08 s, `gpt-oss-120b` 56.55 s                      |
| > 60 s | 6 models timed out entirely                                              |

**Consequence:** a single fixed timeout is wrong in both directions — it
aborts `gpt-oss-120b` (which does answer, in 57 s) and wastes 60 s on a dead
model. Discovery needs a per-provider timeout and must record latency so
callers can set their own.

---

## 22:30 — SenseNova full discovery run (SDK end-to-end)

```
OK   | sensenova-6.7-flash-lite [chat] in=T ctx=262144 out=65536
                                 thinking(on,off via thinking.type=disabled) tools=y json=y
FAIL | sensenova-u1-fast        [unknown]          not served by this gateway (model is not found)
OK   | sensenova-6.8-flash-lite [chat] in=T ctx=262144 out=65536
                                 thinking(on,off via thinking.type=disabled) tools=n json=y
FAIL | sensenova-u1.5-lite      [image-generation] not served by this gateway (model is not found)
```

Correct: `u1.5-lite` was classified **image-generation** by endpoint routing
despite 404-ing on chat — exactly the J-015 case.

### Finding J-017 — a `400` from an endpoint is evidence the model *is* there

**Symptom:** `sensenova-u1-fast` was reported `role=unknown` even though it is
an image model. `/v1/images/generations` rejected it with HTTP **400**
enumerating valid sizes, then returned **500** on a well-formed request.

**Root cause:** `route_discover` only accepted HTTP 2xx as a match, so a
model whose endpoint is reachable but whose backend is failing was
indistinguishable from a model that is not served at all.

**Why the distinction matters:** `404 model is not found` means *this endpoint
does not serve this model*; `400 invalid size` means *this endpoint **does**
route to this model and rejected my parameters*. The second is positive
evidence that the first is not.

**Fix:** `route_discover` now discriminates by failure mode:
`BadRequest`/`ContextTooLarge` → endpoint matches (recorded as
`images/generations (routed; minimal payload rejected)`); `ServerError` →
endpoint matches but backend unhealthy (recorded as an anomaly);
`ModelNotFound` → not served here.

### Finding J-018 — empty model id from an empty CLI argument

**Symptom:** a fifth "model" with a blank id appeared in the run and produced
`bad_request: required model`.

**Root cause:** `--extra ""` with `value_delimiter=','` yields `[""]`, not
`[]`; the empty string was treated as a model id.

**Fix:** skip blank entries when building the probe set.

---

## 23:03 — Workshop: the last edit never compiled

The very first command of this session (`cargo check -p ai-discovery`) failed:
`unexpected closing delimiter: }` at `probe.rs:346`, plus two `RawResponse`
initializers missing the new `retry_after` field and the example missing
`transport_policy`.

### Finding J-019 — the tree was never built after the last refactor

**Symptom:** three compile errors, one of them a **stray closing brace** at
module level in `probe.rs`.

**Root cause:** the `retry_after` field and the `transport_policy` field were
added to their structs but the test initializers and the example were not
updated, and a duplicate `}` survived an edit. The preceding session ended
with edits on disk rather than a green tree.

**Fix:** delete the stray `}`, add `retry_after: None` to the two test
initializers, add `transport_policy` to the example's `DiscoveryConfig` literal
(and correct `cfg.timeout`, which read a field the struct does not have).

**Consequence for the workflow:** a vetted vocabulary ("green tree is the
baseline") must be re-asserted before any live run; the b.ai data below was
measured with a *rebuilt* binary, not the broken tree.

---

## 23:07 — SenseNova re-verification (full run with final engine)

```
OK   | sensenova-6.7-flash-lite [chat] in=T ctx=262144 out=65536
                                 thinking(on,off via thinking.type=disabled) tools=y json=y
FAIL | sensenova-u1-fast        [unknown]  not served by this gateway (model is not found)
OK   | sensenova-6.8-flash-lite [chat] …thinking(on,off via thinking.type=disabled) tools=n json=y
FAIL | sensenova-u1.5-lite      [unknown]  not served by this gateway (model is not found)
```

Reproduces everything recorded earlier on the rebuilt engine, including the
sibling-model capability split (J-027): `6.7-flash-lite` `tools=y` while
`6.8-flash-lite` `tools=n` for the same declared feature list.

---

## 23:05–23:11 — Wire-level edge-case battery (mock server)

Live gateways cannot produce controlled anomalies on demand (a model cannot
return `choices: []` for one request and a perfect completion for the next).
A local mock HTTP server (`tests/edge_harness.rs`, 26 tests) serves crafted
responses through the **real** transport and engine. It surfaced four bugs
that no live provider would have exposed cleanly.

### Finding J-020 — client timeouts are classified as `Network`, not `Timeout`

**Symptom:** a request cancelled by the client timeout reported
`network: error sending request for url (…)` — the most important branch for
discovery (a 60 s cutoff is *how* a slow model is classified) was never
exercised.

**Root cause:** `reqwest::Error`'s `Display` renders only the outer message;
the real cause (`operation timed out`) lives in the **source chain**.
`RawResponse::error` classified on `format!("{e}")`, which never contains
the words "timed out".

**Fix:** `describe_transport_error` walks the full source chain into the
message, so classification sees the cause. Verified by the mock: a server
that sleeps past the timeout now yields `timeout`, and the regression test
`timeout_is_classified_as_timeout` pins it.

### Finding J-021 — a 404 with an empty body was `Other`, not `ModelNotFound`

**Symptom:** `route_discover` and `blocker` treat "404 with no body" as an
unclassified failure, so the message says `other: HTTP 404 with empty body`
instead of `model_not_found`.

**Root cause:** `classify` derived the class from the *message text*, and an
empty body contains no "not found" tokens. But for 404 the status itself is
the signal; the body is redundant.

**Fix:** `envelope == "empty"` with status 404 → `ModelNotFound`.
`chat_404_empty_body_is_model_not_found` pins it.

### Finding J-022 — `probe_streaming` false-positives on `"data:"` inside content

**Symptom:** a non-streamed 200 whose *text* contains the substring `data:`
(e.g. the model answering about data points) counted as SSE.

**Root cause:** `body.contains("data:")` scans the whole body, not the
line-start frames that SSE actually uses.

**Fix:** require a `data:` prefix at the start of a line. The mock test
`streaming_silently_ignored_is_recorded` sends exactly the adversarial
content and still reports `streaming=false`.

### Finding J-023 — HTTP 200 with no `choices` was silently "reachable"

**Symptom:** `choices` missing or `[]` → `probe_reachable` returned
`reachable: true, message: None`, and `discover_one` recorded **no anomaly**:
an SDK consumer sees a healthy model with no answer at all.

**Root cause:** reachability was defined as "HTTP 2xx". The payload shape is
a second, independent failure mode ("gateway accepted the request but its
response is unusable").

**Fix:** an anomaly `HTTP 200 but no usable message: choices array missing or
empty` is recorded when a 2xx carries no message. Tested both shapes.

---

## 23:12 — `discover()` swallowed a failed `/v1/models` listing

### Finding J-024 — a wrong API key looked identical to an empty catalog

**Symptom:** `DiscoveryEngine::discover` did
`list_models().await.unwrap_or_default()`. A 401/500/HTML response became
`[]` → "0 models found", which is indistinguishable, at a glance, from a
gateway with no models.

**Root cause:** the listing error was discarded at the only boundary that
could report it.

**Fix:** `discover()` now returns `Result<Vec<_>, DiscoveryError>` and
propagates `ListFailed`; the example exits non-zero with FATAL. Mock test
`discover_propagates_list_failure` (500 → error containing `500`).

---

## 23:14 — b.ai full sweep with the rebuilt engine

```
total 46 (44 listed + the 2 task-specified capitalized ids)
reachable 6 → deepseek-v4-flash, deepseek-v4-flash-vision-exp, hy3,
             mimo-v2.5, glm-5.3-flash, qwen3.8-flash
38 × billing (account not funded: either "Deposit required to unlock premium
   models" 403, or "credit insufficient balance: balance=0 required=N" 400)
 2 × not served → DeepSeek-V4-Flash / DeepSeek-V4-Flash-Vision-Exp (J-004
   reproduced verbatim: exact-cased ids do not exist; lowercase do)
```

Mining during the sweep (each reachable model gets the max-output probe):
`qwen3.8-flash out=131072`, `glm-5.3-flash out=131072`,
`deepseek-v4-flash-vision-exp out=393216` — three different output ceilings,
all recovered from rejection text, none of it declared anywhere.

### Finding J-025 — the runtime SDK defaulted capabilities to `true` for every model under test

**Symptom:** `openai_compat::model_info_from_entry` (the no-catalog branch)
set `supports_streaming/tools/structured_output: true` unconditionally. The
hardcoded `default_catalog` in `ai-models` contains **zero** entries for b.ai,
NVIDIA or SenseNova, so *every one of the 133 models in this task* took that
branch — the runtime SDK reported tools/streaming/structured output as
supported for all of them, with no evidence.

**Root cause:** the branch duplicated a different set of guesses from
`model_info_for_id` (which used `false`/`false`/`false`), so the two paths
disagreed, and both violated the module's own comment ("never blanket true").

**Fix:** no-catalog branch now defaults all three to `false` (the
`model_info_for_id` values), leaving capability evidence to probing.
All 72 ai-providers tests still pass — nothing had asserted the old defaults.

---

## 23:19–23:25 — b.ai deep battery (the six task models, full capability battery)

| Model | in | ctx | out | thinking | tools | json |
|-------|----|-----|-----|----------|-------|------|
| deepseek-v4-flash | T | ? | ? | (pending) | | |
| deepseek-v4-flash-vision-exp | T | ? | 393216 (mined) | | | |
| hy3 | **TI** | ? | ? | on, off via `thinking.type=disabled` | y | y |
| mimo-v2.5 | T | ? | ? | (pending) | | |
| glm-5.3-flash | T | ? | 131072 (mined) | on, off via `reasoning_effort=low` | y | y |
| qwen3.8-flash | T | ? | 131072 (mined) | on, off via `enable_thinking=false` | y | y |

### Finding J-026 — the working thinking-toggle spelling is **model-local**, not provider-local

**Symptom:** three models on *one* gateway each honour a different spelling:
`hy3` → `thinking.type=disabled`; `glm-5.3-flash` → `reasoning_effort=low`;
`qwen3.8-flash` → `enable_thinking=false`.

**Root cause:** models route to different backends (distributors), and each
backend implements its own parameter vocabulary. The provider's gateway
accepts (and silently ignores) all spellings, so *acceptance is never
evidence of support* — only observing the disappearance of reasoning is.

**Consequence:** any cached "the b.ai thinking toggle is X" is wrong for at
least two of three models. The toggle battery must run **per model**, and the
discovered spelling must be stored per model (it is: `ThinkingSupport`).

### Finding J-027 — tool support differs between sibling models with identical declared features

**Symptom:** SenseNova `6.7-flash-lite` and `6.8-flash-lite` both declare
`supported_features: ["tools", …]`, but the tool probe returns a call for
`6.7` and **no call for `6.8`** (`tools=n`).

**Root cause:** `tools=y` requires the model to actually *emit* a tool call
for the prompt; declaration only claims acceptance. One of the two silently
ignores the `tools` parameter (the constant J-009 danger, now seen on a
different capability).

**Consequence:** a "feature" reported by the *catalog* is not a feature;
per-model probing is the only sound source.

### Finding J-028 — capability probes are stochastic; a single sample is not evidence

**Symptom:** the very next run of the same SenseNova battery reported
`6.8-flash-lite tools=y`, flipping the J-027 verdict. Identical requests,
different outcome.

**Root cause:** tool-call *emission* is stochastic. Asking "use the tool"
produces a call only sometimes, even on a model that genuinely supports
tools. One sample asserts a capability on the result of a coin flip.

**This is the same bug as J-009 in a different costume:** J-009 was "the
gateway accepts a spelling it ignores"; J-028 is "the model accepts the
parameter but its observable behaviour is a random variable". In both cases
the **status is not the signal**; the signal is the response content, and it
must be sampled.

**Fix:** `probe_tools` now takes [`TOOL_SAMPLES`] = 3 samples at
`temperature: 0` and votes by majority, exposing `positive/samples` in the
evidence (`tool_calls returned in 2/3 samples`) and setting confidence by
agreement (unanimous 0.9, mixed 0.6). Mock tests pin both directions:
`tools_probe_uses_majority_vote` (2/3 → true), `tools_probe_minority_is_not_supported`
(1/3 → false).

**Residual risk:** the same stochasticity applies to the vision, structured
and thinking probes, all single-shot. Text answer presence is *far* less
stochastic than tool-call emission, and the thinking probe checks for
reasoning *absence* (deterministic when a toggle works), so the tool probe
was the worst offender; the others remain single-shot deliberately.

### Finding J-029 — a model can be intermittently unservable for an account-level reason

**Symptom:** `mimo-v2.5` answered OK at 23:14 (Stage A) and failed at 23:26
and 23:28 (Stage B + dedicated retest) with `bad_request: Error from provider
(Console Go): Upstream request failed: [404] No allowed providers are
available for the selected model. Providers serving xiaomi/mimo-v2.5-20260422:
gmicloud, deepinfra, xiaomi, parasail, streamlake, novita, but your request's
***.only preference permits only: tencent.`

**Root cause:** the gateway proxies to upstream providers selected by an
account-level `***.only` preference. This account's preference permits only
`tencent`, which is **not** in the six-provider pool serving this model. The
filter changes over time, which is why the same request succeeded earlier.

**Consequence (two):**
1. A single-shot failure is not a verdict. `BadRequest` is non-retryable in
   our classifier, yet the identical request succeeded 15 minutes earlier —
   the classification was correct, but *one observation of a flaky model* is
   not evidence of its health. Discovery must record the observation time and
   re-test on demand (see architectural improvements: flakiness sampling).
2. The nested reason is inside the message, not the status. If the SDK
   truncates the message for presentation, the *true* cause ("***.only
   preference permits only tencent") is lost — the model id, the upstream
   pool and the permitted pool must all remain visible.

---

### Finding J-030 — NVIDIA function states change between runs: `DEGRADED` → 404

**Symptom:** `nvidia/nemotron-3.5-lightning-30b-a3b` returned at 23:20
`bad_request: Function id '…': DEGRADED function cannot be invoked`, then at
23:53 the same model returned **HTTP 404 with an empty body** (classified
`ModelNotFound`). Two different transient states of the same shipped id, 33
minutes apart.

**Root cause:** NVIDIA serves models as per-account "functions" that cycle
through deployment states (active → degraded → not found). None of these is a
model-defect signal; the id exists and is merely not invokable *right now*.

**Consequence:** an SDK that treats a single 400/404 as the model's health
verdict will cycle models between `bad_request` and `model_not_found` across
its own tests. The same reasoning as J-029 applies: verdicts need
retest-on-demand and the classifier needs a `TemporarilyUnavailable` class
for `DEGRADED`-style vocabulary instead of `BadRequest`.

### Finding J-031 — a 500 on the image probe is not evidence of "no vision"

**Symptom:** `meta/llama-3.2-11b-vision-instruct` — a model whose *name*
declares vision — probed `in=T` (no image modality). The evidence was
`image_url part rejected: Internal Server Error`, i.e. the image request got
a **500**. Same for `muse-glimmer-30b`.

**Root cause:** the vision probe treated *any non-2xx* as a negative. But a
500 means the request reached the model's serving path and the backend
failed — *routed, but unhealthy* — which is the exact J-017 situation, where a
5xx was already treated as inconclusive for endpoint routing but not for
vision.

**Why this is worse than a plain miss:** a confident `vision: false` on a
vision model is a *wrong capability*. The SDK will refuse to send images to a
model that accepts them.

**Fix:** `discover_one` now distinguishes probe outcomes by error class:
`ServerError`/`Timeout`/`Network` → `vision probe inconclusive (server_error):
…` anomaly, confidence 0.3 (unknown-flavoured, not negative); only genuine
rejections (400/404/422) count as negative. Mock test
`vision_probe_5xx_is_inconclusive_not_negative` pins it.

---

## 23:52 — NVIDIA deep battery (12-model subset, fixed binary)

```
OK  gpt-oss-20b        in=TI thinking(on,no-off-switch) tools=y json=y
OK  gpt-oss-120b       in=TI thinking(on,no-off-switch) tools=y json=y   (24 s)
OK  nemotron-3-nano-omni-30b-a3b-reasoning  thinking(on,off via reasoning_effort=none)
OK  muse-glimmer-30b   thinking(on,no-off-switch) tools=n (0/3) json=y
OK  gemma-4-31b-it     thinking(off) tools=n (0/3) json=y
OK  diffusiongemma-26b-a4b-it thinking(off) tools=y json=n
OK  llama-3.2-11b-vision-instruct  in=T (probe 500 → J-031) tools=y(3/3)
OK  riva-translate-4b-instruct-v2  thinking(off) tools=n json=y   (0.27 s)
OK  nemotron-3.5-content-safety    thinking(off) tools=n json=y   (0.35 s)
OK  mistral-nemotron   thinking(off) tools=y json=y               (15.8 s)
FAIL nemotron-3.5-lightning-30b-a3b  == 404-empty (was DEGRADED)  (J-030)
FAIL nemotron-parse                 == text-input rejection        (J-014)
```

Observations:
* Reasoning models on NVIDIA (`gpt-oss-20b/120b`, `muse-glimmer`) suppress
  reasoning under **none** of the eight tested spellings — `no-off-switch`.
  These are the most expensive models to use; an SDK must not burn tokens on
  a toggle that silently no-ops.
* `gemma-4-31b-it` and `muse-glimmer-30b` are `tools=n` with **0/3 samples** —
  the majority vote is doing its job (single-sample results would have been
  flips around 1/3).
* `gpt-oss-*` probe as **image-capable** (2xx on the image payload); NVIDIA
  serves them multimodal. Residual risk on the 2xx-as-acceptance caveat
  (REPORT §6).

---

## Live results

*(appended as runs complete)*

### Run: b.ai — full sweep + deep battery (2026-08-28 23:00–23:27 IST)

46 ids (44 listed + 2 task-specified capitalized ids):
`6 reachable, 38 billing-blocked, 2 model-not-found (the capitalized ids)`.

Deep battery (the six task models; `ctx` all `0` — b.ai publishes no context
metadata and binary-search probing was not enabled for this run):

| Model | in | ctx | out | thinking | tools | json |
|-------|----|-----|-----|----------|-------|------|
| deepseek-v4-flash | T | 0 | 0 (gateway accepted 100 M silently) | on,off via `enable_thinking=false` | y | y |
| deepseek-v4-flash-vision-exp | T | 0 | 393216 (mined) | on,off via `thinking.type=disabled` | y | y (no schema) |
| glm-5.3-flash | T | 0 | 131072 (mined) | on,off via `reasoning_effort=low` | y | y |
| hy3 | **T+I** | 0 | 0 (accepted 100 M silently) | on,off via `thinking.type=disabled` | y | y |
| mimo-v2.5 | T | 0 | — | — | — | — |
| qwen3.8-flash | T | 0 | 131072 (mined) | on,off via `enable_thinking=false` | y | y |

Notes:
* `hy3` is the only vision model of the six; declared metadata said nothing.
* `deepseek-v4-flash-vision-exp` accepts `json_object` but **rejects**
  `json_schema` — anomaly recorded (`json_object accepted but json_schema
  rejected`).
* `mimo-v2.5` was reachable at 23:14 and failed at 23:26 and 23:28 — see
  J-029. Its tools/structured columns are therefore unmeasured, not negative.

### Run: SenseNova — deep battery (2026-08-28 23:21–23:26 IST)

| Model | role | ctx | out | thinking | tools | json |
|-------|------|-----|-----|----------|-------|------|
| sensenova-6.7-flash-lite | chat | 262144 (declared) | 65536 (declared) | on,off via `thinking.type=disabled` | y | y obj / **no schema** |
| sensenova-6.8-flash-lite | chat | 262144 (declared) | 65536 (declared) | on,off via `thinking.type=disabled` | **y** | y obj / **no schema** |
| sensenova-u1-fast | unknown | — | — | — | — | — |
| sensenova-u1.5-lite | image-generation | — | — | — | — | — |

Notes:
* Both chat models carry the anomaly pair `json_object accepted but
  json_schema rejected` and `declared image input at $.input_modalities but
  image probe failed — declaration is wrong` (J-008 reproduced on both).
* `6.8-flash-lite` returned `tools=y` on this run and `tools=n` on the 22:30
  run — the J-028 coin-flip that motivated majority sampling.
* `u1-fast` is *routed* to `/images/generations` (400/500 evidence) but the
  backend returned 500 → recorded as an anomaly, role stays `unknown`
  (J-017's "routed but unhealthy" case, now with its message visible).
* `u1.5-lite` answers `/images/generations` successfully (`2xx`), hence the
  `image-generation` role on the 404-chat line.

### Run: NVIDIA — full sweep + role pass + deep battery (23:00–23:58 IST)

**Full sweep (83 listed):** `19 reachable (0.25–22.4 s), 55 not served for this
account, 2 server-error, 2 bad-request, 5 transport-level` — but see J-020's
proof below: the 5 "transport-level" were **client timeouts misread by the
old binary**; the rebuilt binary classifies them `no response within 90 s`.

Reachable set changed since the 22:10 sweep: `gemma-4-31b-it`,
`poolside/laguna-xs-2.1`, `nemotron-3-ultra-550b`, `ising-calibration`,
`nemotron-3-nano-omni-30b-a3b-reasoning` are newly OK; several 22:10-OK
models are now not-served. Availability is account- and time-dependent.

**Notable unreachable listings:**
* 55 × `Not Found: Function '<uuid>': Not found for account …` (RFC 7807)
* `nvidia/nemotron-3.5-lightning-30b-a3b` — `DEGRADED function cannot be
  invoked` at 23:20 → 404-empty at 23:53 (J-030)
* `nvidia/nemotron-parse` — `Content cannot be a plain string. The model does
  not support text input.` (J-014, reproduced)
* `nvidia/ai-synthetic-video-detector`, `llama-3.1-nemoguard-8b-topic-control`
  — 500s (retryable, still failing after 4 attempts)
* 5 × timeout (formerly "network", J-020): `deepseek-v4-flash-0731`,
  `deepseek-v4-pro-0813`, `llama-3.2-90b-vision-instruct`, `llama-guard-4-12b`,
  `kimi-k3` — each slower than 90 s per attempt.

**Deep battery:** see the 23:52 entry above (10/12 reachable).

**Role pass (23:35–23:59, all 83 via endpoint routing):** roles are
`20 Chat, 2 Embedding, 61 Unknown` — the unknown bucket is almost entirely
`not served for this account`, plus:

* `nvidia/nemotron-3-embed-1b` → **Embedding** (accepted on `/embeddings`)
* `nvidia/llama-nemotron-embed-vl-1b-v2` → **Embedding** (routed; minimal
  payload rejected — the J-017 BadRequest-as-evidence rule applied to a live
  embedding model)
* J-020 fully closed: of the five "network" from the 23:14 sweep, the
  rebuilt binary + retest shows **4 × timeout** (`deepseek-v4-flash-0731`,
  `llama-3.2-90b-vision-instruct`, `llama-guard-4-12b`, `kimi-k3`) and
  **1 × reachable-on-retry** (`deepseek-v4-pro-0813` answered in the role
  pass). Zero genuine network errors — the old classification was wrong
  about all five.

---

## Continuation — 2026-08-29 04:44 (follow-up session)

*Context: between sessions an independent audit (AUDIT-REPORT.md) and live
campaign (LIVE-CAMPAIGN-JOURNAL.md) ran against the same tree; its fixes
(`NotEntitled`/`Gone` classes, message truncation, https-validation,
shape-aware reachability, `Debug` redaction, limit-mining guards) are present
and are **not** re-litigated here. This session implemented the outstanding
improvements from REPORT §5 and found four more defects along the way.*

### Finding J-032 — synonym rank lost to JSON traversal order

**Symptom:** `declared::first_u64(entry, Concept::MaxOutputTokens)` on
`{"max_tokens": 100, "max_output_tokens": 8192}` returned **100** whenever the
gateway happened to serialise `max_tokens` first.

**Root cause:** `first_u64`/`first_bool`/`first_str` took the *first hit in
traversal order*. The synonym list is ordered deliberately ("most specific
spelling wins"), but that order was never applied to the hits — so which of
two spellings of the same concept won depended on the gateway's key order.

**Consequence:** `context_window` vs `context_window`-in-`capabilities`,
`max_output_tokens` vs `max_tokens` — the value an SDK loads from a
declaration is data-dependent. This is exactly the "confident wrong value"
class (J-008, J-025): no error, just the wrong limit.

**Fix:** `ranked_hits` sorts by `(synonym_rank, path depth)`; rank beats
traversal order, ties break shallow. Test `synonym_priority_beats_traversal_order`
pins both the same-level and deeper-but-higher-rank cases.

### Finding J-033 — temporary infrastructure states were `BadRequest`

**Symptom:** `DEGRADED function cannot be invoked` (NVIDIA) and
`Upstream request failed: [404] No allowed providers … ***.only preference
permits only: tencent` (b.ai, J-029) both classified as `bad_request` —
non-retryable, and neither the model's nor the caller's fault.

**Root cause:** the classifier had vocabulary for *permanent* conditions
(billing, entitlement, gone) but none for *temporary* ones, so the two
temporary states fell through to the status default.

**Fix:** new `ErrorClass::TemporarilyUnavailable` (retryable, distinct from
`RateLimited` since no Retry-After is promised) driven by vocabulary
(`degraded function`, `upstream request failed`, `no allowed providers`,
`temporarily unavailable`, `retry later`, …) checked before the status
branches; blocker text `temporarily unavailable (…)`. Tests pin the observed
phrases and confirm the NVIDIA entitlement shape is **not** captured.

### Finding J-034 — a timeout retried ×4 costs ~6.5 min per dead model

**Symptom:** the NVIDIA sweep's wall clock was dominated by models slower
than the timeout: 90 s × 4 attempts + backoff ≈ 6.5 min each, and they were
in the majority.

**Root cause:** `ErrorClass::Timeout` is retryable, and the transport's only
retry cap was `max_attempts` — which applies to the classes where retrying
often helps (429, 5xx). A timeout is evidence of a model slower than the
window; a second identical window rarely helps, and queue artifacts (J-005)
are satisfied by a single retry.

**Fix:** `TransportPolicy::max_timeout_attempts` (default **2**: one retry is
enough to absorb a queue spike; `none()` = 1) enforced in `should_retry`.
Mock test `timeout_is_not_repeated_in_sweep_mode` asserts 1 hit at
`max_timeout_attempts = 1` and exactly 2 at the default.

### Finding J-035 — "does not support text input" was a bare `bad_request`

**Symptom:** `nvidia/nemotron-parse` (J-014) reported its actual defect in the
message — the model takes structured content parts, not plain text — but the
SDK surfaced only `bad_request: Content cannot be a plain string…`, losing
the reason. Role stayed `unknown` with no pointer at why.

**Root cause:** text-input rejection vocabulary was not classified at all.

**Fix:** in the unreachable branch, messages matching
`does not support text input` / `cannot be a plain string` produce a
traceable anomaly (`chat endpoint rejected plain-text input: … — the model
expects structured/content-part input; role unknown, text modality not
asserted`). Mock test `non_text_rejection_is_a_traceable_anomaly` pins it.

### Implemented improvements (from REPORT §5, now landed)

| Improvement | Landing |
|---|---|
| Thinking toggle cached per model | `ModelInfo::thinking_control` + `with_thinking_control`; `to_model_info` fills it from `ThinkingSupport` (J-026's per-model value rides with the model) |
| Runtime list_models uses the generic scanner | `ai-providers::model_info_from_entry` now reads limits/vision via `ai_discovery::declared` (one synonym table, nested paths, rank precedence) instead of hand-listed spellings; `extract_u64`/`extract_vision` deleted |
| Gateway state vocabulary | `TemporarilyUnavailable` (J-033) |

**State after continuation:** `cargo check --workspace --all-targets` clean,
`cargo clippy --workspace --all-targets` zero warnings, 147 tests green
(ai-discovery 44 unit + 31 wire-level, ai-providers 72).
