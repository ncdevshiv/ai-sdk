# AI SDK — Scalable Architecture & Long-Term Development Roadmap

**Prepared by:** Software Architect (expert engagement)
**Date:** 2026-08-29
**Scope:** `ncdevshiv/ai-sdk` Rust workspace (30 crates)
**Status:** Design proposal for team review — not yet ratified

---

## 0. TL;DR

- The codebase is **architecturally strong**: a clean, layered crate design, real implementations with no mocks (per `ENGINEERING-SPEC.md §33` and confirmed by `AUDIT-REPORT.md`), and a genuinely clever `ai-discovery` provenance model. The risk now is not *"does it work"* but *"will it survive the team that builds it"* — evolution, API stability, and security hygiene.
- **Stop-the-line action (do this first):** `AUDIT-REPORT.md` finding **C-1** reports live plaintext API keys sitting in the working tree (`tools/discovery-probe/cfg_*.json`, `out/*.log`), untracked and **not matched by `.gitignore`**. One `git add -A` publishes them permanently. Rotate → scrub → ignore → add a pre-commit secret scan.
- **Biggest architecture gap:** the ADR set (001–010) was written on 2026-02-17 in a **TypeScript/npm** framing and is still marked *"Proposed."* It does not describe the Rust workspace. Re-baseline and ratify before planning further.
- **"Scalable system architecture" is two different problems.** We must design them separately or we will optimize the wrong one:
  - **(A) The SDK as a product** scales by *compile-time modularity, a stable API contract, feature-gating, and a bounded crate count*.
  - **(B) The systems built on the SDK** scale by *runtime topology*: orchestrator + worker pools + message bus + stores + observability.
- Proposal: a **4-phase, ~18-month roadmap** → Secure & Reconcile → Stabilize the Contract → Close the Deferred Surface → Scale the Runtime & Ecosystem.

---

## 1. Framing — two scales, one codebase

| | (A) SDK as a product | (B) System built on the SDK |
|---|---|---|
| What "scales" | Number of users / features without breaking consumers | Number of agents, requests, contexts at runtime |
| Primary lever | API stability, crate boundaries, feature flags | Deployment topology, concurrency, durability |
| Failure mode | Breaking changes, crate sprawl, churn | Latency, cost, partial failure, state loss |
| Owned by | The SDK team (us) | The SDK team (reference) + adopters |
| Current state | Implicit (no stability policy) | Implicit (no reference topology) |

Most "make it scalable" requests silently mean (B). But for a 30-crate library, **(A) is the higher-leverage risk** — every adopter you break compounds. The design below treats (A) as the foundation and (B) as a separately-shipped reference.

---

## 2. Current-state assessment

### Strengths
- **Layered crate separation** matching the PRD's architectural layers (`ai-types` → `ai-core` → feature crates → `ai-sdk` facade). Dependency direction is documented and mostly enforced.
- **Real implementations, no mocks** — verified by the audit's own discipline.
- **`ai-discovery` provenance model** (declared · inferred · probed, "probe beats declaration") is a real differentiator and better than what it replaced.
- **Audit culture exists** — `AUDIT-REPORT.md` is honest about limitations instead of disguising them (the spec's §36 "never disguise a limitation as success" is being followed).
- **Protocols are modern** — ADR-013 adopted the 2026-07-28 MCP revision (no legacy handshake), and A2A client/server exist.

### Gaps (ranked)
1. **Security hygiene (from today's audit).** C-1 live keys in tree; C-2 `Transport` leaks API key via derived `Debug`; H-1 no `base_url` scheme validation / SSRF guard; H-2 context-window probe fabricates evidence and reports `Fact::probed` with wrong confidence. These are correctness *and* trust failures.
2. **Stale, unratified ADRs.** 001–010 describe a TS/npm system; all "Proposed." They are decision records that no longer describe the system.
3. **No API-stability policy.** 30 public crates, real SemVer surface, `ReasoningEffort::Max` added without `#[non_exhaustive]` (audit M-8). There is no contract promising adopters what is safe to depend on.
4. **Fuzzy consumption boundary.** `ai-sdk` facade + `ai-core` + `ai-types` + 28 feature crates, but no explicit "internal / `-sys`" seam. Adopters can reach into anything.
5. **No crate-count ceiling.** At 30 crates with deferred features queued (LoRA, computer use, IDE/LSP, self-skill, GraphRAG, voice realtime), sprawl is the likely trajectory.
6. **Deployment story mismatch.** ADR-009 is npm/Edge-oriented; the real deployment surface is `ai-sidecar` (stdio JSON-RPC gateway), `ai-cli`, and `ai-edge` (WASM). No reference topology for a distributed agent system.

---

## 3. Architectural principles (carry forward + add)

From `ENGINEERING-SPEC.md §38` priority order (Correctness → Real functionality → Security → Reliability → Performance → Maintainability → DX). Added:

- **Stable core, unstable edges.** The traits/types layer changes slowly; feature crates may churn behind flags.
- **Dependency direction is law, not guidance.** Enforce with an architecture test, not just a README note.
- **Feature-flag, don't fork.** Opt-in capability (`#[cfg(feature=…)]`) beats a second crate or a v2 rewrite.
- **Observability is cross-cutting**, not a crate you opt into. Emit OTel-native spans from the core.
- **Prefer reversible decisions.** Thin provider adapters, trait-based storage, and pluggable transports keep us able to change our minds cheaply.

---

## 4. Target architecture (C4 views)

### 4.1 System Context
The SDK is a **Rust workspace** consumed three ways: (1) linked directly into Rust apps, (2) driven over `ai-sidecar` stdio JSON-RPC from non-Rust hosts, (3) compiled to WASM via `ai-edge`. It integrates outward with LLM providers, MCP/A2A peers, vector/KV/object stores, and web/search backends.

### 4.2 Container — layered crate map
Group the 30 crates into five layers plus the facade. This is the "container" view; the dependency rule is *strictly inward*.

| Layer | Crates | Role |
|---|---|---|
| **Facade** | `ai-sdk` | The only broad public API most adopters import |
| **Orchestration** | `ai-agents`, `ai-orchestra`, `ai-workflows`, `ai-protocols` (MCP/A2A) | Multi-agent, swarms, self-healing, workflows, inter-agent protocols |
| **Capability** | `ai-tools`, `ai-memory`, `ai-rag`, `ai-voice`, `ai-web`, `ai-computer`, `ai-discovery` | Skills, memory, retrieval, voice, web research, computer use, capability discovery |
| **Foundation** | `ai-providers`, `ai-models`, `ai-stream`, `ai-runtime` | Provider adapters, model registry, streaming, parallel execution & resilience |
| **Infrastructure** | `ai-security`, `ai-cache`, `ai-storage`, `ai-observability`, `ai-analytics`, `ai-devtools`, `ai-config`, `ai-errors` | Cross-cutting: security, caching, storage, telemetry, cost, debugging, config |
| **Core** | `ai-types`, `ai-core` | Domain types + traits (the stable contract) |
| **Edge** | `ai-edge`, `ai-cli`, `ai-sidecar` | WASM target, CLI, stdio gateway |

### 4.3 Component — consumption boundary & stability tiers
The single most important SDK-internal decision. Define four tiers and enforce them:

- **Tier 1 — Stable (SemVer-major frozen):** `ai-types`, `ai-core` traits, `ai-sdk` facade surface. Breaking changes require a major version.
- **Tier 2 — Evolving (SemVer-minor):** feature crates (`ai-agents`, `ai-rag`, …). New features land in minors; breaking changes only in majors.
- **Tier 3 — Unstable (`#[doc(hidden)]`, `*-internal`/`-sys`):** serialization details, probe internals, transport plumbing. Not a public contract; may change in patch.
- **Tier 4 — Experimental (feature-gated):** computer use, IDE/LSP, self-skill, GraphRAG, voice realtime. Behind `feature = "unstable-*"`.

Adopters depend on Tiers 1–2 only. The `ai-sdk` facade is the *only* recommended entry point for Tier-2 composition.

### 4.4 Deployment — reference runtime topology for a large-scale agent system
The SDK stays **embeddable**; we additionally ship a **reference topology** for systems that need "1000s of agents" (PRD §3.10):

```
            ┌─────────────────── Control Plane ───────────────────┐
            │  Orchestrator (ai-orchestra) · Planner · Watchdogs   │
            │  Policy/Security sidecar (ai-security) · HITL gate   │
            └───────────────┬───────────────────────┬──────────────┘
                            │ tasks/results          │ traces/metrics
                   ┌────────▼────────┐       ┌───────▼─────────────┐
                   │  Message Bus     │       │  Observability       │
                   │ (NATS/Kafka, or  │       │  OTel → Collector →  │
                   │  in-proc channel │       │  Tempo/Prom/Grafana  │
                   │  for small depls)│       └──────────────────────┘
                   └────────┬────────┘
            ┌──────────────┼───────────────┐  (worker pool, autoscaled)
        ┌───▼────┐    ┌────▼────┐    ┌─────▼───┐
        │Worker A│    │Worker B │    │ Worker N │   each runs agent executors
        │(agents)│    │(agents) │    │ (agents) │   (ai-agents + ai-runtime)
        └───┬────┘    └────┬────┘    └─────┬───┘
            │             │               │
     ┌──────▼──────────────▼───────────────▼──────┐
     │  Stores: KV (state) · Vector (RAG) · Object  │
     │  (sqlite now; postgres/redis/qdrant adapters) │
     └───────────────────────────────────────────────┘
            ▲
     ┌───────┴────────┐      ┌──────────────────────┐
     │  Gateway        │      │  External peers        │
     │ (ai-sidecar /   │      │  MCP servers · A2A      │
     │  ai-cli)        │      │  agents · LLM providers │
     └─────────────────┘      └──────────────────────┘
```

For small deployments the bus and stores collapse to in-process channels + sqlite; the *same* `ai-*` crates power both. That is the key property: **topology is a deployment choice, not a code fork.**

---

## 5. Key decisions & trade-offs

| # | Decision | Trade-off (what we give up) |
|---|---|---|
| D1 | Re-baseline ADRs 001–010 to Rust and ratify (status → Accepted) | Upfront doc effort; some "decisions" become "superseded" |
| D2 | API stability tiers + SemVer + feature flags (ADR-014) | Slower velocity on Tier-1; more discipline |
| D3 | Crate-count ceiling (~36) with vertical-slice grouping | Some features share a crate instead of getting their own |
| D4 | Explicit consumption boundary (facade + `*-internal` seam, ADR-015) | Less "reach into internals" flexibility for power users |
| D5 | Ship a reference runtime topology *separate* from the SDK | SDK is no longer "just a library"; more to maintain |
| D6 | OTel-native observability from the core | Adds a dependency weight to the foundation layer |
| D7 | Security hardening pass now (rotate keys, redact `Debug`, SSRF guard, gitleaks) | Blocks feature work for ~1–2 weeks |

---

## 6. Long-term roadmap (phased, ~18 months)

### Phase 0 — Secure & Reconcile (Weeks 0–4)
- **Goals:** eliminate the audit's critical/high findings; make the ADRs truthful.
- **Work:** rotate the two exposed keys; delete `tools/discovery-probe/cfg_*.json` + `out/*.log`; extend `.gitignore`; add `gitleaks`/secret-scan pre-commit. Fix C-2 (hand-write `Debug` for `Transport`, redact key), H-1 (`base_url` scheme validation + pinned redirects + SSRF guard), H-2 (stop overwriting probe evidence; return `None` on abort). Re-baseline ADRs 001–013 to Rust; add an **architecture test** enforcing inward dependency direction. Adopt `CODEOWNERS` / crate owners.
- **Exit:** `git` tree contains zero secrets; `cargo clippy -D warnings` clean; ADRs reflect Rust; dependency-direction test in CI.
- **Trade-off:** feature work pauses ~1–2 weeks.
- **Risk:** key rotation breaks live probes until re-keyed from env.

### Phase 1 — Stabilize the Contract (Months 1–4)
- **Goals:** adopters can trust what they depend on.
- **Work:** define Tier-1/2/3/4 stability (§4.3); publish SemVer + MSRV policy; add `#[non_exhaustive]` to public enums; feature-flag experimental surface; automate `CHANGELOG.md`; facade-freeze plan; resolve audit M-series (dead fallback, `mine_limits` tightening, `has_feature` tri-state, `ReasoningEffort::Max`).
- **Exit:** public API documented with stability badges; clippy + doc build green; one minor release under the new policy.
- **Trade-off:** velocity on Tier-1 drops; more review overhead.
- **Risk:** retrofitting stability on 30 crates surfaces hidden couplings.

### Phase 2 — Close the Deferred Surface (Months 4–10)
- **Goals:** deliver the PRD features currently marked "deferred," each behind a flag.
- **Work (priority order):** promote `ai-computer` (browser/desktop control already implemented) to a documented, feature-gated capability; IDE/LSP bridge; self-skill creation framework; GraphRAG; voice realtime (credential-gated); prompt registry + A/B + DSPy optimization (PRD §2.5). Each lands as Tier-4 → Tier-2 as it matures.
- **Exit:** each feature compiles, is feature-gated, and documented as "experimental" or "stable" — never "complete" unless verified.
- **Trade-off:** broad surface area stretches review capacity; keep crate ceiling (D3).
- **Risk:** provider API churn (OpenAI/Anthropic/Gemini) breaks adapters — keep them thin.

### Phase 3 — Scale the Runtime (Months 10–15)
- **Goals:** a reference topology that actually runs "1000s of agents."
- **Work:** durable checkpoint/state store backends (postgres/redis); swarm coordination over a real message bus (NATS/Kafka adapter behind `ai-runtime`); worker-pool autoscaling reference; OTel pipeline (`ai-observability` → collector); cost-governance (budget caps, per-tenant metering in `ai-analytics`); reference Docker Compose + K8s manifests.
- **Exit:** reference deployment runs the swarm benchmark from `benchmarks/` with stated p95 latency and cost; topology choice is config, not a fork.
- **Trade-off:** we now maintain a "runtime" in addition to a "library."
- **Risk (open question):** do we ship a *managed* runtime, or stay embeddable-only? Decide in Phase 2 review.

### Phase 4 — Ecosystem & Reach (Months 15–18)
- **Goals:** grow adoption without fracturing the core.
- **Work:** language bindings (TS/Python) over `ai-sidecar`; plugin/registry for skills & providers; managed reference deployment (one-click); docs + compiling examples per crate; a benchmark/SLA regime in CI.
- **Exit:** non-Rust host drives the SDK over `ai-sidecar`; example apps build against a released version.
- **Trade-off:** support burden shifts outward to the ecosystem.
- **Risk:** binding drift from the Rust API.

---

## 7. Proposed new / updated ADRs

- **ADR-014 — API Stability & Versioning Policy** (new; codifies §4.3 + Phase 1).
- **ADR-015 — Consumption Boundary & Crate Stability Tiers** (new; facade + `*-internal` seam, crate ceiling).
- **ADR-016 — Reference Runtime Topology for Agent Systems** (new; supersedes the TS framing of ADR-009).
- **ADR-017 — Security Hardening & Secret Handling** (new; ratifies fixes for C-1/C-2/H-1).
- **Re-baseline** ADR-002/003/004/005/006/007/008/010 to Rust (status Proposed → Accepted-in-Rust; language-neutral where possible). ADR-009 explicitly **superseded** by ADR-016.

---

## 8. Risks & open questions

- **Bus factor.** 30 crates, unclear ownership. Mitigation: `CODEOWNERS`, crate owners, architecture test.
- **Rust MSRV & WASM toolchain maturity** for `ai-edge` — pin and test in CI.
- **Provider API churn** — keep adapters thin; provider-native features pass through typed extension points (ADR-004 spirit).
- **Open:** managed runtime vs. embeddable-only? (drives Phase 3 scope)
- **Open:** what is the crate ceiling number? (proposed ~36)

---

## 9. Immediate next actions (this week)

1. **Rotate** the two exposed API keys; **delete** `tools/discovery-probe/cfg_*.json` and `out/*.log`; **extend** `.gitignore` (see audit C-1); **add** a `gitleaks` pre-commit scan.
2. **Fix** `Transport` `Debug` (redact key, C-2); **validate** `base_url` scheme + pin redirects (H-1); **stop overwriting** probe evidence and return `None` on abort (H-2).
3. **Schedule** the ADR re-baseline review (D1).
4. **Adopt** `CODEOWNERS` / crate owners.

---

*This document is a design proposal. It intentionally names trade-offs (what we give up) rather than only what we gain, and proposes decisions that are easy to reverse (thin adapters, feature flags, trait-based storage) over "optimal" forks. Ratify via the ADR process before implementation.*
