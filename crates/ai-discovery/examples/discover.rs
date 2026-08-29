//! Live discovery runner.
//!
//! Runs the full discovery pipeline against a real provider and emits both a
//! machine-readable result set and a chronological journal.
//!
//! Usage:
//! ```text
//! cargo run -p ai-discovery --example discover -- \
//!     --name bai --base-url https://api.b.ai/v1 --key "$KEY" \
//!     --concurrency 3 --out out/bai.json
//! ```
//!
//! Flags mirror [`DiscoveryConfig`]; every capability toggle can be disabled
//! so an expensive probe can be skipped on a heavily throttled gateway.

use std::sync::Arc;
use std::time::Duration;

use ai_discovery::{DiscoveredModel, DiscoveryConfig, DiscoveryEngine};
use clap::Parser;
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(name = "discover", about = "Run generic model/capability discovery")]
struct Args {
    /// Provider label used in output (no behavioural effect).
    #[arg(long)]
    name: String,
    /// Base URL, e.g. https://api.example.com/v1
    #[arg(long)]
    base_url: String,
    /// API key.
    #[arg(long)]
    key: String,
    /// Per-request timeout in seconds.
    #[arg(long, default_value_t = 60)]
    timeout: u64,
    /// Transport pacing policy: default | conservative | none.
    ///
    /// `conservative` (≈1.2 s between requests, patient backoff) is required
    /// for gateways that throttle without advertising limits — otherwise the
    /// discovery run reports its own throttling as model failure.
    #[arg(long, default_value = "default")]
    policy: String,
    /// Retry attempts for retryable failures, overriding the policy preset.
    ///
    /// This is a *time budget*, not a reliability knob: retrying a gateway
    /// that never answers costs `attempts x timeout` per request, and a
    /// discovery run issues ~17 requests per model. The defaults (4 attempts
    /// x 60 s) make a dead model cost up to four minutes of wall clock.
    #[arg(long, default_value_t = 0)]
    attempts: usize,
    /// Skip every capability probe and report reachability only.
    ///
    /// Turns a run into a cheap triage pass (3 requests per model instead of
    /// ~17) so a large catalog can be triaged before anything expensive is
    /// spent on the models that answer.
    #[arg(long)]
    triage: bool,
    /// Cap total wall-clock for the run, in seconds (0 = unlimited).
    ///
    /// Without a global budget a single pathological gateway can stall a run
    /// indefinitely; on expiry the results gathered so far are still written.
    #[arg(long, default_value_t = 0)]
    budget: u64,
    /// How many models to probe concurrently.
    #[arg(long, default_value_t = 2)]
    concurrency: usize,
    /// Max models to probe (0 = all).
    #[arg(long, default_value_t = 0)]
    limit: usize,
    /// Extra model ids to probe even if not listed by /models.
    #[arg(long, value_delimiter = ',', default_value = "")]
    extra: Vec<String>,
    /// Only probe these model ids (comma separated).
    #[arg(long, value_delimiter = ',', default_value = "")]
    only: Vec<String>,
    /// Skip the image-input probe.
    #[arg(long)]
    no_vision: bool,
    /// Skip the tool-calling probe.
    #[arg(long)]
    no_tools: bool,
    /// Skip structured-output probes.
    #[arg(long)]
    no_structured: bool,
    /// Skip endpoint-routing probes.
    #[arg(long)]
    no_endpoints: bool,
    /// Skip the reasoning-toggle battery.
    #[arg(long)]
    no_thinking: bool,
    /// Binary-search the context window (slow).
    #[arg(long)]
    probe_context: bool,
    /// Upper bound for the context search.
    #[arg(long, default_value_t = 64_000)]
    max_context: usize,
    /// Output JSON path.
    #[arg(long, default_value = "discovery.json")]
    out: String,
}

#[derive(Serialize)]
struct Report {
    provider: String,
    base_url: String,
    generated_at: String,
    total: usize,
    reachable: usize,
    models: Vec<DiscoveredModel>,
}

fn now() -> String {
    // Local wall-clock stamp; kept dependency-free on purpose.
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("unix:{}", d.as_secs())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let timeout = Duration::from_secs(args.timeout);
    let mut policy = match args.policy.as_str() {
        "conservative" => ai_discovery::probe::TransportPolicy::conservative(),
        "none" => ai_discovery::probe::TransportPolicy::none(),
        _ => ai_discovery::probe::TransportPolicy::default(),
    };
    if args.attempts > 0 {
        policy.max_attempts = args.attempts;
    }
    // Captured before `policy` is moved into the engine, for the startup log.
    let attempts = policy.max_attempts;
    let cfg = DiscoveryConfig {
        timeout,
        transport_policy: policy.clone(),
        // `--triage` turns off the whole capability battery; what remains is
        // the reachability probe, the streaming probe and the output-ceiling
        // probe, which is the cheapest set that still says "does it answer".
        probe_vision: !args.no_vision && !args.triage,
        probe_tools: !args.no_tools && !args.triage,
        probe_structured_output: !args.no_structured && !args.triage,
        // Endpoint routing stays ON under `--triage`. It is what tells us a
        // model is an embedder rather than broken; disabling it made every
        // non-chat model in a catalog look dead.
        probe_endpoints: !args.no_endpoints,
        probe_thinking: !args.no_thinking && !args.triage,
        probe_context: args.probe_context,
        max_context_probe: args.max_context,
        limit: args.limit,
        context_rounds: 6,
        extra_models: args.extra.clone(),
    };

    let engine = Arc::new(DiscoveryEngine::with_policy(
        args.name.clone(),
        args.base_url.clone(),
        args.key.clone(),
        timeout,
        policy,
    )?);

    let mut entries = engine.list_models().await.map_err(|e| {
        eprintln!("[{}] FATAL: could not list models: {e}", now());
        std::process::exit(1);
    })?;
    let mut ids: Vec<(String, Option<serde_json::Value>, bool)> = Vec::new();
    for e in entries.drain(..) {
        if let Some(id) = e.get("id").and_then(|i| i.as_str()) {
            ids.push((id.to_string(), Some(e), true));
        }
    }
    for extra in &cfg.extra_models {
        // Guard against an empty element from `--extra ""`, which would
        // otherwise probe a model with a blank id and pollute the report.
        if extra.trim().is_empty() || ids.iter().any(|(i, _, _)| i == extra) {
            continue;
        }
        ids.push((extra.clone(), None, false));
    }
    if !args.only.is_empty() && !args.only.iter().any(|o| o.is_empty()) {
        ids.retain(|(i, _, _)| args.only.contains(i));
    }
    if cfg.limit > 0 {
        ids.truncate(cfg.limit);
    }

    let started = std::time::Instant::now();
    let budget = if args.budget > 0 {
        Some(Duration::from_secs(args.budget))
    } else {
        None
    };

    eprintln!(
        "[{}] {} : probing {} models (concurrency {}, attempts {}, timeout {:?})",
        now(),
        args.name,
        ids.len(),
        args.concurrency,
        attempts,
        timeout
    );

    let mut results: Vec<DiscoveredModel> = Vec::new();
    // Chunk so a throttled gateway is never hammered by the full set at once.
    'outer: for chunk in ids.chunks(args.concurrency.max(1)) {
        // A global budget matters because a subset of any real catalog is
        // dead, and each dead model costs `attempts x timeout` per probe.
        // Without this a run can stall for hours; with it, whatever was
        // measured is still written out.
        if let Some(b) = budget {
            if started.elapsed() >= b {
                eprintln!(
                    "[{}] budget of {:?} exhausted after {} of {} models; writing partial results",
                    now(),
                    b,
                    results.len(),
                    ids.len()
                );
                break 'outer;
            }
        }
        let mut handles = Vec::new();
        for (id, entry, listed) in chunk {
            let engine = Arc::clone(&engine);
            let cfg = cfg.clone();
            let id = id.clone();
            let entry = entry.clone();
            let listed = *listed;
            handles.push(tokio::spawn(async move {
                engine.discover_one(&id, entry.as_ref(), listed, &cfg).await
            }));
        }
        for h in handles {
            match h.await {
                Ok(m) => {
                    eprintln!(
                        "[{}] {} | {} | {}",
                        now(),
                        if m.reachable { "  OK  " } else { " FAIL " },
                        m.summary(),
                        m.blocker.as_deref().unwrap_or("")
                    );
                    results.push(m);
                }
                Err(e) => eprintln!("[{}] task panicked: {e}", now()),
            }
        }
    }

    results.sort_by(|a, b| a.id.cmp(&b.id));

    let reachable = results.iter().filter(|m| m.reachable).count();
    let report = Report {
        provider: args.name.clone(),
        base_url: args.base_url.clone(),
        generated_at: now(),
        total: results.len(),
        reachable,
        models: results,
    };

    if let Some(parent) = std::path::Path::new(&args.out).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(&args.out, serde_json::to_string_pretty(&report)?)?;
    eprintln!(
        "[{}] {} : {}/{} reachable -> {}",
        now(),
        args.name,
        reachable,
        report.total,
        args.out
    );

    Ok(())
}
