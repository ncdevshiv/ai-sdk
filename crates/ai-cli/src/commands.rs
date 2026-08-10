//! Command implementations for the `ai-sdk` CLI. Every command does real
//! work: environment checks, real provider API calls, trace inspection.

use std::sync::Arc;

use ai_config::Config;
use ai_core::AiClient;
use ai_devtools::Inspector;
use ai_errors::AiError;
use ai_observability::EventCollector;
use ai_providers::client_from_config;

/// Builds the client from the environment (`.env` + env vars).
pub fn build_client() -> Result<AiClient, AiError> {
    let config = Config::load()?;
    client_from_config(&config)
}

/// `ai-sdk doctor`: environment and provider health checks.
pub async fn doctor() -> Result<(), AiError> {
    let config = Config::load()?;
    println!("== ai-sdk doctor ==");
    println!("config: {}", config.redacted_summary());

    let mut ok = true;
    for (name, provider_config) in &config.providers {
        let key_status = if provider_config.api_key.is_some() {
            "key: set"
        } else {
            ok = false;
            "key: MISSING"
        };
        println!("provider {name}: {key_status}");
    }

    // Probe the gateway with a real model list call.
    match build_client() {
        Ok(client) => {
            for provider_id in client.provider_ids() {
                match client.provider(&provider_id) {
                    Ok(provider) => match provider.list_models().await {
                        Ok(models) => {
                            println!(
                                "provider {provider_id}: reachable ({} models)",
                                models.len()
                            );
                        }
                        Err(e) => {
                            ok = false;
                            println!("provider {provider_id}: UNREACHABLE: {e}");
                        }
                    },
                    Err(e) => {
                        ok = false;
                        println!("provider {provider_id}: error: {e}");
                    }
                }
            }
        }
        Err(e) => {
            ok = false;
            println!("client build failed: {e}");
        }
    }

    if ok {
        println!("doctor: OK");
    } else {
        println!("doctor: ISSUES FOUND (see above)");
    }
    Ok(())
}

/// `ai-sdk providers`: lists configured providers.
pub async fn providers() -> Result<(), AiError> {
    let config = Config::load()?;
    if config.providers.is_empty() {
        println!("no providers configured");
        return Ok(());
    }
    for (name, provider_config) in &config.providers {
        let model = provider_config
            .default_model
            .clone()
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{name}\tmodel={model}\tbase_url={}",
            provider_config.base_url.as_deref().unwrap_or("-")
        );
    }
    Ok(())
}

/// `ai-sdk models`: real `GET /models` against a provider.
pub async fn models(provider_name: &str) -> Result<(), AiError> {
    let client = build_client()?;
    let provider = client.provider(provider_name)?;
    let models = provider.list_models().await?;
    println!("{} models from {provider_name}:", models.len());
    for model in models {
        println!("  {}", model.id);
    }
    Ok(())
}

/// `ai-sdk config`: validates and prints a redacted summary.
pub fn config() -> Result<(), AiError> {
    let config = Config::load()?;
    config.validate()?;
    println!("{}", config.redacted_summary());
    Ok(())
}

/// `ai-sdk run`: real generation against a model.
pub async fn run(model: &str, prompt: &str, stream: bool) -> Result<(), AiError> {
    let client = build_client()?;
    if stream {
        let events = client
            .stream(
                model,
                vec![ai_types::Message::text(ai_types::Role::User, prompt)],
            )
            .await?;
        let text = ai_stream::collect_text(events).await?;
        println!("{text}");
    } else {
        let completion = client
            .generate(
                model,
                vec![ai_types::Message::text(ai_types::Role::User, prompt)],
            )
            .await?;
        println!("{}", completion.text);
        if let Some(reasoning) = &completion.reasoning {
            println!("\n[reasoning: {} chars]", reasoning.len());
        }
        println!(
            "[tokens: in={} out={}]",
            completion.usage.input_tokens, completion.usage.output_tokens
        );
    }
    Ok(())
}

/// `ai-sdk inspect`: summarizes traces from a JSONL export.
pub fn inspect(file: &str) -> Result<(), AiError> {
    let collector = load_trace_file(file)?;
    let inspector = Inspector::new(collector);
    let traces = inspector.traces();
    println!("{} traces in {file}", traces.len());
    for trace in traces {
        println!(
            "  trace {}: {} events, {:?} ({:?})",
            trace.trace_id,
            trace.events.len(),
            std::time::Duration::from_millis(trace.duration_ms),
            trace.status
        );
    }
    Ok(())
}

/// `ai-sdk trace`: chronological report of one trace (redacted).
pub fn trace(file: &str, trace_id: Option<String>) -> Result<(), AiError> {
    let collector = load_trace_file(file)?;
    let inspector = Inspector::new(collector);
    let trace_id = match trace_id {
        Some(id) => id,
        None => inspector
            .traces()
            .first()
            .map(|t| t.trace_id.clone())
            .ok_or_else(|| AiError::Internal(ai_errors::InternalError::new("no traces in file")))?,
    };
    println!("== trace {trace_id} ==");
    print!("{}", inspector.report(&trace_id));
    Ok(())
}

/// `ai-sdk benchmark`: real parallel generation latency/throughput.
pub async fn benchmark(model: &str, requests: usize, concurrency: usize) -> Result<(), AiError> {
    let client = Arc::new(build_client()?);
    let tasks: Vec<ai_runtime::Task<ai_types::Completion>> = (0..requests)
        .map(|index| {
            let client = client.clone();
            let model = model.to_string();
            ai_runtime::Task::new(format!("bench-{index}"), async move {
                client
                    .generate(
                        &model,
                        vec![ai_types::Message::text(
                            ai_types::Role::User,
                            "Reply with exactly: OK",
                        )],
                    )
                    .await
            })
        })
        .collect();

    let started = std::time::Instant::now();
    let results = ai_runtime::Parallel::new()
        .with_limit(concurrency)
        .execute(tasks)
        .await;
    let elapsed = started.elapsed();

    let succeeded = results.iter().filter(|r| r.succeeded()).count();
    let total_tokens: u64 = results
        .iter()
        .filter_map(|r| r.outcome.as_ref().ok())
        .map(|c| c.usage.total())
        .sum();

    println!(
        "benchmark: {succeeded}/{} succeeded in {:.2}s (concurrency {concurrency})",
        results.len(),
        elapsed.as_secs_f64()
    );
    println!(
        "  throughput: {:.1} req/s, latency p50 estimate {:.0} ms",
        succeeded as f64 / elapsed.as_secs_f64().max(0.001),
        elapsed.as_millis() as f64 / succeeded.max(1) as f64
    );
    println!("  total tokens: {total_tokens}");
    Ok(())
}

/// Loads a JSONL event export into a collector.
fn load_trace_file(file: &str) -> Result<EventCollector, AiError> {
    let content = std::fs::read_to_string(file).map_err(|e| {
        AiError::Internal(ai_errors::InternalError::new(format!(
            "cannot read {file}: {e}"
        )))
    })?;
    let collector = EventCollector::new();
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let event: ai_observability::ExecutionEvent = serde_json::from_str(line).map_err(|e| {
            AiError::Internal(ai_errors::InternalError::new(format!(
                "invalid event line: {e}"
            )))
        })?;
        collector.record_with_ids(
            event.kind,
            &event.operation,
            event.status,
            event.metadata,
            event.trace_id,
            event.span_id,
            event.parent_span_id,
            event.duration_ms,
        );
    }
    Ok(collector)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_trace_file_parses_jsonl() {
        let dir = std::env::temp_dir();
        let path = dir.join("ai-sdk-cli-test.jsonl");
        std::fs::write(
            &path,
            "{\"wall_time\":\"x\",\"offset_ms\":0,\"trace_id\":\"t1\",\"span_id\":\"s1\",\"kind\":\"model_call\",\"operation\":\"op\",\"status\":\"succeeded\"}\n",
        )
        .unwrap();
        let collector = load_trace_file(path.to_str().unwrap()).unwrap();
        assert_eq!(collector.events().len(), 1);
        std::fs::remove_file(&path).ok();
    }
}
