//! The `ai-sdk` command-line interface (spec §29): every command performs
//! real work against the configured providers.

pub mod commands;

use clap::{Parser, Subcommand};

/// The AI SDK CLI.
#[derive(Parser)]
#[command(name = "ai-sdk", version, about = "AI SDK command-line interface")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Checks the environment and provider configuration.
    Doctor,
    /// Lists configured providers.
    Providers,
    /// Lists models available from a provider (real API call).
    Models {
        /// Provider id (defaults to the gateway provider).
        #[arg(long, default_value = "opencode")]
        provider: String,
    },
    /// Shows and validates the configuration (redacted).
    Config,
    /// Runs a prompt against a model and prints the completion.
    Run {
        /// Model reference, e.g. `opencode:deepseek-v4-flash`.
        #[arg(long, default_value = "opencode:deepseek-v4-flash")]
        model: String,
        /// The user prompt.
        prompt: String,
        /// Stream the response.
        #[arg(long)]
        stream: bool,
    },
    /// Inspects execution traces from a JSONL export.
    Inspect {
        /// Path to a `.jsonl` trace export.
        file: String,
    },
    /// Prints a chronological report of a trace.
    Trace {
        /// Path to a `.jsonl` trace export.
        file: String,
        /// Trace id to report (defaults to the first trace).
        #[arg(long)]
        trace_id: Option<String>,
    },
    /// Benchmarks generation latency/throughput against a model.
    Benchmark {
        /// Model reference.
        #[arg(long, default_value = "opencode:deepseek-v4-flash")]
        model: String,
        /// Number of parallel requests.
        #[arg(long, default_value = "8")]
        requests: usize,
        /// Concurrency limit.
        #[arg(long, default_value = "4")]
        concurrency: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parses_subcommands() {
        let cli = Cli::try_parse_from(["ai-sdk", "doctor"]).unwrap();
        assert!(matches!(cli.command, Commands::Doctor));

        let cli = Cli::try_parse_from(["ai-sdk", "run", "--model", "x:y", "hello"]).unwrap();
        match cli.command {
            Commands::Run {
                model,
                prompt,
                stream,
            } => {
                assert_eq!(model, "x:y");
                assert_eq!(prompt, "hello");
                assert!(!stream);
            }
            _ => panic!("expected run"),
        }
    }
}
