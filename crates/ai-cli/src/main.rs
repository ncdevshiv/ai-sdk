//! The `ai-sdk` CLI binary.

use ai_cli::{Cli, Commands};
use clap::Parser;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Doctor => ai_cli::commands::doctor().await,
        Commands::Providers => ai_cli::commands::providers().await,
        Commands::Models { provider } => ai_cli::commands::models(&provider).await,
        Commands::Config => ai_cli::commands::config(),
        Commands::Run {
            model,
            prompt,
            stream,
        } => ai_cli::commands::run(&model, &prompt, stream).await,
        Commands::Inspect { file } => ai_cli::commands::inspect(&file),
        Commands::Trace { file, trace_id } => ai_cli::commands::trace(&file, trace_id),
        Commands::Benchmark {
            model,
            requests,
            concurrency,
        } => ai_cli::commands::benchmark(&model, requests, concurrency).await,
    };

    match result {
        Ok(()) => Ok(()),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
