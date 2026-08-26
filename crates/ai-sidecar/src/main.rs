//! Binary entry: wires the sidecar server to stdin/stdout.
//!
//! Usage: `ai-sidecar [--config <path.toml>]`. With `--config`, providers
//! load from an `ai-config` TOML file before the first `configure`; without
//! one the sidecar starts unconfigured and waits for the host.

use std::sync::Arc;

use ai_sidecar::Sidecar;
use tokio::io::stdout;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_path = args
        .iter()
        .position(|arg| arg == "--config")
        .and_then(|index| args.get(index + 1));

    let writer: ai_sidecar::SharedWriter = Arc::new(tokio::sync::Mutex::new(Box::new(stdout())));
    let sidecar = match config_path {
        Some(path) => {
            let path = std::path::Path::new(path);
            match ai_config::Config::load_file(path) {
                Ok(file_config) => {
                    let mut config = ai_config::Config::default();
                    config.merge_file(file_config);
                    if let Err(err) = config.validate() {
                        eprintln!(
                            "ai-sidecar: invalid provider config in {}: {err}",
                            path.display()
                        );
                        std::process::exit(2);
                    }
                    match ai_providers::client_from_config(&config) {
                        Ok(client) => Sidecar::with_client(writer, client),
                        Err(err) => {
                            eprintln!(
                                "ai-sidecar: cannot build providers from {}: {err}",
                                path.display()
                            );
                            std::process::exit(2);
                        }
                    }
                }
                Err(err) => {
                    eprintln!("ai-sidecar: unreadable config {}: {err}", path.display());
                    std::process::exit(2);
                }
            }
        }
        None => Sidecar::new(writer),
    };

    Arc::new(sidecar).serve(tokio::io::stdin()).await;
}
