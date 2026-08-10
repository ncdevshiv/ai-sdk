//! Provider adapters: real HTTP integrations with LLM APIs.
//!
//! Current state:
//!
//! - [`openai_compat`] — full OpenAI Chat Completions wire protocol adapter.
//!   Serves `openai`, `openrouter`, `ollama`, and arbitrary OpenAI-compatible
//!   gateways (e.g. the project gateway at `opencode.ai/zen/go/v1`).
//!   Streaming, tool calling, structured output, vision input, embeddings
//!   capability metadata, DeepSeek-style reasoning, cache-aware usage.
//! - Anthropic and Google Gemini native adapters: **not yet implemented**
//!   (documented limitation; their APIs differ from the OpenAI wire format).
//!   See `ENGINEERING-SPEC.md` §40 for status.
//!
//! Nothing here is mocked: every adapter performs real HTTP requests.

pub mod http;
pub mod openai_compat;

use std::sync::Arc;

use ai_core::Provider;
use ai_errors::{AiError, ConfigurationError};
use openai_compat::{OpenAiCompatConfig, OpenAiCompatProvider};

/// Creates a provider adapter from its id and configuration.
///
/// Provider ids currently supported (OpenAI-compatible wire protocol):
/// `openai`, `openrouter`, `ollama`, and any custom id with an explicit
/// `base_url` (used for OpenAI-compatible gateways).
pub fn create_provider(
    id: &str,
    config: &ai_config::ProviderConfig,
) -> Result<Arc<dyn Provider>, AiError> {
    let openai_compat_config = OpenAiCompatConfig::from_provider_config(id, config)?;
    Ok(Arc::new(OpenAiCompatProvider::new(openai_compat_config)?))
}

/// Builds a provider from a raw id, api key, and base URL (programmatic
/// configuration path).
pub fn create_provider_direct(
    id: impl Into<String>,
    api_key: impl Into<String>,
    base_url: impl Into<String>,
) -> Result<Arc<dyn Provider>, AiError> {
    let config = OpenAiCompatConfig::new(id, api_key, base_url);
    Ok(Arc::new(OpenAiCompatProvider::new(config)?))
}

/// Builds the AI SDK client from a [`ai_config::Config`], registering every
/// configured provider.
pub fn client_from_config(config: &ai_config::Config) -> Result<ai_core::AiClient, AiError> {
    config.validate()?;
    let mut builder = ai_core::AiClient::builder();
    for (name, provider_config) in &config.providers {
        if provider_config.api_key.is_some() {
            builder = builder.provider(create_provider(name, provider_config)?);
        }
    }
    builder.build().map_err(|e| {
        AiError::Configuration(ConfigurationError::with_source(
            "providers",
            "failed to build AiClient from config",
            e,
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_without_key_fails_with_typed_error() {
        let cfg = ai_config::ProviderConfig::default();
        let err = match create_provider("openai", &cfg) {
            Ok(_) => panic!("expected an error for a provider without an API key"),
            Err(e) => e,
        };
        assert!(matches!(err, AiError::Configuration(_)));
        assert!(err.to_string().contains("OPENAI_API_KEY"), "{err}");
    }

    #[test]
    fn openai_compat_provider_debug_redacts_key() {
        let provider = OpenAiCompatProvider::new(OpenAiCompatConfig::new(
            "opencode",
            "sk-super-secret",
            "https://opencode.ai/zen/go/v1",
        ))
        .unwrap();
        let debug = format!("{provider:?}");
        assert!(!debug.contains("sk-super-secret"), "{debug}");
        assert!(debug.contains("***redacted***"), "{debug}");
    }
}
