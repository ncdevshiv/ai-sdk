//! Shared HTTP layer for provider adapters: request execution with typed
//! error mapping (auth, rate limit, provider, network, timeout) and
//! streaming bodies.

use std::time::Duration;

use ai_errors::{
    AiError, AuthenticationError, NetworkError, ProviderError, RateLimitError, SerializationError,
    TimeoutError,
};

/// A shared, connection-pooled HTTP client.
#[derive(Clone)]
pub struct HttpClient {
    inner: reqwest::Client,
}

impl HttpClient {
    /// Builds the client with sensible defaults: pooled connections, gzip
    /// content decoding, and TLS via rustls.
    pub fn new() -> Result<Self, AiError> {
        let inner = reqwest::Client::builder()
            .user_agent(concat!("ai-sdk/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| {
                AiError::Internal(ai_errors::InternalError::with_source(
                    "http client build failed",
                    e,
                ))
            })?;
        Ok(Self { inner })
    }

    /// Builds the client with a per-request connect timeout.
    pub fn new_with_connect_timeout(timeout: Duration) -> Result<Self, AiError> {
        let inner = reqwest::Client::builder()
            .user_agent(concat!("ai-sdk/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(timeout)
            .build()
            .map_err(|e| {
                AiError::Internal(ai_errors::InternalError::with_source(
                    "http client build failed",
                    e,
                ))
            })?;
        Ok(Self { inner })
    }

    pub fn inner(&self) -> &reqwest::Client {
        &self.inner
    }
}

/// Parses an OpenAI-style error body (`{"error": {"message", "type", "code"}}`)
/// or falls back to the raw text.
fn parse_error_body(text: &str) -> (String, Option<String>) {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(text) {
        if let Some(error) = value.get("error") {
            let message = error
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or(text)
                .to_string();
            let code = error
                .get("code")
                .and_then(|c| c.as_str())
                .map(|c| c.to_string());
            return (message, code);
        }
    }
    (text.to_string(), None)
}

/// Maps an HTTP response to a typed [`AiError`] based on status code.
pub async fn map_response_error(
    provider: &str,
    status: reqwest::StatusCode,
    body: &[u8],
) -> AiError {
    let text = String::from_utf8_lossy(body);
    let (message, code) = parse_error_body(&text);

    match status.as_u16() {
        401 | 403 => AiError::Authentication(AuthenticationError::new(format!(
            "{provider} rejected the API key (HTTP {status}): {message}"
        ))),
        429 => {
            let mut err = RateLimitError::new(provider, message);
            err.retry_after = parse_retry_after(&text);
            AiError::RateLimit(err)
        }
        400 | 404 | 405 | 415 | 422 => AiError::Provider(
            ProviderError::new(provider, message)
                .with_status(status.as_u16())
                .with_code(code.unwrap_or_default()),
        ),
        500..=599 => AiError::Provider(
            ProviderError::new(provider, message)
                .with_status(status.as_u16())
                .with_code(code.unwrap_or_default()),
        ),
        _ => AiError::Provider(
            ProviderError::new(provider, format!("unexpected HTTP {status}: {message}"))
                .with_status(status.as_u16()),
        ),
    }
}

/// Extracts `Retry-After` from an OpenAI-style rate-limit payload or header
/// text when present.
fn parse_retry_after(body: &str) -> Option<Duration> {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(body) {
        if let Some(after) = value
            .pointer("/error/retry_after")
            .or_else(|| value.get("retry_after"))
        {
            if let Some(secs) = after.as_f64() {
                return Some(Duration::from_secs_f64(secs.max(0.0)));
            }
        }
    }
    None
}

/// Maps a `reqwest` failure to a typed [`AiError`].
pub fn map_reqwest_error(operation: &str, err: reqwest::Error) -> AiError {
    if err.is_timeout() {
        AiError::Timeout(TimeoutError::new(operation, Duration::from_secs(0)))
    } else {
        AiError::Network(NetworkError::new(operation, err.to_string()))
    }
}

/// Parses a JSON body, mapping failures to a serialization error.
pub fn parse_json<T: serde::de::DeserializeOwned>(
    operation: &str,
    body: &[u8],
) -> Result<T, AiError> {
    serde_json::from_slice(body).map_err(|e| {
        AiError::Serialization(SerializationError::new(format!(
            "failed to parse {operation} response: {e}"
        )))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_error_body() {
        let (msg, code) = parse_error_body(
            r#"{"error":{"message":"bad request","type":"invalid_request_error","code":"invalid_param"}}"#,
        );
        assert_eq!(msg, "bad request");
        assert_eq!(code.as_deref(), Some("invalid_param"));
    }

    #[test]
    fn falls_back_to_raw_text() {
        let (msg, code) = parse_error_body("<html>gateway error</html>");
        assert_eq!(msg, "<html>gateway error</html>");
        assert!(code.is_none());
    }

    #[test]
    fn parses_retry_after_from_body() {
        let d = parse_retry_after(r#"{"error":{"retry_after":2.5}}"#);
        assert_eq!(d, Some(Duration::from_secs_f64(2.5)));
    }

    #[tokio::test]
    async fn maps_status_codes() {
        let e = map_response_error(
            "openai",
            reqwest::StatusCode::UNAUTHORIZED,
            b"{\"error\":{\"message\":\"nope\"}}",
        )
        .await;
        assert!(matches!(e, AiError::Authentication(_)));

        let e = map_response_error(
            "openai",
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            b"{\"error\":{\"message\":\"slow\"}}",
        )
        .await;
        assert!(matches!(e, AiError::RateLimit(_)));

        let e = map_response_error(
            "openai",
            reqwest::StatusCode::BAD_REQUEST,
            b"{\"error\":{\"message\":\"bad\"}}",
        )
        .await;
        assert!(matches!(e, AiError::Provider(_)));
        assert!(!e.is_retryable());

        let e = map_response_error(
            "openai",
            reqwest::StatusCode::SERVICE_UNAVAILABLE,
            b"{\"error\":{\"message\":\"busy\"}}",
        )
        .await;
        assert!(e.is_retryable());
    }
}
