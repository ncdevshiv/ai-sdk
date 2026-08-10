//! Research layer (spec §12): Firecrawl-like operations implemented natively
//! and via the real Firecrawl REST API behind a common trait.

use std::collections::BTreeMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use ai_errors::{AiError, WebError};

use crate::search::SearchProvider;
use crate::web_client::WebClient;

/// A research result (page content + metadata).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchPage {
    pub url: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub markdown: String,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

/// A map operation result: discovered URLs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapResult {
    pub urls: Vec<String>,
}

/// Firecrawl-compatible research operations (spec §12).
#[async_trait]
pub trait ResearchBackend: Send + Sync {
    async fn search(&self, request: SearchRequest) -> Result<Vec<ResearchPage>, AiError>;
    async fn scrape(&self, request: ScrapeRequest) -> Result<ResearchPage, AiError>;
    async fn crawl(&self, request: CrawlRequest) -> Result<Vec<ResearchPage>, AiError>;
    async fn map(&self, request: MapRequest) -> Result<MapResult, AiError>;
    async fn extract(
        &self,
        request: ExtractRequest,
    ) -> Result<BTreeMap<String, serde_json::Value>, AiError>;
}

/// Request shapes (Firecrawl-compatible field names).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_ten")]
    pub limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrapeRequest {
    pub url: String,
    #[serde(default)]
    pub formats: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawlRequest {
    pub url: String,
    #[serde(default = "default_ten")]
    pub limit: usize,
    #[serde(default = "default_two")]
    pub max_depth: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapRequest {
    pub url: String,
    #[serde(default = "default_hundred")]
    pub limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractRequest {
    pub url: String,
    /// JSON Schema describing the data to extract.
    pub schema: serde_json::Value,
}

fn default_ten() -> usize {
    10
}
fn default_two() -> usize {
    2
}
fn default_hundred() -> usize {
    100
}

/// Extracts structured data from page content against a JSON Schema.
/// Implementations are real (e.g. LLM-driven); the native backend fails
/// fast when no extractor is configured instead of returning partial data.
#[async_trait::async_trait]
pub trait StructuredExtractor: Send + Sync {
    async fn extract(
        &self,
        content: &str,
        schema: serde_json::Value,
    ) -> Result<serde_json::Value, AiError>;
}

/// An LLM-driven structured extractor: asks a model to fill the schema and
/// returns parsed JSON (real generation through the provider).
pub struct LlmStructuredExtractor {
    model: std::sync::Arc<dyn ai_core::Model>,
}

impl LlmStructuredExtractor {
    pub fn new(model: std::sync::Arc<dyn ai_core::Model>) -> Self {
        Self { model }
    }
}

#[async_trait::async_trait]
impl StructuredExtractor for LlmStructuredExtractor {
    async fn extract(
        &self,
        content: &str,
        schema: serde_json::Value,
    ) -> Result<serde_json::Value, AiError> {
        let prompt = format!(
            "Extract the requested fields from the content below as a single JSON object              matching this schema: {schema}

Content:
{content}"
        );
        let request =
            ai_core::ChatRequest::new(vec![ai_types::Message::text(ai_types::Role::User, prompt)])
                .with_response_format(ai_core::ResponseFormat::JsonObject)
                .with_max_tokens(2000);
        let completion = self.model.generate(request).await?;
        serde_json::from_str(&completion.text).map_err(|e| {
            AiError::Web(WebError::new(
                "extract",
                format!("model output is not JSON: {e}"),
            ))
        })
    }
}

/// The native research backend: real search + fetch + crawl, no external
/// service required.
pub struct NativeResearchBackend {
    pub search: Box<dyn SearchProvider>,
    pub client: WebClient,
    pub extractor: Option<std::sync::Arc<dyn StructuredExtractor>>,
}

impl NativeResearchBackend {
    pub fn new(search: Box<dyn SearchProvider>, client: WebClient) -> Self {
        Self {
            search,
            client,
            extractor: None,
        }
    }

    pub fn with_extractor(mut self, extractor: std::sync::Arc<dyn StructuredExtractor>) -> Self {
        self.extractor = Some(extractor);
        self
    }
}

#[async_trait]
impl ResearchBackend for NativeResearchBackend {
    async fn search(&self, request: SearchRequest) -> Result<Vec<ResearchPage>, AiError> {
        let results = self.search.search(&request.query, request.limit).await?;
        let mut pages = Vec::new();
        for result in results {
            match self.client.fetch(&result.url).await {
                Ok(page) => pages.push(ResearchPage {
                    url: page.final_url,
                    title: page.title,
                    markdown: page.text,
                    metadata: BTreeMap::new(),
                }),
                Err(e) => {
                    // Partial results: keep the search hit without content.
                    pages.push(ResearchPage {
                        url: result.url,
                        title: Some(result.title),
                        markdown: format!("[fetch failed: {e}]"),
                        metadata: BTreeMap::new(),
                    });
                }
            }
        }
        Ok(pages)
    }

    async fn scrape(&self, request: ScrapeRequest) -> Result<ResearchPage, AiError> {
        let page = self.client.fetch(&request.url).await?;
        Ok(ResearchPage {
            url: page.final_url,
            title: page.title,
            markdown: page.text,
            metadata: BTreeMap::from([(
                "content_type".into(),
                serde_json::json!(page.content_type),
            )]),
        })
    }

    async fn crawl(&self, request: CrawlRequest) -> Result<Vec<ResearchPage>, AiError> {
        let crawler = crate::Crawler::new(
            self.client.clone(),
            crate::CrawlConfig {
                max_pages: request.limit,
                max_depth: request.max_depth,
                ..Default::default()
            },
        );
        let result = crawler.crawl(&request.url).await?;
        Ok(result
            .pages
            .into_iter()
            .map(|page| ResearchPage {
                url: page.url,
                title: page.title,
                markdown: page.text,
                metadata: BTreeMap::new(),
            })
            .collect())
    }

    async fn map(&self, request: MapRequest) -> Result<MapResult, AiError> {
        let page = self.client.fetch(&request.url).await?;
        let urls: Vec<String> = page.links.into_iter().take(request.limit).collect();
        Ok(MapResult { urls })
    }

    async fn extract(
        &self,
        request: ExtractRequest,
    ) -> Result<BTreeMap<String, serde_json::Value>, AiError> {
        let extractor = self.extractor.as_ref().ok_or_else(|| {
            AiError::Web(WebError::new(
                "extract",
                "no StructuredExtractor configured; build the backend with                  .with_extractor(LlmStructuredExtractor::new(model))",
            ))
        })?;
        let page = self.client.fetch(&request.url).await?;
        let data = extractor.extract(&page.text, request.schema).await?;
        let mut result = BTreeMap::new();
        result.insert("url".into(), serde_json::json!(page.final_url));
        if let Some(obj) = data.as_object() {
            for (key, value) in obj {
                result.insert(key.clone(), value.clone());
            }
        } else {
            result.insert("data".into(), data);
        }
        Ok(result)
    }
}

/// Firecrawl REST API backend (real adapter; requires `FIRECRAWL_API_KEY`).
///
/// Uses `https://api.firecrawl.dev/v1` endpoints: `/search`, `/scrape`,
/// `/crawl`, `/map`, `/extract`.
pub struct FirecrawlBackend {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

impl FirecrawlBackend {
    pub fn new(api_key: impl Into<String>) -> Result<Self, AiError> {
        Self::with_base_url(api_key, "https://api.firecrawl.dev/v1")
    }

    pub fn with_base_url(api_key: impl Into<String>, base_url: &str) -> Result<Self, AiError> {
        Ok(Self {
            api_key: api_key.into(),
            base_url: base_url.trim_end_matches('/').to_string(),
            client: reqwest::Client::builder()
                .user_agent("ai-sdk/0.1")
                .build()
                .map_err(|e| AiError::Web(WebError::new("firecrawl client", e.to_string())))?,
        })
    }

    async fn post_json(
        &self,
        path: &str,
        body: serde_json::Value,
    ) -> Result<serde_json::Value, AiError> {
        let response = self
            .client
            .post(format!("{}/{path}", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| AiError::Web(WebError::new("firecrawl", e.to_string())))?;

        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| AiError::Web(WebError::new("firecrawl", e.to_string())))?
            .to_vec();
        if !status.is_success() {
            return Err(AiError::Web(WebError::new(
                "firecrawl",
                format!("HTTP {status}: {}", String::from_utf8_lossy(&bytes)),
            )));
        }
        serde_json::from_slice(&bytes)
            .map_err(|e| AiError::Web(WebError::new("firecrawl", format!("invalid JSON: {e}"))))
    }
}

#[async_trait]
impl ResearchBackend for FirecrawlBackend {
    async fn search(&self, request: SearchRequest) -> Result<Vec<ResearchPage>, AiError> {
        let json = self
            .post_json(
                "search",
                serde_json::json!({"query": request.query, "limit": request.limit}),
            )
            .await?;
        let data = json
            .get("data")
            .and_then(|d| d.as_array())
            .cloned()
            .unwrap_or_default();
        Ok(data
            .into_iter()
            .map(|item| ResearchPage {
                url: item
                    .get("url")
                    .and_then(|u| u.as_str())
                    .unwrap_or("")
                    .to_string(),
                title: item.get("title").and_then(|t| t.as_str()).map(String::from),
                markdown: item
                    .get("markdown")
                    .and_then(|m| m.as_str())
                    .unwrap_or("")
                    .to_string(),
                metadata: BTreeMap::new(),
            })
            .collect())
    }

    async fn scrape(&self, request: ScrapeRequest) -> Result<ResearchPage, AiError> {
        let json = self
            .post_json("scrape", serde_json::json!({"url": request.url}))
            .await?;
        let data = json.get("data").cloned().unwrap_or(json);
        Ok(ResearchPage {
            url: data
                .get("url")
                .and_then(|u| u.as_str())
                .unwrap_or(&request.url)
                .to_string(),
            title: data.get("title").and_then(|t| t.as_str()).map(String::from),
            markdown: data
                .get("markdown")
                .and_then(|m| m.as_str())
                .unwrap_or("")
                .to_string(),
            metadata: BTreeMap::new(),
        })
    }

    async fn crawl(&self, request: CrawlRequest) -> Result<Vec<ResearchPage>, AiError> {
        let json = self
            .post_json(
                "crawl",
                serde_json::json!({
                    "url": request.url,
                    "limit": request.limit,
                    "maxDepth": request.max_depth
                }),
            )
            .await?;
        let data = json
            .get("data")
            .and_then(|d| d.as_array())
            .cloned()
            .unwrap_or_default();
        Ok(data
            .into_iter()
            .map(|item| ResearchPage {
                url: item
                    .get("url")
                    .and_then(|u| u.as_str())
                    .unwrap_or("")
                    .to_string(),
                title: item.get("title").and_then(|t| t.as_str()).map(String::from),
                markdown: item
                    .get("markdown")
                    .and_then(|m| m.as_str())
                    .unwrap_or("")
                    .to_string(),
                metadata: BTreeMap::new(),
            })
            .collect())
    }

    async fn map(&self, request: MapRequest) -> Result<MapResult, AiError> {
        let json = self
            .post_json(
                "map",
                serde_json::json!({"url": request.url, "limit": request.limit}),
            )
            .await?;
        let urls = json
            .get("links")
            .and_then(|l| l.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        Ok(MapResult { urls })
    }

    async fn extract(
        &self,
        request: ExtractRequest,
    ) -> Result<BTreeMap<String, serde_json::Value>, AiError> {
        let json = self
            .post_json(
                "extract",
                serde_json::json!({"urls": [request.url], "schema": request.schema}),
            )
            .await?;
        let mut map = BTreeMap::new();
        if let Some(data) = json.get("data") {
            if let Some(obj) = data.as_object() {
                for (key, value) in obj {
                    map.insert(key.clone(), value.clone());
                }
            }
        }
        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_core::{ChatRequest, Model};
    use ai_models::{ModelCapabilities, ModelInfo};
    use ai_types::{Completion, ModelId, ProviderId, StreamEvent, Usage};
    use std::sync::Arc;

    /// A deterministic model fixture for unit tests (ADR-007: unit tests
    /// use scripted models; live tests use the real gateway).
    pub struct ScriptedModel {
        output: String,
    }

    impl ScriptedModel {
        pub fn new(output: impl Into<String>) -> Self {
            Self {
                output: output.into(),
            }
        }
    }

    #[async_trait::async_trait]
    impl Model for ScriptedModel {
        fn info(&self) -> &ModelInfo {
            static INFO: std::sync::OnceLock<ModelInfo> = std::sync::OnceLock::new();
            INFO.get_or_init(|| {
                ModelInfo::new(
                    ProviderId::new("test"),
                    ModelId::new("scripted"),
                    128_000,
                    8_192,
                )
                .with_capabilities(ModelCapabilities::default())
            })
        }

        async fn generate(&self, _request: ChatRequest) -> Result<Completion, AiError> {
            Ok(Completion {
                provider: ProviderId::new("test"),
                model: ModelId::new("scripted"),
                text: self.output.clone(),
                tool_calls: vec![],
                usage: Usage::default(),
                reasoning: None,
                raw: serde_json::Value::Null,
                finish_reason: Some("stop".into()),
            })
        }

        async fn stream(&self, request: ChatRequest) -> Result<ai_core::EventStream, AiError> {
            let completion = self.generate(request).await?;
            let events = vec![
                Ok(StreamEvent::TextDelta {
                    delta: completion.text,
                }),
                Ok(StreamEvent::Completed {
                    finish_reason: None,
                }),
            ];
            Ok(Box::pin(futures::stream::iter(events)))
        }
    }

    #[tokio::test]
    async fn llm_extractor_parses_model_json_output() {
        let model: Arc<dyn Model> =
            Arc::new(ScriptedModel::new(r#"{"name":"Ada","role":"engineer"}"#));
        let extractor = LlmStructuredExtractor::new(model);
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "role": {"type": "string"}
            }
        });
        let data = extractor.extract("page content", schema).await.unwrap();
        assert_eq!(data["name"], "Ada");
        assert_eq!(data["role"], "engineer");
    }

    #[tokio::test]
    async fn llm_extractor_rejects_non_json_output() {
        let model: Arc<dyn Model> = Arc::new(ScriptedModel::new("not json at all"));
        let extractor = LlmStructuredExtractor::new(model);
        let err = extractor
            .extract("content", serde_json::json!({"type": "object"}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not JSON"), "{err}");
    }

    #[tokio::test]
    async fn native_extract_fails_fast_without_extractor() {
        let backend = NativeResearchBackend::new(
            Box::new(crate::DuckDuckGoSearch::default()),
            crate::WebClient::new(crate::WebClientConfig::default()).unwrap(),
        );
        let err = backend
            .extract(ExtractRequest {
                url: "https://example.com".into(),
                schema: serde_json::json!({"type": "object"}),
            })
            .await
            .unwrap_err();
        assert!(err.to_string().contains("no StructuredExtractor"), "{err}");
    }
}
