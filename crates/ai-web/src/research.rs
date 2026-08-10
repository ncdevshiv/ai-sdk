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

/// The native research backend: real search + fetch + crawl, no external
/// service required.
pub struct NativeResearchBackend {
    pub search: Box<dyn SearchProvider>,
    pub client: WebClient,
}

impl NativeResearchBackend {
    pub fn new(search: Box<dyn SearchProvider>, client: WebClient) -> Self {
        Self { search, client }
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
        // Native structured extraction: fetch the page and use the LLM
        // gateway via a callback-free approach — we return the raw text and
        // schema so callers (e.g. agents) can drive the model themselves.
        // To keep this honest, we expose the fetched content; the actual
        // schema-constrained extraction is delegated to `ai-rag`/agents.
        let page = self.client.fetch(&request.url).await?;
        Ok(BTreeMap::from([
            ("url".into(), serde_json::json!(page.final_url)),
            ("content".into(), serde_json::json!(page.text)),
            ("schema".into(), request.schema),
        ]))
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
