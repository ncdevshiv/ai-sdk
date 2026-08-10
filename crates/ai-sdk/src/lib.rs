//! The AI SDK — unified public API.
//!
//! This facade re-exports the workspace crates behind a single import so
//! applications can build with `use ai_sdk::*` and access every subsystem:
//! providers, streaming, agents, tools, memory, RAG, workflows, protocols
//! (MCP + A2A), web research, observability, analytics, security, caching,
//! storage, voice, edge, and the CLI commands.

pub use ai_agents::{Agent, AgentBuilder, AgentResult, AgentState, AgentSwarm, HumanInTheLoop};
pub use ai_analytics::{Metric, MetricsRegistry, RateCounter};
pub use ai_cache::{SemanticCache, TtlCache, cosine_similarity};
pub use ai_cli::commands;
pub use ai_config::{Config, ProviderConfig};
pub use ai_core::{
    AiClient, AiClientBuilder, ChatRequest, Model, Provider, ResponseFormat, ToolDefinition,
};
pub use ai_devtools::Inspector;
pub use ai_edge::{Capabilities, Runtime, detect_runtime, unsupported};
pub use ai_errors::{AiError, Result};
pub use ai_memory::{CompactingMemory, LongTermMemory, Memory, SemanticMemory, WorkingMemory};
pub use ai_models::{ModelCapabilities, ModelInfo, ModelRegistry, Pricing, default_catalog};
pub use ai_observability::{EventCollector, EventKind, EventStatus, ExecutionEvent};
pub use ai_protocols::{
    A2AClient, A2AServer, AgentCard, McpClient, McpHttpClient, McpResource, McpServer, McpTool,
    TaskStatus,
};
pub use ai_providers::{create_provider, create_provider_direct, openai_compat};
pub use ai_rag::{ChunkingStrategy, ContextAssembler, KeywordReranker, RagPipeline, Reranker};
pub use ai_runtime::{CircuitBreaker, ConcurrencyLimiter, Parallel, RetryPolicy, Task};
pub use ai_security::{Permissions, PiiDetector, Redactor, UrlPolicy};
pub use ai_storage::{DocumentStore, InMemoryVectorStore, KeyValueStore, SqliteStore, VectorStore};
pub use ai_stream::{collect_completion, collect_text, sse_parse};
pub use ai_tools::{Tool, ToolContext, ToolOutput, ToolRegistry, default_tools, run_tool};
pub use ai_types::{Completion, ContentPart, Message, Role, StreamEvent, Usage};
pub use ai_voice::{Audio, SpeechToText, TextToSpeech, VoiceActivityDetector};
pub use ai_web::{
    Crawler, DuckDuckGoSearch, FirecrawlBackend, NativeResearchBackend, ResearchBackend,
    RobotsPolicy, SearchProvider, WebClient,
};
pub use ai_workflows::{Node, NodeBuilder, NodeHandler, Workflow};

/// Convenience prelude for typical applications.
pub mod prelude {
    pub use ai_agents::{Agent, AgentBuilder};
    pub use ai_core::{AiClient, ChatRequest, Model};
    pub use ai_memory::Memory;
    pub use ai_tools::{Tool, ToolRegistry, default_tools};
    pub use ai_types::{Message, Role, StreamEvent};
}
