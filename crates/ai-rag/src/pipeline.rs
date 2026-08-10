//! The RAG pipeline: chunk → embed → store → retrieve (hybrid) → rerank →
//! assemble context.

use std::sync::Arc;

use ai_errors::{AiError, StorageError};
use ai_memory::EmbeddingsProvider;
use ai_storage::{VectorEntry, VectorStore};

use crate::chunking::{ChunkingStrategy, chunk_document};
use crate::hybrid::{hybrid_fusion, keyword_search};
use crate::{IdentityReranker, Reranker};

/// A retrieved chunk with its score.
#[derive(Debug, Clone)]
pub struct RetrievedChunk {
    pub id: String,
    pub text: String,
    pub score: f32,
    pub metadata: serde_json::Value,
}

/// RAG pipeline configuration.
#[derive(Debug, Clone)]
pub struct RagConfig {
    pub chunking: ChunkingStrategy,
    /// Semantic similarity threshold for vector hits.
    pub min_similarity: f32,
    /// Blend factor for hybrid fusion (1 = semantic only, 0 = keyword only).
    pub hybrid_alpha: f32,
    /// BM25 k1 parameter.
    pub bm25_k1: f32,
    /// BM25 b parameter.
    pub bm25_b: f32,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            chunking: ChunkingStrategy::Fixed {
                size: 1200,
                overlap: 200,
            },
            min_similarity: 0.5,
            hybrid_alpha: 0.7,
            bm25_k1: 1.2,
            bm25_b: 0.75,
        }
    }
}

/// A RAG pipeline over a vector store with an embeddings provider.
///
/// An in-memory [`EntryMirror`] keeps chunk payloads retrievable by id
/// (the vector store interface is search-only by design).
pub struct RagPipeline {
    store: Arc<dyn VectorStore>,
    embeddings: Arc<dyn EmbeddingsProvider>,
    config: RagConfig,
    reranker: Arc<dyn Reranker>,
    mirror: mirror::EntryMirror,
}

impl RagPipeline {
    pub fn new(
        store: Arc<dyn VectorStore>,
        embeddings: Arc<dyn EmbeddingsProvider>,
        config: RagConfig,
    ) -> Self {
        Self {
            store,
            embeddings,
            config,
            reranker: Arc::new(IdentityReranker),
            mirror: mirror::EntryMirror::new(),
        }
    }

    pub fn with_reranker(mut self, reranker: Arc<dyn Reranker>) -> Self {
        self.reranker = reranker;
        self
    }

    /// Ingests a document: chunk, embed each chunk, upsert into the store.
    pub async fn ingest(&self, document_id: &str, document: &str) -> Result<usize, AiError> {
        let chunks = chunk_document(document, self.config.chunking);
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let vectors = self
            .embeddings
            .embed(&texts)
            .await
            .map_err(|e| AiError::Storage(StorageError::new("rag", e.to_string())))?;

        for (chunk, vector) in chunks.into_iter().zip(vectors) {
            let id = format!("{document_id}:{chunk_index:05}", chunk_index = chunk.index);
            let entry = VectorEntry {
                id: id.clone(),
                vector,
                payload: serde_json::json!({
                    "document": document_id,
                    "text": chunk.text,
                    "start": chunk.start,
                    "index": chunk.index,
                }),
            };
            self.mirror.put(entry.clone());
            self.store.upsert(entry).await?;
        }
        Ok(texts.len())
    }

    /// Retrieves the top-k chunks for a query using hybrid search.
    pub async fn retrieve(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<RetrievedChunk>, AiError> {
        let query_vector = self
            .embeddings
            .embed(&[query.to_string()])
            .await
            .map_err(|e| AiError::Storage(StorageError::new("rag", e.to_string())))?
            .into_iter()
            .next()
            .ok_or_else(|| {
                AiError::Storage(StorageError::new("rag", "embeddings returned no vector"))
            })?;

        // Semantic candidates (superset for fusion).
        let semantic = self.store.search(&query_vector, top_k * 3).await?;
        let semantic: Vec<(String, f32)> = semantic
            .into_iter()
            .filter(|(_, score)| *score >= self.config.min_similarity)
            .map(|(entry, score)| (entry.id, score))
            .collect();

        // Keyword candidates over the semantic hits.
        let semantic_chunks: Vec<RetrievedChunk> = semantic
            .iter()
            .filter_map(|(id, score)| {
                let entry = self.mirror.get(id)?;
                Some(RetrievedChunk {
                    id: id.clone(),
                    text: entry
                        .payload
                        .get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                    score: *score,
                    metadata: entry.payload.clone(),
                })
            })
            .collect();

        let keyword = keyword_search(
            query,
            &semantic_chunks,
            self.config.bm25_k1,
            self.config.bm25_b,
        );
        let keyword_by_id: std::collections::HashMap<String, f32> = keyword
            .iter()
            .filter_map(|(index, score)| {
                semantic_chunks.get(*index).map(|c| (c.id.clone(), *score))
            })
            .collect();

        // Hybrid fusion.
        let mut fused = hybrid_fusion(&semantic, &keyword_by_id, self.config.hybrid_alpha);
        fused.truncate(top_k);

        let mut retrieved = Vec::with_capacity(fused.len());
        for (id, score) in fused {
            if let Some(entry) = self.mirror.get(&id) {
                retrieved.push(RetrievedChunk {
                    id,
                    text: entry
                        .payload
                        .get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                    score,
                    metadata: entry.payload,
                });
            }
        }

        self.reranker.rerank(query, retrieved).await
    }
}

/// Assembles a prompt context from retrieved chunks.
pub struct ContextAssembler {
    /// Separator between chunks.
    pub separator: String,
    /// Header line.
    pub header: String,
    /// Maximum total characters (truncates from the end).
    pub max_chars: usize,
}

impl Default for ContextAssembler {
    fn default() -> Self {
        Self {
            separator: "\n\n---\n\n".to_string(),
            header: "Relevant context:".to_string(),
            max_chars: 8_000,
        }
    }
}

impl ContextAssembler {
    pub fn assemble(&self, chunks: &[RetrievedChunk]) -> String {
        let mut out = String::new();
        out.push_str(&self.header);
        out.push('\n');
        for chunk in chunks {
            if out.len() >= self.max_chars {
                break;
            }
            if !out.ends_with('\n') {
                out.push('\n');
            }
            out.push_str(&self.separator);
            out.push_str(&chunk.text);
        }
        out.truncate(self.max_chars.min(out.len()));
        out
    }
}

// The store trait lacks point reads; RagPipeline keeps a mirror of ingested
// entries so retrieval can reconstruct chunks without extra API surface.
pub(crate) mod mirror {
    use std::collections::HashMap;
    use std::sync::Arc;

    use parking_lot::RwLock;

    use ai_storage::VectorEntry;

    /// An in-memory mirror of entries (id → entry) kept in sync with the
    /// vector store by [`crate::pipeline::RagPipeline`].
    #[derive(Clone, Default)]
    pub struct EntryMirror {
        entries: Arc<RwLock<HashMap<String, VectorEntry>>>,
    }

    impl EntryMirror {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn put(&self, entry: VectorEntry) {
            self.entries.write().insert(entry.id.clone(), entry);
        }

        pub fn get(&self, id: &str) -> Option<VectorEntry> {
            self.entries.read().get(id).cloned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_memory::EmbeddingsError;
    use ai_storage::InMemoryVectorStore;
    use std::sync::Arc;

    /// Deterministic hash-based embeddings (no external service).
    struct HashEmbeddings;

    #[async_trait::async_trait]
    impl EmbeddingsProvider for HashEmbeddings {
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingsError> {
            Ok(texts
                .iter()
                .map(|text| {
                    let mut vector = vec![0.0f32; 16];
                    for word in crate::hybrid::tokenize(text) {
                        let index = word.bytes().fold(0usize, |acc, b| (acc + b as usize) % 16);
                        vector[index] += 1.0;
                    }
                    vector
                })
                .collect())
        }
    }

    #[tokio::test]
    async fn rag_ingest_and_retrieve_roundtrip() {
        let store: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new(1000));
        let pipeline = RagPipeline::new(
            store,
            Arc::new(HashEmbeddings),
            RagConfig {
                chunking: ChunkingStrategy::Fixed {
                    size: 200,
                    overlap: 20,
                },
                min_similarity: 0.3,
                ..Default::default()
            },
        );

        let document = "The AI SDK supports many providers. ".repeat(20)
            + "Rust is a systems programming language. "
            + "Web crawling respects robots.txt rules.";
        let chunk_count = pipeline.ingest("doc1", &document).await.unwrap();
        assert!(
            chunk_count >= 3,
            "document split into chunks: {chunk_count}"
        );

        let results = pipeline
            .retrieve("rust programming language", 3)
            .await
            .unwrap();
        assert!(!results.is_empty(), "query must retrieve chunks");
        assert!(
            results.iter().any(|c| c.text.contains("Rust")),
            "the Rust chunk is retrieved: {:?}",
            results.iter().map(|c| &c.text).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn keyword_reranker_reorders_by_query_terms() {
        let store: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new(100));
        let pipeline = RagPipeline::new(
            store,
            Arc::new(HashEmbeddings),
            RagConfig {
                chunking: ChunkingStrategy::Fixed {
                    size: 1000,
                    overlap: 0,
                },
                min_similarity: 0.1,
                hybrid_alpha: 0.5,
                ..Default::default()
            },
        )
        .with_reranker(Arc::new(crate::KeywordReranker { blend: 0.8 }));

        pipeline
            .ingest("d1", "The cat sat on the mat.")
            .await
            .unwrap();
        pipeline
            .ingest("d2", "Dogs love to run in parks.")
            .await
            .unwrap();

        let results = pipeline.retrieve("cat mat", 2).await.unwrap();
        assert!(!results.is_empty());
        // The cat document should rank first for this query.
        assert!(
            results[0].text.contains("cat"),
            "{:?}",
            results.iter().map(|c| &c.text).collect::<Vec<_>>()
        );
    }

    #[test]
    fn context_assembler_formats_chunks() {
        let assembler = ContextAssembler::default();
        let chunks = vec![
            RetrievedChunk {
                id: "a".into(),
                text: "first chunk".into(),
                score: 1.0,
                metadata: serde_json::Value::Null,
            },
            RetrievedChunk {
                id: "b".into(),
                text: "second chunk".into(),
                score: 0.9,
                metadata: serde_json::Value::Null,
            },
        ];
        let context = assembler.assemble(&chunks);
        assert!(context.contains("Relevant context"));
        assert!(context.contains("first chunk"));
        assert!(context.contains("second chunk"));
        assert!(context.contains("---"));
    }

    #[test]
    fn tokenize_available_for_keyword_reranker_tests() {
        assert_eq!(crate::hybrid::tokenize("Rust Lang"), vec!["rust", "lang"]);
    }
}
