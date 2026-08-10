//! Hybrid retrieval: BM25-style keyword scoring combined with semantic
//! cosine similarity.

use std::collections::HashMap;

use crate::RetrievedChunk;

/// Tokenizes text into lowercase word stems.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// BM25 keyword score of `query` against a document.
///
/// `k1` controls term-frequency saturation (typical 1.2), `b` controls
/// document-length normalization (typical 0.75). With no corpus statistics
/// available we use a fixed average document length (1.0-normalized).
pub fn bm25_score(query: &str, document: &str, k1: f32, b: f32) -> f32 {
    let query_terms = tokenize(query);
    let doc_terms = tokenize(document);
    if doc_terms.is_empty() || query_terms.is_empty() {
        return 0.0;
    }
    let doc_len = doc_terms.len() as f32;
    let mut term_freq: HashMap<String, usize> = HashMap::new();
    for term in &doc_terms {
        *term_freq.entry(term.clone()).or_insert(0) += 1;
    }

    let mut score = 0.0f32;
    for term in &query_terms {
        let tf = *term_freq.get(term).unwrap_or(&0) as f32;
        if tf == 0.0 {
            continue;
        }
        // idf ≈ ln(1 + (N - n + 0.5)/(n + 0.5)) with N=1, n=1 → ln(1.0)=0;
        // use a small constant so present terms contribute.
        let idf = 1.0f32;
        let normalization = 1.0 - b + b * (doc_len / 1.0);
        score += idf * (tf * (k1 + 1.0)) / (tf + k1 * normalization);
    }
    score
}

/// Cosine similarity between two vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return None;
    }
    Some(dot / (na.sqrt() * nb.sqrt()))
}

/// Ranks candidate chunks by BM25 keyword overlap with `query`.
/// Returns a map from candidate index → keyword score.
pub fn keyword_search(
    query: &str,
    candidates: &[RetrievedChunk],
    k1: f32,
    b: f32,
) -> HashMap<usize, f32> {
    candidates
        .iter()
        .enumerate()
        .map(|(index, chunk)| (index, bm25_score(query, &chunk.text, k1, b)))
        .filter(|(_, score)| *score > 0.0)
        .collect()
}

/// Combines semantic and keyword scores (RRF-style fusion is available via
/// [`hybrid_fusion`]).
pub fn hybrid_fusion(
    semantic: &[(String, f32)],
    keyword: &HashMap<String, f32>,
    alpha: f32,
) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    for (id, score) in semantic {
        *scores.entry(id.clone()).or_insert(0.0) += alpha * score;
    }
    for (id, score) in keyword {
        *scores.entry(id.clone()).or_insert(0.0) += (1.0 - alpha) * score;
    }
    let mut ranked: Vec<(String, f32)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_splits_and_lowercases() {
        assert_eq!(tokenize("Hello, World! 42"), vec!["hello", "world", "42"]);
        assert!(tokenize("   ").is_empty());
    }

    #[test]
    fn bm25_ranks_relevant_documents_higher() {
        let relevant = "The quick brown fox jumps over the lazy dog. Foxes are quick.";
        let irrelevant = "The stock market opened flat on Tuesday morning.";
        let q = "quick fox";
        let score_relevant = bm25_score(q, relevant, 1.2, 0.75);
        let score_irrelevant = bm25_score(q, irrelevant, 1.2, 0.75);
        assert!(
            score_relevant > score_irrelevant,
            "{score_relevant} vs {score_irrelevant}"
        );
        assert!(score_irrelevant == 0.0);
    }

    #[test]
    fn cosine_identical_and_orthogonal() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]).unwrap() - 1.0).abs() < 1e-6);
        assert!((cosine(&[1.0, 0.0], &[0.0, 1.0]).unwrap() - 0.0).abs() < 1e-6);
        assert!(cosine(&[1.0], &[1.0, 2.0]).is_none());
    }

    #[test]
    fn keyword_search_scores_only_matching() {
        let candidates = vec![
            RetrievedChunk {
                id: "a".into(),
                text: "rust is a systems language".into(),
                score: 0.0,
                metadata: serde_json::Value::Null,
            },
            RetrievedChunk {
                id: "b".into(),
                text: "python is a scripting language".into(),
                score: 0.0,
                metadata: serde_json::Value::Null,
            },
        ];
        let scores = keyword_search("rust", &candidates, 1.2, 0.75);
        assert!(scores.contains_key(&0));
        assert!(!scores.contains_key(&1));
    }

    #[test]
    fn hybrid_fusion_combines_scores() {
        let semantic = vec![("a".to_string(), 0.8), ("b".to_string(), 0.2)];
        let keyword: HashMap<String, f32> =
            HashMap::from([("a".to_string(), 2.0), ("c".to_string(), 1.0)]);
        let ranked = hybrid_fusion(&semantic, &keyword, 0.5);
        assert_eq!(ranked[0].0, "a", "present in both ranks first");
        assert!(ranked.iter().any(|(id, _)| id == "c"));
    }
}
