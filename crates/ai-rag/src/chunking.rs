//! Document chunking: fixed-size (with overlap) and sentence-based
//! strategies (PRD §3.8.1).

/// A chunk of a document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub text: String,
    /// Character offset of the chunk start in the source document.
    pub start: usize,
    pub index: usize,
}

/// Chunking strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkingStrategy {
    /// Fixed-size chunks with optional overlap (in characters).
    Fixed { size: usize, overlap: usize },
    /// Split on sentence boundaries (`.`, `!`, `?`), keeping chunks under
    /// `max_size` characters.
    Sentence { max_size: usize },
}

/// Splits a document into chunks per the strategy.
pub fn chunk_document(document: &str, strategy: ChunkingStrategy) -> Vec<Chunk> {
    match strategy {
        ChunkingStrategy::Fixed { size, overlap } => {
            let size = size.max(1);
            let overlap = overlap.min(size.saturating_sub(1));
            let mut chunks = Vec::new();
            let mut index = 0usize;
            let mut start = 0usize;
            while start < document.len() {
                let end = (start + size).min(document.len());
                let text = &document[start..end];
                if text.trim().is_empty() {
                    break;
                }
                chunks.push(Chunk {
                    text: text.to_string(),
                    start,
                    index,
                });
                index += 1;
                if end >= document.len() {
                    break;
                }
                start = end.saturating_sub(overlap);
            }
            chunks
        }
        ChunkingStrategy::Sentence { max_size } => {
            let max_size = max_size.max(1);
            let mut chunks = Vec::new();
            let mut current = String::new();
            let mut current_start = 0usize;
            let mut index = 0usize;

            let mut char_positions: Vec<usize> = document.char_indices().map(|(i, _)| i).collect();
            char_positions.push(document.len());

            for (i, sentence) in split_sentences(document).into_iter().enumerate() {
                let sentence_start = char_positions.get(i).copied().unwrap_or(0);
                if current.len() + sentence.len() > max_size && !current.is_empty() {
                    chunks.push(Chunk {
                        text: std::mem::take(&mut current),
                        start: current_start,
                        index,
                    });
                    index += 1;
                    current_start = sentence_start;
                }
                current.push_str(&sentence);
            }
            if !current.trim().is_empty() {
                chunks.push(Chunk {
                    text: current,
                    start: current_start,
                    index,
                });
            }
            chunks
        }
    }
}

/// Splits text into sentences on `.`, `!`, `?` boundaries (keeping the
/// terminator), handling common abbreviations conservatively (a terminator
/// followed by whitespace + capital letter or end-of-input).
fn split_sentences(document: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let mut chars = document.chars().peekable();

    while let Some(ch) = chars.next() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            // Look ahead: sentence ends if the next non-space char is a
            // capital letter or there is no more input.
            let mut ahead = chars.clone().peekable();
            let mut next_meaningful = None;
            for candidate in ahead.by_ref() {
                if !candidate.is_whitespace() {
                    next_meaningful = Some(candidate);
                    break;
                }
            }
            let ends = match next_meaningful {
                None => true,
                Some(c) => c.is_uppercase() || c.is_ascii_digit(),
            };
            if ends {
                // Include trailing whitespace with the sentence.
                while let Some(&ws) = chars.peek() {
                    if ws.is_whitespace() {
                        current.push(ws);
                        chars.next();
                    } else {
                        break;
                    }
                }
                sentences.push(std::mem::take(&mut current));
            }
        }
    }
    if !current.trim().is_empty() {
        sentences.push(current);
    }
    sentences
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_chunking_respects_size_and_overlap() {
        let doc = "abcdefghijklmnopqrstuvwxyz";
        let chunks = chunk_document(
            doc,
            ChunkingStrategy::Fixed {
                size: 10,
                overlap: 3,
            },
        );
        assert_eq!(chunks.len(), 4, "{chunks:?}");
        assert_eq!(chunks[0].text, "abcdefghij");
        assert_eq!(chunks[1].text, "hijklmnopq");
        assert_eq!(chunks[0].start, 0);
        assert!(chunks[1].start > 0);
        // The whole document is covered (with overlap).
        assert!(chunks.last().unwrap().text.ends_with('z'));
    }

    #[test]
    fn fixed_chunking_handles_short_documents() {
        let chunks = chunk_document(
            "hi",
            ChunkingStrategy::Fixed {
                size: 10,
                overlap: 0,
            },
        );
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hi");
    }

    #[test]
    fn sentence_chunking_splits_on_boundaries() {
        let doc = "First sentence. Second sentence! Third? Fourth.";
        // With a large max_size the whole document is one chunk.
        let one = chunk_document(doc, ChunkingStrategy::Sentence { max_size: 200 });
        assert_eq!(one.len(), 1);
        assert!(one[0].text.contains("Fourth."));
        // With a small max_size, chunks split at sentence boundaries.
        let many = chunk_document(doc, ChunkingStrategy::Sentence { max_size: 17 });
        assert!(many.len() >= 3, "split at boundaries: {many:?}");
        assert!(many[0].text.contains("First sentence"), "{:?}", many[0]);
        assert!(many[0].text.contains('.'), "keeps the terminator");
    }

    #[test]
    fn sentence_chunking_bounds_chunk_size() {
        let doc = "Alpha. ".repeat(60);
        let chunks = chunk_document(&doc, ChunkingStrategy::Sentence { max_size: 100 });
        assert!(
            chunks.iter().all(|c| c.text.len() <= 100),
            "all chunks bounded"
        );
        assert!(chunks.len() >= 2, "multiple chunks: {}", chunks.len());
    }
}
