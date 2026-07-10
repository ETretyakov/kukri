// Package scorer defines the Scorer interface for pluggable similarity backends.
//
// Implement Scorer to add custom merge logic (dense embeddings, BM25, etc.)
// and pass it to [chunker.New]. Passing nil uses token-count-only merging.
//
// Score[i] is the similarity between texts[i] and texts[i+1].
package scorer

import "context"

// Scorer computes pairwise cosine-similarity scores between adjacent text passages.
// len(result) == len(texts) - 1; returns nil when len(texts) < 2.
type Scorer interface {
	Score(ctx context.Context, texts []string) ([]float32, error)
}
