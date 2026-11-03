// Package scorer defines the Scorer interface for pluggable similarity backends.
//
// Implement Scorer to add custom merge logic (dense embeddings, BM25, etc.)
// and pass it to chunker.New with MergeSimilarity. Other modes use no Scorer.
//
// Score[i] is the similarity between texts[i] and texts[i+1].
package scorer

import "context"

// Scorer computes finite similarities in [-1,1] between adjacent text passages.
// len(result) == len(texts) - 1; returns nil when len(texts) < 2.
type Scorer interface {
	Score(ctx context.Context, texts []string) ([]float32, error)
}
