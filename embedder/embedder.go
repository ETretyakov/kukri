// Package embedder wraps a hugot sentence-transformer pipeline for
// computing dense vector representations of short text passages.
//
// The pure-Go GoMLX backend is used so no ONNX Runtime shared library is
// needed: the binary is fully self-contained.
//
// Typical usage:
//
//	emb, err := embedder.New(ctx, "./models/KnightsAnalytics_all-MiniLM-L6-v2")
//	if err != nil { ... }
//	defer emb.Close()
//
//	vecs, err := emb.Embed(ctx, []string{"first sentence", "second sentence"})
//	sim := embedder.CosineSimilarity(vecs[0], vecs[1])
package embedder

import (
	"context"
	"math"

	hugot "github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
)

// Embedder wraps a hugot feature-extraction pipeline.
//
// A single Embedder may be shared across goroutines for reads (Close,
// CosineSimilarity), but individual Embed calls must not be issued
// concurrently — hugot's RunPipeline is not goroutine-safe.
type Embedder struct {
	session  *hugot.Session
	pipeline *pipelines.FeatureExtractionPipeline
}

// New loads the sentence-transformer ONNX model from modelPath and
// returns a ready-to-use Embedder. Call Close when done to release
// native resources.
func New(ctx context.Context, modelPath string) (*Embedder, error) {
	session, err := hugot.NewGoSession(ctx)
	if err != nil {
		return nil, err
	}

	config := hugot.FeatureExtractionConfig{
		ModelPath: modelPath,
		Name:      "kukri-embedder",
		Options: []hugot.FeatureExtractionOption{
			// WithNormalization makes every output vector unit-length so
			// cosine similarity reduces to a plain dot product.
			pipelines.WithNormalization(),
		},
	}

	pipe, err := hugot.NewPipeline[*pipelines.FeatureExtractionPipeline](session, config)
	if err != nil {
		_ = session.Destroy()
		return nil, err
	}

	return &Embedder{session: session, pipeline: pipe}, nil
}

// Embed returns one 384-dim embedding vector per input sentence.
// Vectors are L2-normalised (see WithNormalization above), so
// CosineSimilarity between them is equivalent to the dot product.
func (e *Embedder) Embed(ctx context.Context, sentences []string) ([][]float32, error) {
	out, err := e.pipeline.RunPipeline(ctx, sentences)
	if err != nil {
		return nil, err
	}
	return out.Embeddings, nil
}

// CosineSimilarity returns the cosine similarity ∈ [-1, 1] between two
// L2-normalised vectors. For vectors produced by Embed this is identical
// to the dot product; the clamp guards against floating-point drift.
func CosineSimilarity(a, b []float32) float32 {
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return float32(math.Max(-1, math.Min(1, dot)))
}

// Close releases the hugot session and all loaded model weights.
func (e *Embedder) Close() error {
	return e.session.Destroy()
}
