// Package kukri implements hybrid Markdown-aware + semantic chunking for
// Retrieval-Augmented Generation (RAG) pipelines.
//
// The pipeline has four stages:
//
//  1. [parser.Parse]     — line-by-line Markdown → typed blocks
//  2. [segmenter.Segment] — blocks → sections with breadcrumb paths
//  3. [embedder.Embed]   — sentence-level ONNX inference (all-MiniLM-L6-v2)
//  4. [chunker.New/Chop] — semantic greedy merge within each section
//
// Most callers only need this root package:
//
//	emb, err := kukri.NewEmbedder(ctx, "./models/KnightsAnalytics_all-MiniLM-L6-v2")
//	if err != nil { ... }
//	defer emb.Close()
//
//	c := kukri.New(kukri.Config{MaxTokens: 512, Threshold: 0.7}, emb)
//	chunks, err := c.Chop(ctx, markdownText)
//
// For access to individual pipeline stages import the sub-packages directly:
//
//	import "github.com/ETretyakov/kukri/parser"
//	import "github.com/ETretyakov/kukri/segmenter"
//	import "github.com/ETretyakov/kukri/embedder"
//	import "github.com/ETretyakov/kukri/chunker"
package kukri

import (
	"context"

	"github.com/ETretyakov/kukri/chunker"
	"github.com/ETretyakov/kukri/embedder"
	"github.com/ETretyakov/kukri/parser"
	"github.com/ETretyakov/kukri/segmenter"
	"github.com/ETretyakov/kukri/types"
)

// Type aliases so callers can use kukri.Chunk, kukri.Config, etc. without
// importing the types sub-package.
type (
	Chunk     = types.Chunk
	Config    = types.Config
	BlockType = types.BlockType
	Block     = types.Block
	Section   = types.Section
)

// BlockType constants re-exported for convenience.
const (
	BlockText    = types.BlockText
	BlockTable   = types.BlockTable
	BlockCode    = types.BlockCode
	BlockDiagram = types.BlockDiagram
	BlockList    = types.BlockList
	BlockImage   = types.BlockImage
)

// Embedder is re-exported from [embedder.Embedder].
type Embedder = embedder.Embedder

// NewEmbedder loads the ONNX sentence-transformer model at modelPath and
// returns a ready-to-use Embedder. Uses the pure-Go GoMLX backend — no
// ONNX Runtime shared library is required.
//
// Call [Embedder.Close] when done to release native resources.
func NewEmbedder(ctx context.Context, modelPath string) (*Embedder, error) {
	return embedder.New(ctx, modelPath)
}

// Chunker is the top-level entry point. It accepts raw Markdown text and
// returns a slice of linked [Chunk] values ready for embedding and indexing.
type Chunker struct {
	inner *chunker.Chunker
}

// New creates a Chunker. emb may be nil — semantic grouping is then skipped
// and text blocks are split by token count alone.
func New(cfg Config, emb *Embedder) *Chunker {
	return &Chunker{inner: chunker.New(cfg, emb)}
}

// Chop parses, segments, and chunks the Markdown text in a single call.
// Chunks are doubly linked via PrevID / NextID and carry section breadcrumbs,
// block type metadata, source line numbers, and token counts.
func (c *Chunker) Chop(ctx context.Context, text string) ([]Chunk, error) {
	return c.inner.Chop(ctx, segmenter.Segment(parser.Parse(text)))
}
