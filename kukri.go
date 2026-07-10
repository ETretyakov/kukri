// Package kukri implements structure-aware Markdown chunking for
// Retrieval-Augmented Generation (RAG) pipelines.
//
// Pipeline:
//  1. [parser.Parse]      — line-by-line Markdown → typed blocks
//  2. [segmenter.Segment] — blocks → sections with breadcrumb paths
//  3. [chunker.New/Chop]  — greedy block-level merge within each section
//
// Quickstart:
//
//	c := kukri.New(kukri.Config{MaxTokens: 128, Threshold: 0}, nil)
//	chunks, err := c.Chop(ctx, markdownText)
//
// With a custom similarity scorer:
//
//	var scr kukri.Scorer = myScorer{}
//	c := kukri.New(kukri.Config{MaxTokens: 128, Threshold: 0.6}, scr)
package kukri

import (
	"context"

	"github.com/ETretyakov/kukri/chunker"
	"github.com/ETretyakov/kukri/parser"
	"github.com/ETretyakov/kukri/scorer"
	"github.com/ETretyakov/kukri/segmenter"
	"github.com/ETretyakov/kukri/types"
)

// Type aliases so callers can use kukri.Chunk, kukri.Config, etc.
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

// Scorer is the pairwise similarity interface, re-exported from [scorer.Scorer].
// Implement it to plug in a custom merge backend (dense embeddings, BM25, etc.).
type Scorer = scorer.Scorer

// Chunker is the top-level entry point.
type Chunker struct {
	inner *chunker.Chunker
}

// New creates a Chunker. scr may be nil — blocks are then merged purely by
// token count (all similarities treated as 0).
func New(cfg Config, scr Scorer) *Chunker {
	return &Chunker{inner: chunker.New(cfg, scr)}
}

// Chop parses, segments, and chunks the Markdown text in a single call.
// Chunks are doubly linked via PrevID/NextID and carry section breadcrumbs,
// block type metadata, source line numbers, and token counts.
func (c *Chunker) Chop(ctx context.Context, text string) ([]Chunk, error) {
	return c.inner.Chop(ctx, segmenter.Segment(parser.Parse(text)))
}
