// Package segmenter groups a flat []Block stream into []Section values,
// one section per ATX header encountered.  Blocks that appear before the
// first header are collected in a "preamble" section with an empty title.
//
// Each Section carries a SectionPath breadcrumb that reflects the full
// heading hierarchy at the point of that section:
//
//	Input:
//	  # Chapter 1
//	  ## 1.1 Introduction
//	  ### 1.1.1 Background
//	  ## 1.2 Methods          ← drops 1.1.1 from the path
//
//	Output paths:
//	  ["Chapter 1"]
//	  ["Chapter 1", "1.1 Introduction"]
//	  ["Chapter 1", "1.1 Introduction", "1.1.1 Background"]
//	  ["Chapter 1", "1.2 Methods"]
package segmenter

import (
	"github.com/ETretyakov/kukri/types"
)

// Segment groups blocks into sections delimited by headers.
// Blocks before the first header go into a preamble section (empty title).
// Sections without any content blocks are omitted from the output.
func Segment(blocks []types.Block) []types.Section {
	var sections []types.Section

	current := types.Section{Path: []string{}}

	// path[level] holds the heading text currently active at that level (1–6).
	// Index 0 is unused so that levels map directly to slice indices.
	path := make([]string, 7)

	for _, b := range blocks {
		if b.Type == types.BlockHeader && (b.Level < 1 || b.Level > 6) {
			b.Type = types.BlockText // tolerate externally constructed invalid blocks
		}
		if b.Type != types.BlockHeader {
			current.Blocks = append(current.Blocks, b)
			continue
		}

		// Flush the current section before starting a new one.
		if len(current.Blocks) > 0 {
			sections = append(sections, current)
		}

		// Record this heading and clear all deeper levels so that a jump
		// from H3 back to H2 doesn't retain the stale H3 in the path.
		path[b.Level] = b.Content
		for l := b.Level + 1; l <= 6; l++ {
			path[l] = ""
		}

		current = types.Section{
			Title: b.Content,
			Level: b.Level,
			Path:  buildPath(path, b.Level),
		}
	}

	if len(current.Blocks) > 0 {
		sections = append(sections, current)
	}

	return sections
}

// buildPath returns a breadcrumb slice from level 1 down to level,
// skipping any levels that have no heading text (e.g. a doc that starts
// with ## without a preceding #).
func buildPath(path []string, level int) []string {
	var result []string
	for l := 1; l <= level; l++ {
		if path[l] != "" {
			result = append(result, path[l])
		}
	}
	return result
}
