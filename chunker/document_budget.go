package chunker

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/ETretyakov/kukri/types"
)

func (c *Chunker) validateDocumentBudget() error {
	b := c.cfg.DocumentBudget
	if b == nil {
		return nil
	}
	if b.MaxTokens < 0 || b.MaxChunks < 0 || math.IsNaN(b.RetainFraction) || math.IsInf(b.RetainFraction, 0) || b.RetainFraction < 0 || b.RetainFraction > 1 {
		return fmt.Errorf("DocumentBudget requires nonnegative limits and RetainFraction in [0,1]")
	}
	if b.MaxTokens == 0 && b.MaxChunks == 0 && b.RetainFraction == 0 {
		return fmt.Errorf("DocumentBudget requires at least one limit")
	}
	if (b.MaxTokens > 0 || b.RetainFraction > 0) && c.cfg.EmbeddingBudget == nil {
		return fmt.Errorf("DocumentBudget token limits require EmbeddingBudget")
	}
	switch b.Strategy {
	case types.SelectUniform, types.SelectTerms, types.SelectTermsContext:
	default:
		return fmt.Errorf("unknown document selection strategy %q", b.Strategy)
	}
	return nil
}

// Chop preserves the original API. Use ChopWithReport to audit lossy selection.
func (c *Chunker) Chop(ctx context.Context, sections []types.Section) ([]types.Chunk, error) {
	r, err := c.ChopWithReport(ctx, sections)
	return r.Chunks, err
}

// ChopWithReport returns a nil Compression report when selection is disabled.
func (c *Chunker) ChopWithReport(ctx context.Context, sections []types.Section) (types.Result, error) {
	chunks, err := c.chopAll(ctx, sections)
	if err != nil {
		return types.Result{}, err
	}
	if c.cfg.DocumentBudget == nil {
		return types.Result{Chunks: chunks}, nil
	}
	selected, report, err := c.selectDocument(ctx, chunks, sections)
	if err != nil {
		return types.Result{}, err
	}
	return types.Result{Chunks: selected, Compression: report}, nil
}

func (c *Chunker) selectDocument(ctx context.Context, chunks []types.Chunk, sections []types.Section) ([]types.Chunk, *types.CompressionReport, error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	b := c.cfg.DocumentBudget
	r := &types.CompressionReport{Strategy: b.Strategy, OriginalChunks: len(chunks), ChunkLimit: b.MaxChunks, TokenBudgetEnabled: b.MaxTokens > 0 || b.RetainFraction > 0}
	for _, ch := range chunks {
		if ch.EmbeddingTokens > int(^uint(0)>>1)-r.OriginalTokens {
			return nil, nil, fmt.Errorf("document token count overflows int")
		}
		r.OriginalTokens += ch.EmbeddingTokens
	}
	r.TokenLimit = r.OriginalTokens
	if b.RetainFraction > 0 && b.RetainFraction < 1 {
		r.TokenLimit = int(math.Floor(b.RetainFraction * float64(r.OriginalTokens)))
	}
	if b.MaxTokens > 0 && b.MaxTokens < r.TokenLimit {
		r.TokenLimit = b.MaxTokens
	}
	if !r.TokenBudgetEnabled {
		r.TokenLimit = 0
	}
	if (!r.TokenBudgetEnabled || r.OriginalTokens <= r.TokenLimit) && (b.MaxChunks == 0 || len(chunks) <= b.MaxChunks) {
		r.SelectedChunks, r.SelectedTokens = len(chunks), r.OriginalTokens
		return chunks, r, nil
	}
	features, weights := documentFeatures(chunks, sections)
	deps := make([][]int, len(chunks))
	if b.Strategy == types.SelectTermsContext {
		deps = precedingContext(chunks, sections)
	}
	selected := make([]bool, len(chunks))
	counts := make([]int, len(weights))
	priority := uniformOrder(len(chunks))
	for {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		best, bestScore := -1, -1.0
		var bestBundle []int
		for _, i := range priority {
			if selected[i] {
				continue
			}
			bundle := make([]int, 0, len(deps[i])+1)
			for _, j := range deps[i] {
				if !selected[j] {
					bundle = append(bundle, j)
				}
			}
			bundle = append(bundle, i)
			cost := 0
			for _, j := range bundle {
				cost += chunks[j].EmbeddingTokens
			}
			if (r.TokenBudgetEnabled && cost > r.TokenLimit-r.SelectedTokens) || (b.MaxChunks > 0 && len(bundle) > b.MaxChunks-r.SelectedChunks) {
				continue
			}
			if b.Strategy == types.SelectUniform {
				best, bestBundle = i, bundle
				break
			}
			gain := 0.0
			// Repeated terms contribute less, both across selections and within a bundle.
			if len(bundle) == 1 {
				for _, term := range features[i] {
					gain += weights[term] / float64(1+counts[term])
				}
			} else {
				local := map[int]int{}
				for _, j := range bundle {
					for _, term := range features[j] {
						gain += weights[term] / float64(1+counts[term]+local[term])
						local[term]++
					}
				}
			}
			denominator := len(bundle)
			if r.TokenBudgetEnabled {
				denominator = cost
			}
			if denominator < 1 {
				denominator = 1
			}
			score := gain / float64(denominator)
			if score > bestScore || (score == bestScore && (best < 0 || i < best)) {
				best, bestScore, bestBundle = i, score, bundle
			}
		}
		if best < 0 {
			break
		}
		for _, j := range bestBundle {
			selected[j] = true
			r.SelectedChunks++
			r.SelectedTokens += chunks[j].EmbeddingTokens
			if j != best {
				r.ContextAddedChunks++
			}
			for _, term := range features[j] {
				counts[term]++
			}
		}
	}
	r.OmittedChunks = r.OriginalChunks - r.SelectedChunks
	r.NoCandidateFits = len(chunks) > 0 && r.SelectedChunks == 0
	out := make([]types.Chunk, 0, r.SelectedChunks)
	for i, ch := range chunks {
		if selected[i] {
			// IDs retain source identity; links now describe the retained sequence.
			ch.PrevID, ch.NextID = "", ""
			out = append(out, ch)
		}
	}
	link(out)
	return out, r, nil
}

// uniformOrder repeatedly bisects source-order intervals. It uses no text/gold.
func uniformOrder(n int) []int {
	queue := [][2]int{{0, n}}
	out := make([]int, 0, n)
	for len(queue) > 0 {
		x := queue[0]
		queue = queue[1:]
		if x[0] >= x[1] {
			continue
		}
		mid := (x[0] + x[1]) / 2
		out = append(out, mid)
		queue = append(queue, [2]int{x[0], mid}, [2]int{mid + 1, x[1]})
	}
	return out
}

// precedingContext is a conservative adjacency heuristic, not semantic parsing.
// A table/list/code/diagram retains the immediately preceding text source group
// in its section. Repeated table splits share that context.
func precedingContext(chunks []types.Chunk, sections []types.Section) [][]int {
	sectionIDs := make([]int, len(chunks))
	for i := range sectionIDs {
		sectionIDs[i] = -1
	}
	for si, sec := range sections {
		if len(sec.Blocks) == 0 {
			continue
		}
		lo, hi := sec.Blocks[0].LineStart, sec.Blocks[len(sec.Blocks)-1].LineEnd
		if lo <= 0 || hi < lo {
			continue
		}
		start := sort.Search(len(chunks), func(i int) bool { return chunks[i].SourceLines[0] >= lo })
		for i := start; i < len(chunks) && chunks[i].SourceLines[0] <= hi; i++ {
			sectionIDs[i] = si
		}
	}
	deps := make([][]int, len(chunks))
	for i, ch := range chunks {
		switch ch.BlockType {
		case types.BlockTable, types.BlockList, types.BlockCode, types.BlockDiagram:
		default:
			continue
		}
		if i == 0 || sectionIDs[i] < 0 || sectionIDs[i] != sectionIDs[i-1] {
			continue
		}
		prev := chunks[i-1]
		if prev.BlockType == ch.BlockType && prev.SourceLines == ch.SourceLines {
			deps[i] = deps[i-1]
			continue
		}
		if prev.BlockType != types.BlockText {
			continue
		}
		j := i - 1
		for j > 0 && sectionIDs[j-1] == sectionIDs[i] && chunks[j-1].BlockType == types.BlockText && chunks[j-1].SourceLines[1] >= prev.SourceLines[0] {
			j--
		}
		for ; j < i; j++ {
			deps[i] = append(deps[i], j)
		}
	}
	return deps
}
