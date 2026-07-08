// Package chunker converts []Section into []Chunk using a hybrid strategy
// that combines structural awareness of Markdown block types with
// sentence-level semantic similarity.
//
// # Splitting strategy by block type
//
//   - text    — split into sentences, then greedily merge adjacent sentences
//     while cosine similarity ≥ Config.Threshold AND combined token count
//     stays within Config.MaxTokens.  When no Embedder is provided the
//     semantic step is skipped and sentences are merged by token count only.
//   - table   — emitted as one chunk if it fits; otherwise split row-by-row
//     with the header row (and separator) repeated in every part so each
//     chunk is a self-contained table fragment.
//   - list    — emitted as one chunk if it fits; otherwise split on
//     top-level item boundaries.
//   - code / diagram / image — always emitted as a single chunk regardless
//     of size, to avoid breaking executable or visual content.
//
// # Chunk identity
//
// After splitting, every chunk receives a slug-based ID of the form
// "<section-slug>_<counter>" (e.g. "introduction_0", "methods_3").
// Adjacent chunks are doubly linked via PrevID / NextID so that a RAG
// pipeline can expand a retrieved chunk with its neighbours for context.
package chunker

import (
	"context"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/ETretyakov/kukri/embedder"
	"github.com/ETretyakov/kukri/types"
)

// Chunker orchestrates the full block→chunk pipeline.
type Chunker struct {
	cfg types.Config
	emb *embedder.Embedder // nil → fall back to token-count splitting
}

// New creates a Chunker. emb may be nil; in that case semantic grouping is
// skipped and text blocks are split purely by token count.
func New(cfg types.Config, emb *embedder.Embedder) *Chunker {
	return &Chunker{cfg: cfg, emb: emb}
}

// Chop converts a flat list of sections into linked, ID-bearing chunks.
// It is the single entry point for the pipeline after segmentation.
func (c *Chunker) Chop(ctx context.Context, sections []types.Section) ([]types.Chunk, error) {
	var all []types.Chunk
	for _, sec := range sections {
		chunks, err := c.processSection(ctx, sec)
		if err != nil {
			return nil, fmt.Errorf("section %q: %w", sec.Title, err)
		}
		all = append(all, chunks...)
	}
	assignIDs(all)
	link(all)
	return all, nil
}

// processSection turns one section's blocks into chunks (IDs not yet assigned).
func (c *Chunker) processSection(ctx context.Context, sec types.Section) ([]types.Chunk, error) {
	var chunks []types.Chunk

	for _, block := range sec.Blocks {
		var (
			produced []types.Chunk
			err      error
		)

		switch block.Type {
		case types.BlockText:
			produced, err = c.chunkText(ctx, block, sec)
		case types.BlockTable:
			produced = c.chunkTable(block, sec)
		case types.BlockList:
			produced = c.chunkList(block, sec)
		case types.BlockImage, types.BlockCode, types.BlockDiagram:
			// These block types are always kept whole — never split.
			produced = []types.Chunk{c.makeChunk(block.Content, block, sec)}
		default:
			continue
		}

		if err != nil {
			return nil, err
		}
		chunks = append(chunks, produced...)
	}

	return chunks, nil
}

// chunkText splits a text block into semantically coherent chunks.
//
// Algorithm (greedy merge):
//  1. Split the block into sentences.
//  2. If the whole block fits within MaxTokens, emit it as one chunk.
//  3. Otherwise embed every sentence with the sentence-transformer model.
//  4. Walk sentences left-to-right; extend the current group as long as
//     consecutive cosine similarity ≥ Threshold AND adding the next sentence
//     would not exceed MaxTokens.  When either condition fails, close the
//     current group and start a new one.
//
// When no Embedder is configured (emb == nil) step 3 is skipped and
// sentences are merged by token count alone (step 4 without the similarity
// gate).
func (c *Chunker) chunkText(ctx context.Context, block types.Block, sec types.Section) ([]types.Chunk, error) {
	sentences := splitSentences(block.Content)
	if len(sentences) == 0 {
		return nil, nil
	}

	// Fast path: the whole block fits in a single chunk.
	if countTokens(block.Content) <= c.cfg.MaxTokens {
		return []types.Chunk{c.makeChunk(block.Content, block, sec)}, nil
	}

	// No embedder → token-count-only splitting.
	if c.emb == nil {
		return c.splitByTokens(sentences, block, sec), nil
	}

	embs, err := c.emb.Embed(ctx, sentences)
	if err != nil {
		return nil, err
	}

	// Greedy semantic merge.
	var groups [][]string
	cur := []string{sentences[0]}
	curTokens := countTokens(sentences[0])

	for i := 1; i < len(sentences); i++ {
		sim := embedder.CosineSimilarity(embs[i-1], embs[i])
		nextTokens := countTokens(sentences[i])

		if sim >= c.cfg.Threshold && curTokens+nextTokens <= c.cfg.MaxTokens {
			// Similarity is high and we still have budget — keep going.
			cur = append(cur, sentences[i])
			curTokens += nextTokens
		} else {
			// Topic shift or budget exhausted — close and start fresh.
			groups = append(groups, cur)
			cur = []string{sentences[i]}
			curTokens = nextTokens
		}
	}
	groups = append(groups, cur)

	chunks := make([]types.Chunk, 0, len(groups))
	for _, g := range groups {
		chunks = append(chunks, c.makeChunk(strings.Join(g, " "), block, sec))
	}
	return chunks, nil
}

// splitByTokens is the no-embedder fallback: groups sentences by token count only.
func (c *Chunker) splitByTokens(sentences []string, block types.Block, sec types.Section) []types.Chunk {
	var chunks []types.Chunk
	var cur []string
	curTokens := 0

	for _, s := range sentences {
		t := countTokens(s)
		if len(cur) > 0 && curTokens+t > c.cfg.MaxTokens {
			chunks = append(chunks, c.makeChunk(strings.Join(cur, " "), block, sec))
			cur = nil
			curTokens = 0
		}
		cur = append(cur, s)
		curTokens += t
	}
	if len(cur) > 0 {
		chunks = append(chunks, c.makeChunk(strings.Join(cur, " "), block, sec))
	}
	return chunks
}

// chunkTable returns the table as one chunk, or splits it row-by-row while
// repeating the header row (and GFM separator) in every part so that each
// chunk is a valid, self-contained table fragment.
func (c *Chunker) chunkTable(block types.Block, sec types.Section) []types.Chunk {
	if countTokens(block.Content) <= c.cfg.MaxTokens {
		return []types.Chunk{c.makeChunk(block.Content, block, sec)}
	}

	lines := strings.Split(block.Content, "\n")
	if len(lines) < 3 {
		// Degenerate table — emit as-is.
		return []types.Chunk{c.makeChunk(block.Content, block, sec)}
	}

	header := lines[0]       // "| Col A | Col B |"
	separator := lines[1]    // "|-------|-------|"
	dataRows := lines[2:]
	headerTokens := countTokens(header) + countTokens(separator)

	var chunks []types.Chunk
	var batch []string
	batchTokens := headerTokens

	for _, row := range dataRows {
		rt := countTokens(row)
		if len(batch) > 0 && batchTokens+rt > c.cfg.MaxTokens {
			text := header + "\n" + separator + "\n" + strings.Join(batch, "\n")
			chunks = append(chunks, c.makeChunk(text, block, sec))
			batch = nil
			batchTokens = headerTokens
		}
		batch = append(batch, row)
		batchTokens += rt
	}
	if len(batch) > 0 {
		text := header + "\n" + separator + "\n" + strings.Join(batch, "\n")
		chunks = append(chunks, c.makeChunk(text, block, sec))
	}
	return chunks
}

// chunkList returns the list as one chunk, or splits it on top-level item
// boundaries when the combined token count exceeds MaxTokens.
func (c *Chunker) chunkList(block types.Block, sec types.Section) []types.Chunk {
	if countTokens(block.Content) <= c.cfg.MaxTokens {
		return []types.Chunk{c.makeChunk(block.Content, block, sec)}
	}

	// Group lines into top-level items (each starts with isListItemStart).
	lines := strings.Split(block.Content, "\n")
	var items [][]string
	var cur []string

	for _, line := range lines {
		if isListItemStart(line) && len(cur) > 0 {
			items = append(items, cur)
			cur = nil
		}
		cur = append(cur, line)
	}
	if len(cur) > 0 {
		items = append(items, cur)
	}

	var chunks []types.Chunk
	var batch []string
	batchTokens := 0

	for _, item := range items {
		text := strings.Join(item, "\n")
		t := countTokens(text)
		if len(batch) > 0 && batchTokens+t > c.cfg.MaxTokens {
			chunks = append(chunks, c.makeChunk(strings.Join(batch, "\n"), block, sec))
			batch = nil
			batchTokens = 0
		}
		batch = append(batch, text)
		batchTokens += t
	}
	if len(batch) > 0 {
		chunks = append(chunks, c.makeChunk(strings.Join(batch, "\n"), block, sec))
	}
	return chunks
}

// makeChunk constructs a Chunk without an ID (IDs are assigned in bulk by
// assignIDs after all sections have been processed).
func (c *Chunker) makeChunk(text string, block types.Block, sec types.Section) types.Chunk {
	path := sec.Path
	if path == nil {
		path = []string{}
	}
	return types.Chunk{
		Text:        text,
		Section:     sec.Title,
		SectionPath: path,
		BlockType:   block.Type,
		SourceLines: [2]int{block.LineStart, block.LineEnd},
		TokenCount:  countTokens(text),
		Source:      c.cfg.Source,
	}
}

// assignIDs gives every chunk a human-readable unique ID of the form
// "<section-slug>_<counter>", where the counter resets per slug.
// Chunks with an empty section title use the slug "doc".
func assignIDs(chunks []types.Chunk) {
	counts := make(map[string]int)
	for i := range chunks {
		slug := slugify(chunks[i].Section)
		if slug == "" {
			slug = "doc"
		}
		chunks[i].ID = fmt.Sprintf("%s_%d", slug, counts[slug])
		counts[slug]++
	}
}

// link sets PrevID and NextID on every chunk to form a doubly-linked list.
// This lets downstream retrieval expand a matching chunk with its neighbours.
func link(chunks []types.Chunk) {
	for i := range chunks {
		if i > 0 {
			chunks[i].PrevID = chunks[i-1].ID
		}
		if i < len(chunks)-1 {
			chunks[i].NextID = chunks[i+1].ID
		}
	}
}

// ── helpers ──────────────────────────────────────────────────────────────────

// abbrevSet contains lowercase bare words that should never trigger a sentence
// boundary when followed by a period and an uppercase word.
// Technical documents (IAEA, scientific papers) use many of these.
var abbrevSet = map[string]bool{
	// Titles
	"dr": true, "mr": true, "mrs": true, "ms": true, "prof": true, "sr": true, "jr": true,
	// Academic / document structure
	"fig": true, "figs": true, "no": true, "nos": true, "ref": true, "refs": true,
	"sec": true, "secs": true, "vol": true, "vols": true, "pp": true, "p": true,
	"eq": true, "eqs": true, "tab": true, "app": true, "dept": true,
	// Latin abbreviations
	"etc": true, "vs": true, "cf": true, "al": true, "viz": true,
	// Months
	"jan": true, "feb": true, "mar": true, "apr": true, "jun": true, "jul": true,
	"aug": true, "sep": true, "oct": true, "nov": true, "dec": true,
	// Organisations / misc
	"est": true, "govt": true, "corp": true, "ltd": true, "inc": true,
	"approx": true, "intl": true, "natl": true, "univ": true, "assoc": true,
}

// splitSentences splits text into sentences using a word-by-word scan that
// is aware of common abbreviations and avoids false boundaries at:
//   - known abbreviations  ("Fig. 1", "Dr. Smith", "No. 3")
//   - words with internal periods ("U.S.A.", "e.g.", "i.e.")
//   - single-letter tokens (initials: "A. Smith" in a name list)
//   - numeric context ("IAEA 1.5 GHz" — already avoided because "1.5" has no space)
//
// A boundary is inserted when a word ends with [.!?], the next word starts
// with an uppercase letter or quote, and none of the guard conditions above
// apply.
func splitSentences(text string) []string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return nil
	}

	var sentences []string
	start := 0

	for i, word := range words {
		if !wordEndsPunct(word) {
			continue
		}
		if i == len(words)-1 {
			// Last word — close the final sentence after the loop.
			break
		}

		next := words[i+1]
		if !wordStartsUpperOrQuote(next) {
			continue
		}

		bare := strings.ToLower(strings.TrimRight(word, ".!?"))

		// Guard 1: known abbreviation.
		if abbrevSet[bare] {
			continue
		}
		// Guard 2: word contains internal periods → "U.S.A.", "e.g.", "i.e."
		if strings.Contains(bare, ".") {
			continue
		}
		// Guard 3: single-letter tokens → initials or section labels.
		if len(bare) == 1 {
			continue
		}
		// Guard 4: next word starts with a digit → "Fig. 1", "No. 42".
		if r, _ := utf8.DecodeRuneInString(next); unicode.IsDigit(r) {
			continue
		}

		sentences = append(sentences, strings.Join(words[start:i+1], " "))
		start = i + 1
	}

	if start < len(words) {
		sentences = append(sentences, strings.Join(words[start:], " "))
	}
	return sentences
}

// wordEndsPunct reports whether word ends with sentence-closing punctuation.
func wordEndsPunct(word string) bool {
	if word == "" {
		return false
	}
	switch word[len(word)-1] {
	case '.', '!', '?':
		return true
	}
	return false
}

// wordStartsUpperOrQuote reports whether word begins with an uppercase letter
// or an opening quote — consistent with the original regex boundary check.
func wordStartsUpperOrQuote(word string) bool {
	if word == "" {
		return false
	}
	r, _ := utf8.DecodeRuneInString(word)
	return unicode.IsUpper(r) || r == '"' || r == '\''
}

// countTokens approximates token count as the number of whitespace-delimited
// words.  This is fast and good enough for soft-limit enforcement; replace
// with a proper tokenizer if sub-word accuracy is required.
func countTokens(text string) int {
	return len(strings.Fields(text))
}

// isListItemStart reports whether line begins a top-level list item.
// Unlike parser.isListLine, this does NOT trim leading whitespace: an indented
// "  - sub-item" is part of the parent item and must not trigger a new boundary.
func isListItemStart(line string) bool {
	if strings.HasPrefix(line, "- ") || strings.HasPrefix(line, "* ") {
		return true
	}
	// Ordered list starting at column 0: "1. " "12. " etc.
	for j, ch := range line {
		if ch >= '0' && ch <= '9' {
			continue
		}
		if ch == '.' && j > 0 && j+1 < len(line) && line[j+1] == ' ' {
			return true
		}
		break
	}
	return false
}

// slugify converts an arbitrary string into a lowercase, underscore-separated
// identifier suitable for use in chunk IDs.
// "1.2 Methods & Results" → "1_2_methods_results"
func slugify(s string) string {
	s = strings.ToLower(s)
	var b strings.Builder
	for _, r := range s {
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			b.WriteRune(r)
		case unicode.IsSpace(r) || r == '_' || r == '-' || r == '.' || r == ':' || r == '/':
			b.WriteRune('_')
		}
	}
	result := b.String()
	for strings.Contains(result, "__") {
		result = strings.ReplaceAll(result, "__", "_")
	}
	return strings.Trim(result, "_")
}
