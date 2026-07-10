// Package chunker converts []Section into []Chunk.
//
// Adjacent text blocks within a section are merged greedily: two blocks are
// combined when scorer.Score(block_i, block_{i+1}) >= Config.Threshold AND
// the combined token count stays within Config.MaxTokens.
//
// When no Scorer is provided (nil), block similarity is always 0.  Setting
// Threshold=0 therefore merges every adjacent block up to MaxTokens — the
// recommended default for IAEA-style documents where structure is already
// encoded in section boundaries.
//
// Non-text blocks are never merged across their boundaries:
//   - table   — emitted as one chunk; split row-by-row if it exceeds MaxTokens
//               (header repeated in every part).
//   - list    — emitted as one chunk; split on top-level item boundaries if needed.
//   - code / diagram / image — always one chunk, regardless of size.
//
// Every chunk receives a slug-based ID "<section-slug>_<counter>" and is
// doubly linked via PrevID / NextID for context-window retrieval.
package chunker

import (
	"context"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/ETretyakov/kukri/scorer"
	"github.com/ETretyakov/kukri/types"
)

// Chunker orchestrates the full block→chunk pipeline.
type Chunker struct {
	cfg types.Config
	scr scorer.Scorer // nil → similarity always 0
}

// New creates a Chunker. scr may be nil; blocks are then merged purely by
// token count (similarity is treated as 0 for every adjacent pair).
func New(cfg types.Config, scr scorer.Scorer) *Chunker {
	return &Chunker{cfg: cfg, scr: scr}
}

// Chop converts a flat list of sections into linked, ID-bearing chunks.
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

// processSection merges text blocks within the section, emitting non-text
// blocks independently.
func (c *Chunker) processSection(ctx context.Context, sec types.Section) ([]types.Chunk, error) {
	var chunks []types.Chunk
	var textRun []types.Block

	flush := func() error {
		if len(textRun) == 0 {
			return nil
		}
		produced, err := c.mergeTextBlocks(ctx, textRun, sec)
		if err != nil {
			return err
		}
		chunks = append(chunks, produced...)
		textRun = nil
		return nil
	}

	for _, block := range sec.Blocks {
		switch block.Type {
		case types.BlockText:
			textRun = append(textRun, block)
		case types.BlockTable:
			if err := flush(); err != nil {
				return nil, err
			}
			chunks = append(chunks, c.chunkTable(block, sec)...)
		case types.BlockList:
			if err := flush(); err != nil {
				return nil, err
			}
			chunks = append(chunks, c.chunkList(block, sec)...)
		case types.BlockImage, types.BlockCode, types.BlockDiagram:
			if err := flush(); err != nil {
				return nil, err
			}
			chunks = append(chunks, c.makeChunk(block.Content, block, sec))
		}
	}
	return chunks, flush()
}

// mergeTextBlocks scores adjacent blocks pairwise and merges greedily.
// A single block that exceeds MaxTokens is split at sentence boundaries.
func (c *Chunker) mergeTextBlocks(ctx context.Context, blocks []types.Block, sec types.Section) ([]types.Chunk, error) {
	if len(blocks) == 0 {
		return nil, nil
	}

	var sims []float32
	if c.scr != nil && len(blocks) >= 2 {
		contents := make([]string, len(blocks))
		for i, b := range blocks {
			contents[i] = b.Content
		}
		var err error
		sims, err = c.scr.Score(ctx, contents)
		if err != nil {
			return nil, err
		}
	} else {
		sims = make([]float32, len(blocks)-1) // all zeros
	}

	// Greedy merge: extend current group while sim >= threshold and
	// combined tokens <= MaxTokens.
	var groups [][]types.Block
	cur := []types.Block{blocks[0]}
	curTokens := countTokens(blocks[0].Content)

	for i := 1; i < len(blocks); i++ {
		nextTokens := countTokens(blocks[i].Content)
		if sims[i-1] >= c.cfg.Threshold && curTokens+nextTokens <= c.cfg.MaxTokens {
			cur = append(cur, blocks[i])
			curTokens += nextTokens
		} else {
			groups = append(groups, cur)
			cur = []types.Block{blocks[i]}
			curTokens = nextTokens
		}
	}
	groups = append(groups, cur)

	var out []types.Chunk
	for _, group := range groups {
		if len(group) == 1 && countTokens(group[0].Content) > c.cfg.MaxTokens {
			// Single oversized block: split at sentence boundaries.
			out = append(out, c.splitByTokens(splitSentences(group[0].Content), group[0], sec)...)
			continue
		}
		if len(group) == 1 {
			out = append(out, c.makeChunk(group[0].Content, group[0], sec))
			continue
		}
		parts := make([]string, len(group))
		for i, b := range group {
			parts[i] = b.Content
		}
		span := group[0]
		span.LineEnd = group[len(group)-1].LineEnd
		out = append(out, c.makeChunk(strings.Join(parts, "\n\n"), span, sec))
	}
	return out, nil
}

// splitByTokens groups sentences into chunks bounded by MaxTokens.
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

// chunkTable returns the table as one chunk, or splits row-by-row repeating
// the header in every part.
func (c *Chunker) chunkTable(block types.Block, sec types.Section) []types.Chunk {
	if countTokens(block.Content) <= c.cfg.MaxTokens {
		return []types.Chunk{c.makeChunk(block.Content, block, sec)}
	}

	lines := strings.Split(block.Content, "\n")
	if len(lines) < 3 {
		return []types.Chunk{c.makeChunk(block.Content, block, sec)}
	}

	header := lines[0]
	separator := lines[1]
	dataRows := lines[2:]
	headerTokens := countTokens(header) + countTokens(separator)

	var chunks []types.Chunk
	var batch []string
	batchTokens := headerTokens

	for _, row := range dataRows {
		rt := countTokens(row)
		if len(batch) > 0 && batchTokens+rt > c.cfg.MaxTokens {
			chunks = append(chunks, c.makeChunk(header+"\n"+separator+"\n"+strings.Join(batch, "\n"), block, sec))
			batch = nil
			batchTokens = headerTokens
		}
		batch = append(batch, row)
		batchTokens += rt
	}
	if len(batch) > 0 {
		chunks = append(chunks, c.makeChunk(header+"\n"+separator+"\n"+strings.Join(batch, "\n"), block, sec))
	}
	return chunks
}

// chunkList returns the list as one chunk, or splits on top-level item
// boundaries.
func (c *Chunker) chunkList(block types.Block, sec types.Section) []types.Chunk {
	if countTokens(block.Content) <= c.cfg.MaxTokens {
		return []types.Chunk{c.makeChunk(block.Content, block, sec)}
	}

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

func (c *Chunker) makeChunk(text string, block types.Block, sec types.Section) types.Chunk {
	path := sec.Path
	if path == nil {
		path = []string{}
	}
	embedText := text
	if c.cfg.PrependSectionPath && len(path) > 0 {
		embedText = strings.Join(path, " > ") + "\n\n" + text
	}
	return types.Chunk{
		Text:        text,
		EmbedText:   embedText,
		Section:     sec.Title,
		SectionPath: path,
		BlockType:   block.Type,
		SourceLines: [2]int{block.LineStart, block.LineEnd},
		TokenCount:  countTokens(text),
		Source:      c.cfg.Source,
	}
}

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

var abbrevSet = map[string]bool{
	"dr": true, "mr": true, "mrs": true, "ms": true, "prof": true, "sr": true, "jr": true,
	"fig": true, "figs": true, "no": true, "nos": true, "ref": true, "refs": true,
	"sec": true, "secs": true, "vol": true, "vols": true, "pp": true, "p": true,
	"eq": true, "eqs": true, "tab": true, "app": true, "dept": true,
	"etc": true, "vs": true, "cf": true, "al": true, "viz": true,
	"jan": true, "feb": true, "mar": true, "apr": true, "jun": true, "jul": true,
	"aug": true, "sep": true, "oct": true, "nov": true, "dec": true,
	"est": true, "govt": true, "corp": true, "ltd": true, "inc": true,
	"approx": true, "intl": true, "natl": true, "univ": true, "assoc": true,
}

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
			break
		}
		next := words[i+1]
		if !wordStartsUpperOrQuote(next) {
			continue
		}
		bare := strings.ToLower(strings.TrimRight(word, ".!?"))
		if abbrevSet[bare] {
			continue
		}
		if strings.Contains(bare, ".") {
			continue
		}
		if len(bare) == 1 {
			continue
		}
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

func wordStartsUpperOrQuote(word string) bool {
	if word == "" {
		return false
	}
	r, _ := utf8.DecodeRuneInString(word)
	return unicode.IsUpper(r) || r == '"' || r == '\''
}

func countTokens(text string) int {
	return len(strings.Fields(text))
}

func isListItemStart(line string) bool {
	if strings.HasPrefix(line, "- ") || strings.HasPrefix(line, "* ") {
		return true
	}
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
