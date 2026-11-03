// Package chunker converts sections into chunks. Text runs are merged by
// explicit mode (none, size, similarity), without crossing non-text blocks or
// section boundaries. Candidate sizes include separators. Oversized atomic
// elements are reported, or rejected under OversizeError.
package chunker

import (
	"context"
	"fmt"
	"math"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/ETretyakov/kukri/scorer"
	"github.com/ETretyakov/kukri/types"
)

// Chunker orchestrates the full block→chunk pipeline.
type Chunker struct {
	cfg types.Config
	scr scorer.Scorer // required only by MergeSimilarity
}

// New creates a Chunker. The default mode merges by size with no Scorer.
// Configuration is validated by Validate or Chop.
func New(cfg types.Config, scr scorer.Scorer) *Chunker {
	if cfg.MergeMode == "" {
		cfg.MergeMode = types.MergeSize
	}
	if cfg.OversizePolicy == "" {
		cfg.OversizePolicy = types.OversizeKeep
	}
	if cfg.LengthFunc == nil {
		cfg.LengthFunc = countTokens
		cfg.LengthUnit = "words"
	}
	if cfg.EmbeddingBudget != nil {
		budget := *cfg.EmbeddingBudget
		cfg.EmbeddingBudget = &budget
	}
	if cfg.DocumentBudget != nil {
		budget := *cfg.DocumentBudget
		if budget.Strategy == "" {
			budget.Strategy = types.SelectTermsContext
		}
		cfg.DocumentBudget = &budget
	}
	return &Chunker{cfg: cfg, scr: scr}
}

// Validate checks configuration without processing any input.
func (c *Chunker) Validate() error {
	if err := c.validateDocumentBudget(); err != nil {
		return err
	}
	if c.cfg.MaxTokens < 0 || (c.cfg.MaxTokens == 0 && c.cfg.EmbeddingBudget == nil) {
		return fmt.Errorf("MaxTokens must be positive, or zero with EmbeddingBudget")
	}
	if c.cfg.LengthUnit == "" {
		return fmt.Errorf("LengthUnit is required with a custom LengthFunc")
	}
	if c.length("") != 0 {
		return fmt.Errorf("LengthFunc must return 0 for empty text")
	}
	if c.cfg.RepackText && c.cfg.MergeMode != types.MergeSize {
		return fmt.Errorf("RepackText requires MergeSize")
	}
	if b := c.cfg.EmbeddingBudget; b != nil {
		if b.MaxTokens <= 0 || b.Tokenizer == "" || b.CountTokens == nil {
			return fmt.Errorf("EmbeddingBudget requires positive MaxTokens, Tokenizer and CountTokens")
		}
		if _, err := c.embeddingLength(""); err != nil {
			return err
		}
	}
	if math.IsNaN(float64(c.cfg.Threshold)) || math.IsInf(float64(c.cfg.Threshold), 0) || c.cfg.Threshold < 0 || c.cfg.Threshold > 1 {
		return fmt.Errorf("Threshold must be finite and in [0,1]")
	}
	switch c.cfg.MergeMode {
	case types.MergeSize, types.MergeNone:
		if c.cfg.Threshold != 0 || c.scr != nil {
			return fmt.Errorf("Threshold and Scorer require MergeSimilarity; use MergeNone to disable merging")
		}
	case types.MergeSimilarity:
		if c.scr == nil {
			return fmt.Errorf("MergeSimilarity requires a Scorer")
		}
	default:
		return fmt.Errorf("unknown merge mode %q", c.cfg.MergeMode)
	}
	if c.cfg.OversizePolicy != types.OversizeKeep && c.cfg.OversizePolicy != types.OversizeError {
		return fmt.Errorf("unknown oversize policy %q", c.cfg.OversizePolicy)
	}
	return nil
}

func (c *Chunker) length(text string) int { return c.cfg.LengthFunc(text) }

// chopAll builds the complete representation before optional lossy selection.
func (c *Chunker) chopAll(ctx context.Context, sections []types.Section) ([]types.Chunk, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	all := make([]types.Chunk, 0)
	for _, sec := range sections {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for _, b := range sec.Blocks {
			if strings.TrimSpace(b.Content) != "" && c.length(b.Content) <= 0 {
				return nil, fmt.Errorf("length function returned a non-positive size for non-empty content")
			}
		}
		chunks, err := c.processSection(ctx, sec)
		if err != nil {
			return nil, fmt.Errorf("section %q: %w", sec.Title, err)
		}
		all = append(all, chunks...)
	}
	for _, chunk := range all {
		if strings.TrimSpace(chunk.Text) != "" && chunk.Size <= 0 {
			return nil, fmt.Errorf("length function returned a non-positive chunk size")
		}
		if c.cfg.OversizePolicy == types.OversizeError && chunk.EmbeddingOversized {
			return nil, fmt.Errorf("%s at lines %v exceeds embedding budget: %d > %d tokens (%s)", chunk.BlockType, chunk.SourceLines, chunk.EmbeddingTokens, c.cfg.EmbeddingBudget.MaxTokens, chunk.EmbeddingOversizeReason)
		}
		if c.cfg.OversizePolicy == types.OversizeError && chunk.Oversized {
			return nil, fmt.Errorf("%s at lines %v exceeds budget: %d > %d %s (%s)", chunk.BlockType, chunk.SourceLines, chunk.Size, c.cfg.MaxTokens, chunk.SizeUnit, chunk.OversizeReason)
		}
	}
	assignIDs(all)
	link(all)
	return all, nil
}

// embeddingLength never truncates input. Tokenizer errors abort the operation.
func (c *Chunker) embeddingLength(text string) (int, error) {
	if c.cfg.EmbeddingBudget == nil {
		return 0, nil
	}
	n, err := c.cfg.EmbeddingBudget.CountTokens(text)
	if err != nil {
		return 0, fmt.Errorf("embedding token counter: %w", err)
	}
	if n < 0 {
		return 0, fmt.Errorf("embedding token counter returned negative count %d", n)
	}
	return n, nil
}

func (c *Chunker) embedText(text string, sec types.Section, kind ...types.BlockType) string {
	if c.cfg.CompactTableEmbedding && len(kind) > 0 && kind[0] == types.BlockTable {
		text = compactTableEmbedding(text)
	}
	if c.cfg.PrependSectionPath && len(sec.Path) > 0 {
		return strings.Join(sec.Path, " > ") + "\n\n" + text
	}
	return text
}

func (c *Chunker) fits(text string, sec types.Section, kind ...types.BlockType) (bool, error) {
	n := c.length(text)
	if strings.TrimSpace(text) != "" && n <= 0 {
		return false, fmt.Errorf("length function returned a non-positive candidate size")
	}
	if c.cfg.MaxTokens > 0 && n > c.cfg.MaxTokens {
		return false, nil
	}
	e, err := c.embeddingLength(c.embedText(text, sec, kind...))
	if err != nil {
		return false, err
	}
	return c.cfg.EmbeddingBudget == nil || e <= c.cfg.EmbeddingBudget.MaxTokens, nil
}

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
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if block.Type == types.BlockText {
			if strings.TrimSpace(block.Content) != "" {
				textRun = append(textRun, block)
			}
			continue
		}
		if err := flush(); err != nil {
			return nil, err
		}
		var produced []types.Chunk
		var err error
		switch block.Type {
		case types.BlockTable:
			produced, err = c.chunkTable(ctx, block, sec)
		case types.BlockList:
			produced, err = c.chunkList(ctx, block, sec)
		case types.BlockImage, types.BlockCode, types.BlockDiagram:
			var ch types.Chunk
			ch, err = c.makeChunk(block.Content, block, sec)
			produced = []types.Chunk{ch}
		default:
			return nil, fmt.Errorf("unsupported content block type %q", block.Type)
		}
		if err != nil {
			return nil, err
		}
		chunks = append(chunks, produced...)
	}
	if err := flush(); err != nil {
		return nil, err
	}
	return chunks, nil
}

// mergeTextBlocks preserves historical paragraph-first grouping unless
// RepackText is enabled. Similarity scores always describe original paragraphs.
func (c *Chunker) mergeTextBlocks(ctx context.Context, blocks []types.Block, sec types.Section) ([]types.Chunk, error) {
	if len(blocks) == 0 {
		return nil, nil
	}
	if c.cfg.RepackText {
		return c.repackText(ctx, blocks, sec)
	}
	var sims []float32
	if c.cfg.MergeMode == types.MergeSimilarity && len(blocks) >= 2 {
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
		sims = make([]float32, len(blocks)-1)
	}
	if len(sims) != len(blocks)-1 {
		return nil, fmt.Errorf("scorer returned %d scores for %d blocks; want %d", len(sims), len(blocks), len(blocks)-1)
	}
	for _, sim := range sims {
		if math.IsNaN(float64(sim)) || math.IsInf(float64(sim), 0) || sim < -1 || sim > 1 {
			return nil, fmt.Errorf("scorer returned invalid similarity %v; expected finite value in [-1,1]", sim)
		}
	}
	var groups [][]types.Block
	cur := []types.Block{blocks[0]}
	curText := blocks[0].Content
	for i := 1; i < len(blocks); i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		candidate := curText + "\n\n" + blocks[i].Content
		allowed := c.cfg.MergeMode == types.MergeSize || (c.cfg.MergeMode == types.MergeSimilarity && sims[i-1] >= c.cfg.Threshold)
		fit, err := c.fits(candidate, sec)
		if err != nil {
			return nil, err
		}
		if allowed && fit {
			cur = append(cur, blocks[i])
			curText = candidate
		} else {
			groups = append(groups, cur)
			cur = []types.Block{blocks[i]}
			curText = blocks[i].Content
		}
	}
	groups = append(groups, cur)
	var out []types.Chunk
	for _, group := range groups {
		fit, err := c.fits(group[0].Content, sec)
		if err != nil {
			return nil, err
		}
		if len(group) == 1 && !fit {
			parts, err := c.splitByTokens(ctx, splitSentences(group[0].Content), group[0], sec)
			if err != nil {
				return nil, err
			}
			out = append(out, parts...)
			continue
		}
		parts := make([]string, len(group))
		for i, b := range group {
			parts[i] = b.Content
		}
		span := group[0]
		span.LineEnd = group[len(group)-1].LineEnd
		ch, err := c.makeChunk(strings.Join(parts, "\n\n"), span, sec)
		if err != nil {
			return nil, err
		}
		out = append(out, ch)
	}
	return out, nil
}

// repackText first splits each original paragraph, then greedily packs the
// fragments. Paragraph separators are retained; no structural boundary is crossed.
func (c *Chunker) repackText(ctx context.Context, blocks []types.Block, sec types.Section) ([]types.Chunk, error) {
	var out []types.Chunk
	var pending *types.Chunk
	lastParagraph := -1
	for i, block := range blocks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		fit, err := c.fits(block.Content, sec)
		if err != nil {
			return nil, err
		}
		var parts []types.Chunk
		if fit {
			ch, err := c.makeChunk(block.Content, block, sec)
			if err != nil {
				return nil, err
			}
			parts = []types.Chunk{ch}
		} else {
			parts, err = c.splitByTokens(ctx, splitSentences(block.Content), block, sec)
			if err != nil {
				return nil, err
			}
		}
		for _, part := range parts {
			if pending != nil {
				sep := " "
				if lastParagraph != i {
					sep = "\n\n"
				}
				candidate := pending.Text + sep + part.Text
				fit, err := c.fits(candidate, sec)
				if err != nil {
					return nil, err
				}
				if fit {
					span := block
					span.LineStart = pending.SourceLines[0]
					ch, err := c.makeChunk(candidate, span, sec)
					if err != nil {
						return nil, err
					}
					pending = &ch
					lastParagraph = i
					continue
				}
				out = append(out, *pending)
			}
			ch := part
			pending = &ch
			lastParagraph = i
		}
	}
	if pending != nil {
		out = append(out, *pending)
	}
	return out, nil
}

func (c *Chunker) splitByTokens(ctx context.Context, sentences []string, block types.Block, sec types.Section) ([]types.Chunk, error) {
	// If metadata alone exceeds the limit, token-based splitting cannot repair
	// it. Retain ordinary body-budget splitting instead of emitting one chunk per
	// word. makeChunk still measures and reports the full input overflow.
	splitter := c
	if budget := c.cfg.EmbeddingBudget; budget != nil {
		n, err := c.embeddingLength(c.embedText("", sec))
		if err != nil {
			return nil, err
		}
		if n > budget.MaxTokens {
			bodyOnly := *c
			bodyOnly.cfg.EmbeddingBudget = nil
			splitter = &bodyOnly
		}
	}
	var chunks []types.Chunk
	cur := ""
	flush := func() error {
		if cur != "" {
			ch, err := c.makeChunk(cur, block, sec)
			if err != nil {
				return err
			}
			chunks = append(chunks, ch)
			cur = ""
		}
		return nil
	}
	appendPart := func(part string) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		candidate := part
		if cur != "" {
			candidate = cur + " " + part
		}
		fit, err := splitter.fits(candidate, sec)
		if err != nil {
			return err
		}
		if cur != "" && !fit {
			if err := flush(); err != nil {
				return err
			}
			candidate = part
		}
		cur = candidate
		return nil
	}
	for _, sentence := range sentences {
		fit, err := splitter.fits(sentence, sec)
		if err != nil {
			return nil, err
		}
		if fit {
			if err := appendPart(sentence); err != nil {
				return nil, err
			}
			continue
		}
		if err := flush(); err != nil {
			return nil, err
		}
		for _, word := range strings.Fields(sentence) {
			if err := appendPart(word); err != nil {
				return nil, err
			}
		}
	}
	if err := flush(); err != nil {
		return nil, err
	}
	return chunks, nil
}

// packItems preserves indivisible rows/items. Every table fragment repeats the
// same header, which is included in both budget checks.
func (c *Chunker) packItems(ctx context.Context, items []string, header string, block types.Block, sec types.Section) ([]types.Chunk, error) {
	var chunks []types.Chunk
	var batch []string
	flush := func() error {
		if len(batch) == 0 {
			return nil
		}
		ch, err := c.makeChunk(header+strings.Join(batch, "\n"), block, sec)
		if err != nil {
			return err
		}
		chunks = append(chunks, ch)
		batch = nil
		return nil
	}
	for _, item := range items {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		candidate := header + strings.Join(batch, "\n")
		if len(batch) > 0 {
			candidate += "\n"
		}
		candidate += item
		fit, err := c.fits(candidate, sec, block.Type)
		if err != nil {
			return nil, err
		}
		if len(batch) > 0 && !fit {
			if err := flush(); err != nil {
				return nil, err
			}
		}
		batch = append(batch, item)
	}
	if err := flush(); err != nil {
		return nil, err
	}
	return chunks, nil
}

func (c *Chunker) chunkTable(ctx context.Context, block types.Block, sec types.Section) ([]types.Chunk, error) {
	fit, err := c.fits(block.Content, sec, block.Type)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(block.Content, "\n")
	if fit || len(lines) < 3 {
		ch, err := c.makeChunk(block.Content, block, sec)
		return []types.Chunk{ch}, err
	}
	return c.packItems(ctx, lines[2:], lines[0]+"\n"+lines[1]+"\n", block, sec)
}

func (c *Chunker) chunkList(ctx context.Context, block types.Block, sec types.Section) ([]types.Chunk, error) {
	fit, err := c.fits(block.Content, sec, block.Type)
	if err != nil {
		return nil, err
	}
	if fit {
		ch, err := c.makeChunk(block.Content, block, sec)
		return []types.Chunk{ch}, err
	}
	var items []string
	var cur []string
	for _, line := range strings.Split(block.Content, "\n") {
		if isListItemStart(line) && len(cur) > 0 {
			items = append(items, strings.Join(cur, "\n"))
			cur = nil
		}
		cur = append(cur, line)
	}
	if len(cur) > 0 {
		items = append(items, strings.Join(cur, "\n"))
	}
	return c.packItems(ctx, items, "", block, sec)
}

func (c *Chunker) makeChunk(text string, block types.Block, sec types.Section) (types.Chunk, error) {
	path := sec.Path
	if path == nil {
		path = []string{}
	}
	embedText := c.embedText(text, sec, block.Type)
	size := c.length(text)
	e, err := c.embeddingLength(embedText)
	if err != nil {
		return types.Chunk{}, err
	}
	reason := "atomic_block"
	switch block.Type {
	case types.BlockText:
		reason = "indivisible_word"
	case types.BlockTable:
		reason = "table_header_or_row"
	case types.BlockList:
		reason = "list_item"
	}
	ch := types.Chunk{
		Size: size, SizeUnit: c.cfg.LengthUnit, Oversized: c.cfg.MaxTokens > 0 && size > c.cfg.MaxTokens,
		Text: text, EmbedText: embedText, Section: sec.Title, SectionPath: path,
		BlockType: block.Type, SourceLines: [2]int{block.LineStart, block.LineEnd},
		TokenCount: countTokens(text), Source: c.cfg.Source,
	}
	if ch.Oversized {
		ch.OversizeReason = reason
	}
	if b := c.cfg.EmbeddingBudget; b != nil {
		ch.EmbeddingTokens = e
		ch.EmbeddingTokenizer = b.Tokenizer
		ch.EmbeddingOversized = e > b.MaxTokens
		if ch.EmbeddingOversized {
			ch.EmbeddingOversizeReason = reason
			n, err := c.embeddingLength(c.embedText("", sec))
			if err != nil {
				return types.Chunk{}, err
			}
			if n > b.MaxTokens {
				ch.EmbeddingOversizeReason = "embedding_prefix_or_special_tokens"
			}
		}
	}
	return ch, nil
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
