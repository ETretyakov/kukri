package chunker

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/ETretyakov/kukri/types"
)

// helpers

func textSection(title string, content string) types.Section {
	return types.Section{
		Title: title,
		Level: 1,
		Path:  []string{title},
		Blocks: []types.Block{
			{Type: types.BlockText, Content: content, LineStart: 1, LineEnd: 1},
		},
	}
}

func newChunker(maxTokens int) *Chunker {
	return New(types.Config{MaxTokens: maxTokens, MergeMode: types.MergeNone}, nil)
}

// ── countTokens ──────────────────────────────────────────────────────────────

func TestCountTokens(t *testing.T) {
	cases := []struct {
		in   string
		want int
	}{
		{"hello world", 2},
		{"", 0},
		{"  spaces  ", 1},
		{"one two three four five", 5},
		{"single", 1},
	}
	for _, tc := range cases {
		if got := countTokens(tc.in); got != tc.want {
			t.Errorf("countTokens(%q) = %d, want %d", tc.in, got, tc.want)
		}
	}
}

// ── slugify ───────────────────────────────────────────────────────────────────

func TestSlugify(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"Introduction", "introduction"},
		{"1.2 Methods & Results", "1_2_methods_results"},
		{"", ""},
		{"Hello   World", "hello_world"},
		{"__leading__trailing__", "leading_trailing"},
		{"UPPER CASE", "upper_case"},
		{"café résumé", "café_résumé"}, // unicode letters preserved
	}
	for _, tc := range cases {
		got := slugify(tc.in)
		if got != tc.want {
			t.Errorf("slugify(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// ── splitSentences ───────────────────────────────────────────────────────────

func TestSplitSentences(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{
			"Hello world. This is a test.",
			[]string{"Hello world.", "This is a test."},
		},
		{
			"No boundary here",
			[]string{"No boundary here"},
		},
		{
			"First! Second? Third.",
			[]string{"First!", "Second?", "Third."},
		},
		{
			// lowercase after period — no split
			"Abbrev. like this does not split mid-sentence.",
			[]string{"Abbrev. like this does not split mid-sentence."},
		},
		{
			// known abbreviation: Fig. must not split before a number
			"See Fig. 3 for details. The result is clear.",
			[]string{"See Fig. 3 for details.", "The result is clear."},
		},
		{
			// known abbreviation: Dr. must not split before a name
			"Dr. Smith reported the findings. They were significant.",
			[]string{"Dr. Smith reported the findings.", "They were significant."},
		},
		{
			// internal periods: U.S.A. must not split
			"The U.S.A. reported growth. Results were published.",
			[]string{"The U.S.A. reported growth.", "Results were published."},
		},
		{
			// single-letter initial must not split
			"Written by A. Smith. The paper covers fusion.",
			[]string{"Written by A. Smith.", "The paper covers fusion."},
		},
		{
			"",
			nil,
		},
	}
	for _, tc := range cases {
		got := splitSentences(tc.in)
		if len(got) != len(tc.want) {
			t.Errorf("splitSentences(%q): len %d want %d: %v", tc.in, len(got), len(tc.want), got)
			continue
		}
		for i := range tc.want {
			if got[i] != tc.want[i] {
				t.Errorf("splitSentences(%q)[%d] = %q, want %q", tc.in, i, got[i], tc.want[i])
			}
		}
	}
}

// ── assignIDs ────────────────────────────────────────────────────────────────

func TestAssignIDs(t *testing.T) {
	chunks := []types.Chunk{
		{Section: "Introduction"},
		{Section: "Introduction"},
		{Section: "Methods"},
		{Section: ""}, // empty section → "doc"
	}
	assignIDs(chunks)

	want := []string{"introduction_0", "introduction_1", "methods_0", "doc_0"}
	for i, w := range want {
		if chunks[i].ID != w {
			t.Errorf("chunks[%d].ID = %q, want %q", i, chunks[i].ID, w)
		}
	}
}

// ── link ─────────────────────────────────────────────────────────────────────

func TestLink_SingleChunk(t *testing.T) {
	chunks := []types.Chunk{{ID: "a"}}
	link(chunks)
	if chunks[0].PrevID != "" || chunks[0].NextID != "" {
		t.Error("single chunk should have no prev/next")
	}
}

func TestLink_Chain(t *testing.T) {
	chunks := []types.Chunk{{ID: "a"}, {ID: "b"}, {ID: "c"}}
	link(chunks)

	if chunks[0].PrevID != "" {
		t.Errorf("chunks[0].PrevID = %q, want empty", chunks[0].PrevID)
	}
	if chunks[0].NextID != "b" {
		t.Errorf("chunks[0].NextID = %q, want b", chunks[0].NextID)
	}
	if chunks[1].PrevID != "a" {
		t.Errorf("chunks[1].PrevID = %q, want a", chunks[1].PrevID)
	}
	if chunks[1].NextID != "c" {
		t.Errorf("chunks[1].NextID = %q, want c", chunks[1].NextID)
	}
	if chunks[2].PrevID != "b" {
		t.Errorf("chunks[2].PrevID = %q, want b", chunks[2].PrevID)
	}
	if chunks[2].NextID != "" {
		t.Errorf("chunks[2].NextID = %q, want empty", chunks[2].NextID)
	}
}

// ── Chop — text splitting ────────────────────────────────────────────────────

func TestChop_ShortTextSingleChunk(t *testing.T) {
	c := newChunker(512)
	secs := []types.Section{textSection("Intro", "Short text.")}
	chunks, err := c.Chop(context.Background(), secs)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 1 {
		t.Fatalf("want 1 chunk, got %d", len(chunks))
	}
	if chunks[0].Text != "Short text." {
		t.Errorf("text: %q", chunks[0].Text)
	}
}

func TestChop_LongTextSplitsIntoMultiple(t *testing.T) {
	// MaxTokens=5; each sentence is >5 words, so each goes into its own chunk.
	c := newChunker(5)
	text := "The first sentence has many words. The second sentence also has many words. Third sentence is here too."
	secs := []types.Section{textSection("Sec", text)}
	chunks, err := c.Chop(context.Background(), secs)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks for long text, got %d", len(chunks))
	}
}

func TestChop_IDsAndLinks(t *testing.T) {
	c := newChunker(5)
	text := "Sentence one has words. Sentence two has words. Sentence three has words."
	secs := []types.Section{textSection("Methods", text)}
	chunks, err := c.Chop(context.Background(), secs)
	if err != nil {
		t.Fatal(err)
	}
	for i, ch := range chunks {
		if ch.ID == "" {
			t.Errorf("chunk[%d] has empty ID", i)
		}
	}
	if chunks[0].PrevID != "" {
		t.Errorf("first chunk.PrevID = %q, want empty", chunks[0].PrevID)
	}
	if chunks[len(chunks)-1].NextID != "" {
		t.Errorf("last chunk.NextID = %q, want empty", chunks[len(chunks)-1].NextID)
	}
	if len(chunks) >= 2 {
		if chunks[0].NextID != chunks[1].ID {
			t.Errorf("link broken: chunks[0].NextID=%q, chunks[1].ID=%q", chunks[0].NextID, chunks[1].ID)
		}
	}
}

func TestChop_SectionMetadata(t *testing.T) {
	c := newChunker(512)
	secs := []types.Section{{
		Title: "Results",
		Level: 2,
		Path:  []string{"Chapter 1", "Results"},
		Blocks: []types.Block{
			{Type: types.BlockText, Content: "Some result.", LineStart: 5, LineEnd: 5},
		},
	}}
	chunks, err := c.Chop(context.Background(), secs)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 1 {
		t.Fatal("want 1 chunk")
	}
	ch := chunks[0]
	if ch.Section != "Results" {
		t.Errorf("Section: %q", ch.Section)
	}
	if len(ch.SectionPath) != 2 || ch.SectionPath[1] != "Results" {
		t.Errorf("SectionPath: %v", ch.SectionPath)
	}
	if ch.SourceLines[0] != 5 {
		t.Errorf("SourceLines: %v", ch.SourceLines)
	}
}

// ── Chop — code / diagram / image ───────────────────────────────────────────

func TestChop_CodeBlockAlwaysWholeChunk(t *testing.T) {
	// Even with a tiny MaxTokens, code blocks must never be split.
	c := newChunker(2)
	code := "```go\n" + strings.Repeat("x := 1\n", 50) + "```"
	secs := []types.Section{{
		Title:  "Code",
		Blocks: []types.Block{{Type: types.BlockCode, Content: code}},
	}}
	chunks, err := c.Chop(context.Background(), secs)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 1 {
		t.Fatalf("code block must be single chunk, got %d", len(chunks))
	}
	if chunks[0].BlockType != types.BlockCode {
		t.Errorf("BlockType: want BlockCode, got %s", chunks[0].BlockType)
	}
}

func TestChop_DiagramBlockWholeChunk(t *testing.T) {
	c := newChunker(2)
	diag := "```mermaid\n" + strings.Repeat("A --> B\n", 30) + "```"
	secs := []types.Section{{
		Blocks: []types.Block{{Type: types.BlockDiagram, Content: diag}},
	}}
	chunks, err := c.Chop(context.Background(), secs)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 1 || chunks[0].BlockType != types.BlockDiagram {
		t.Errorf("diagram must be single BlockDiagram chunk, got %d chunks", len(chunks))
	}
}

// ── Chop — table splitting ───────────────────────────────────────────────────

func TestChop_TableFitsInSingleChunk(t *testing.T) {
	c := newChunker(512)
	table := "| A | B |\n|---|---|\n| 1 | 2 |"
	secs := []types.Section{{
		Blocks: []types.Block{{Type: types.BlockTable, Content: table}},
	}}
	chunks, err := c.Chop(context.Background(), secs)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 1 {
		t.Fatalf("small table should be 1 chunk, got %d", len(chunks))
	}
}

func TestChop_TableSplitRowsKeepHeader(t *testing.T) {
	header := "| Column A | Column B | Column C |"
	sep := "|----------|----------|----------|"
	rows := make([]string, 20)
	for i := range rows {
		rows[i] = fmt.Sprintf("| row%02d | value%02d | data%02d |", i, i, i)
	}
	table := header + "\n" + sep + "\n" + strings.Join(rows, "\n")

	c := newChunker(30) // small limit to force splitting
	secs := []types.Section{{
		Blocks: []types.Block{{Type: types.BlockTable, Content: table}},
	}}
	chunks, err := c.Chop(context.Background(), secs)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) < 2 {
		t.Fatalf("large table should be split, got %d chunks", len(chunks))
	}
	// Every chunk must contain the header and separator.
	for i, ch := range chunks {
		if !strings.Contains(ch.Text, header) {
			t.Errorf("chunk[%d] missing table header", i)
		}
		if !strings.Contains(ch.Text, sep) {
			t.Errorf("chunk[%d] missing table separator", i)
		}
	}
}

// ── Chop — list splitting ────────────────────────────────────────────────────

func TestChop_ListFitsInSingleChunk(t *testing.T) {
	c := newChunker(512)
	list := "- alpha\n- beta\n- gamma"
	secs := []types.Section{{
		Blocks: []types.Block{{Type: types.BlockList, Content: list}},
	}}
	chunks, err := c.Chop(context.Background(), secs)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 1 {
		t.Fatalf("small list: want 1 chunk, got %d", len(chunks))
	}
}

func TestChop_ListSplitOnItemBoundaries(t *testing.T) {
	// Each item is ~5 tokens; MaxTokens=8 forces a new chunk every ~1-2 items.
	items := make([]string, 20)
	for i := range items {
		items[i] = fmt.Sprintf("- item number %02d value here", i)
	}
	list := strings.Join(items, "\n")

	c := newChunker(10)
	secs := []types.Section{{
		Blocks: []types.Block{{Type: types.BlockList, Content: list}},
	}}
	chunks, err := c.Chop(context.Background(), secs)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) < 2 {
		t.Fatalf("large list should be split, got %d chunks", len(chunks))
	}
	// Every chunk must start with a list item marker.
	for i, ch := range chunks {
		if !strings.HasPrefix(strings.TrimSpace(ch.Text), "- ") {
			t.Errorf("chunk[%d] does not start with list marker: %q", i, ch.Text[:min(40, len(ch.Text))])
		}
	}
}

// ── Chop — multi-section ─────────────────────────────────────────────────────

func TestChop_MultipleSections(t *testing.T) {
	c := newChunker(512)
	secs := []types.Section{
		textSection("Introduction", "Intro text here."),
		textSection("Methods", "Methods text here."),
		textSection("Results", "Results text here."),
	}
	chunks, err := c.Chop(context.Background(), secs)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 3 {
		t.Fatalf("want 3 chunks (one per section), got %d", len(chunks))
	}
	sections := []string{"Introduction", "Methods", "Results"}
	for i, want := range sections {
		if chunks[i].Section != want {
			t.Errorf("chunk[%d].Section = %q, want %q", i, chunks[i].Section, want)
		}
	}
}

func TestChop_EmptySectionsProduceNoChunks(t *testing.T) {
	c := newChunker(512)
	// Section with no blocks.
	secs := []types.Section{{Title: "Empty", Level: 1, Path: []string{"Empty"}}}
	chunks, err := c.Chop(context.Background(), secs)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 0 {
		t.Fatalf("empty section should produce 0 chunks, got %d", len(chunks))
	}
}

// ── TokenCount field ─────────────────────────────────────────────────────────

func TestChop_TokenCountPopulated(t *testing.T) {
	c := newChunker(512)
	chunks, err := c.Chop(context.Background(), []types.Section{
		textSection("X", "one two three four five"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) == 0 {
		t.Fatal("no chunks")
	}
	if chunks[0].TokenCount != 5 {
		t.Errorf("TokenCount: want 5, got %d", chunks[0].TokenCount)
	}
}

// ── isListItemStart ───────────────────────────────────────────────────────────

func TestIsListItemStart(t *testing.T) {
	trues := []string{"- item", "* item", "1. first", "10. tenth"}
	for _, s := range trues {
		if !isListItemStart(s) {
			t.Errorf("isListItemStart(%q) = false, want true", s)
		}
	}
	falses := []string{"", "  - indented", "plain", "1 no dot"}
	for _, s := range falses {
		if isListItemStart(s) {
			t.Errorf("isListItemStart(%q) = true, want false", s)
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
