package parser

import (
	"strings"
	"testing"

	"github.com/ETretyakov/kukri/types"
)

// ── Parse ────────────────────────────────────────────────────────────────────

func TestParse_EmptyInput(t *testing.T) {
	blocks := Parse("")
	if len(blocks) != 0 {
		t.Fatalf("empty input: want 0 blocks, got %d", len(blocks))
	}
}

func TestParse_TextParagraph(t *testing.T) {
	blocks := Parse("Hello world.\nThis continues.")
	if len(blocks) != 1 {
		t.Fatalf("want 1 block, got %d", len(blocks))
	}
	b := blocks[0]
	if b.Type != types.BlockText {
		t.Errorf("type: want BlockText, got %s", b.Type)
	}
	// Lines joined with a space.
	if b.Content != "Hello world. This continues." {
		t.Errorf("content: %q", b.Content)
	}
	if b.LineStart != 1 || b.LineEnd != 2 {
		t.Errorf("lines: want [1,2], got [%d,%d]", b.LineStart, b.LineEnd)
	}
}

func TestParse_Headers(t *testing.T) {
	cases := []struct {
		input string
		level int
		title string
	}{
		{"# Top", 1, "Top"},
		{"## Section", 2, "Section"},
		{"### Sub", 3, "Sub"},
		{"###### Deep", 6, "Deep"},
	}
	for _, tc := range cases {
		blocks := Parse(tc.input)
		if len(blocks) != 1 {
			t.Fatalf("%q: want 1 block, got %d", tc.input, len(blocks))
		}
		b := blocks[0]
		if b.Type != types.BlockHeader {
			t.Errorf("%q: type want BlockHeader, got %s", tc.input, b.Type)
		}
		if b.Level != tc.level {
			t.Errorf("%q: level want %d, got %d", tc.input, tc.level, b.Level)
		}
		if b.Content != tc.title {
			t.Errorf("%q: title want %q, got %q", tc.input, tc.title, b.Content)
		}
	}
}

func TestParse_HeaderNotSplit_NoSpace(t *testing.T) {
	// "#title" without a space is not an ATX header — treated as text.
	blocks := Parse("#notaheader")
	if len(blocks) == 0 {
		t.Fatal("expected at least one block")
	}
	if blocks[0].Type == types.BlockHeader {
		t.Error("'#notaheader' should not be parsed as a header")
	}
}

func TestParse_FencedCodeBlock(t *testing.T) {
	input := "```go\nfmt.Println(\"hello\")\n```"
	blocks := Parse(input)
	if len(blocks) != 1 {
		t.Fatalf("want 1 block, got %d", len(blocks))
	}
	b := blocks[0]
	if b.Type != types.BlockCode {
		t.Errorf("type: want BlockCode, got %s", b.Type)
	}
	if !strings.Contains(b.Content, "fmt.Println") {
		t.Errorf("content missing body: %q", b.Content)
	}
}

func TestParse_DiagramBlock(t *testing.T) {
	for _, lang := range []string{"mermaid", "plantuml", "dot"} {
		input := "```" + lang + "\ngraph LR\n  A --> B\n```"
		blocks := Parse(input)
		if len(blocks) != 1 {
			t.Fatalf("%s: want 1 block, got %d", lang, len(blocks))
		}
		if blocks[0].Type != types.BlockDiagram {
			t.Errorf("%s: want BlockDiagram, got %s", lang, blocks[0].Type)
		}
	}
}

func TestParse_Table(t *testing.T) {
	input := "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
	blocks := Parse(input)
	if len(blocks) != 1 {
		t.Fatalf("want 1 block, got %d", len(blocks))
	}
	b := blocks[0]
	if b.Type != types.BlockTable {
		t.Errorf("type: want BlockTable, got %s", b.Type)
	}
	if !strings.Contains(b.Content, "| A | B |") {
		t.Error("table content missing header")
	}
}

func TestParse_UnorderedList(t *testing.T) {
	input := "- alpha\n- beta\n- gamma"
	blocks := Parse(input)
	if len(blocks) != 1 {
		t.Fatalf("want 1 block, got %d", len(blocks))
	}
	if blocks[0].Type != types.BlockList {
		t.Errorf("type: want BlockList, got %s", blocks[0].Type)
	}
}

func TestParse_OrderedList(t *testing.T) {
	input := "1. first\n2. second\n3. third"
	blocks := Parse(input)
	if len(blocks) != 1 {
		t.Fatalf("want 1 block, got %d", len(blocks))
	}
	if blocks[0].Type != types.BlockList {
		t.Errorf("type: want BlockList, got %s", blocks[0].Type)
	}
}

func TestParse_ListWithContinuation(t *testing.T) {
	input := "- item one\n  still item one\n- item two"
	blocks := Parse(input)
	if len(blocks) != 1 {
		t.Fatalf("want 1 block, got %d", len(blocks))
	}
	if !strings.Contains(blocks[0].Content, "still item one") {
		t.Error("continuation line not included in list block")
	}
}

func TestParse_ImageBlock(t *testing.T) {
	input := "> **[IMAGE | Figure 1: diagram]**"
	blocks := Parse(input)
	if len(blocks) != 1 {
		t.Fatalf("want 1 block, got %d", len(blocks))
	}
	if blocks[0].Type != types.BlockImage {
		t.Errorf("type: want BlockImage, got %s", blocks[0].Type)
	}
}

func TestParse_MixedDocument(t *testing.T) {
	input := strings.Join([]string{
		"# Introduction",
		"",
		"This is the intro paragraph.",
		"",
		"## Methods",
		"",
		"| Step | Action |",
		"|------|--------|",
		"| 1    | Collect |",
		"",
		"```python",
		"print('hello')",
		"```",
		"",
		"- item A",
		"- item B",
	}, "\n")

	blocks := Parse(input)

	types_ := make([]types.BlockType, len(blocks))
	for i, b := range blocks {
		types_[i] = b.Type
	}
	want := []types.BlockType{
		types.BlockHeader,
		types.BlockText,
		types.BlockHeader,
		types.BlockTable,
		types.BlockCode,
		types.BlockList,
	}
	if len(blocks) != len(want) {
		t.Fatalf("want %d blocks, got %d: %v", len(want), len(blocks), types_)
	}
	for i, w := range want {
		if blocks[i].Type != w {
			t.Errorf("block[%d]: want %s, got %s", i, w, blocks[i].Type)
		}
	}
}

func TestParse_BlankLinesBetweenBlocks(t *testing.T) {
	input := "First paragraph.\n\n\nSecond paragraph."
	blocks := Parse(input)
	if len(blocks) != 2 {
		t.Fatalf("want 2 text blocks, got %d", len(blocks))
	}
}

// ── helpers ──────────────────────────────────────────────────────────────────

func TestIsListLine(t *testing.T) {
	trues := []string{"- item", "* item", "1. first", "12. twelfth", "  - nested"}
	for _, s := range trues {
		if !isListLine(s) {
			t.Errorf("isListLine(%q) = false, want true", s)
		}
	}
	falses := []string{"", "plain text", "#header", "| table |"}
	for _, s := range falses {
		if isListLine(s) {
			t.Errorf("isListLine(%q) = true, want false", s)
		}
	}
}

func TestIsContinuation(t *testing.T) {
	if !isContinuation("  indented line") {
		t.Error("two-space indent should be continuation")
	}
	if !isContinuation("    deep indent") {
		t.Error("four-space indent should be continuation")
	}
	if isContinuation("no indent") {
		t.Error("no indent should not be continuation")
	}
	if isContinuation("  ") {
		t.Error("blank indented line should not be continuation")
	}
}

func TestHeadingLimitsAndWhitespace(t *testing.T) {
	for _, text := range []string{"####### text", "######## text"} {
		blocks := Parse(text)
		if len(blocks) != 1 || blocks[0].Type != types.BlockText || blocks[0].Content != text {
			t.Fatal(blocks)
		}
	}
	blocks := Parse("  ##\tTitle ###\r\nbody\r\n")
	if blocks[0].Level != 2 || blocks[0].Content != "Title" || blocks[1].Content != "body" {
		t.Fatal(blocks)
	}
}

func TestFenceDelimiterMustMatch(t *testing.T) {
	for _, text := range []string{"````md\n```\nstill code\n````", "~~~go\n# code\n~~~"} {
		blocks := Parse(text)
		if len(blocks) != 1 || blocks[0].Type != types.BlockCode || blocks[0].Content != text {
			t.Fatal(blocks)
		}
	}
}
