package segmenter

import (
	"testing"

	"github.com/ETretyakov/kukri/types"
)

func hdr(title string, level int) types.Block {
	return types.Block{Type: types.BlockHeader, Content: title, Level: level}
}

func txt(content string) types.Block {
	return types.Block{Type: types.BlockText, Content: content}
}

// ── Segment ──────────────────────────────────────────────────────────────────

func TestSegment_EmptyInput(t *testing.T) {
	secs := Segment(nil)
	if len(secs) != 0 {
		t.Fatalf("want 0 sections, got %d", len(secs))
	}
}

func TestSegment_NoHeaders_Preamble(t *testing.T) {
	secs := Segment([]types.Block{txt("Hello."), txt("World.")})
	if len(secs) != 1 {
		t.Fatalf("want 1 preamble section, got %d", len(secs))
	}
	if secs[0].Title != "" {
		t.Errorf("preamble title should be empty, got %q", secs[0].Title)
	}
	if secs[0].Level != 0 {
		t.Errorf("preamble level should be 0, got %d", secs[0].Level)
	}
	if len(secs[0].Blocks) != 2 {
		t.Errorf("preamble should have 2 blocks, got %d", len(secs[0].Blocks))
	}
}

func TestSegment_HeaderWithNoContent_Omitted(t *testing.T) {
	// A header with no following content blocks is dropped.
	secs := Segment([]types.Block{hdr("Empty Section", 1)})
	if len(secs) != 0 {
		t.Fatalf("want 0 sections (header with no content), got %d", len(secs))
	}
}

func TestSegment_SingleHeaderWithContent(t *testing.T) {
	secs := Segment([]types.Block{
		hdr("Introduction", 1),
		txt("Some intro text."),
	})
	if len(secs) != 1 {
		t.Fatalf("want 1 section, got %d", len(secs))
	}
	s := secs[0]
	if s.Title != "Introduction" {
		t.Errorf("title: %q", s.Title)
	}
	if s.Level != 1 {
		t.Errorf("level: %d", s.Level)
	}
	if len(s.Path) != 1 || s.Path[0] != "Introduction" {
		t.Errorf("path: %v", s.Path)
	}
}

func TestSegment_NestedHeaders_Paths(t *testing.T) {
	secs := Segment([]types.Block{
		hdr("Chapter 1", 1),
		txt("Chapter intro."),
		hdr("1.1 Introduction", 2),
		txt("Section intro."),
		hdr("1.1.1 Background", 3),
		txt("Background text."),
		hdr("1.2 Methods", 2),
		txt("Methods text."),
	})
	if len(secs) != 4 {
		t.Fatalf("want 4 sections, got %d", len(secs))
	}

	wantPaths := [][]string{
		{"Chapter 1"},
		{"Chapter 1", "1.1 Introduction"},
		{"Chapter 1", "1.1 Introduction", "1.1.1 Background"},
		{"Chapter 1", "1.2 Methods"},
	}
	for i, want := range wantPaths {
		got := secs[i].Path
		if len(got) != len(want) {
			t.Errorf("section[%d] path len: want %d, got %d (%v)", i, len(want), len(got), got)
			continue
		}
		for j := range want {
			if got[j] != want[j] {
				t.Errorf("section[%d].Path[%d]: want %q, got %q", i, j, want[j], got[j])
			}
		}
	}
}

func TestSegment_H3ThenH2_ClearsDeepLevel(t *testing.T) {
	// After a H3, going back to H2 should not carry H3 in the path.
	secs := Segment([]types.Block{
		hdr("H1", 1), txt("text"),
		hdr("H2a", 2), txt("text"),
		hdr("H3", 3), txt("text"),
		hdr("H2b", 2), txt("text"),
	})
	last := secs[len(secs)-1]
	if len(last.Path) != 2 {
		t.Fatalf("H2b path should have 2 elements, got %v", last.Path)
	}
	if last.Path[1] != "H2b" {
		t.Errorf("second path element should be H2b, got %q", last.Path[1])
	}
}

func TestSegment_PreambleThenSections(t *testing.T) {
	secs := Segment([]types.Block{
		txt("Preamble text."),
		hdr("Section A", 1),
		txt("Section A content."),
	})
	if len(secs) != 2 {
		t.Fatalf("want 2 sections (preamble + A), got %d", len(secs))
	}
	if secs[0].Title != "" {
		t.Errorf("first section should be preamble, got title %q", secs[0].Title)
	}
	if secs[1].Title != "Section A" {
		t.Errorf("second section title: %q", secs[1].Title)
	}
}

func TestSegment_HeadersOnlyNonAdjacentContent(t *testing.T) {
	// Multiple headers with content; verify block counts.
	secs := Segment([]types.Block{
		hdr("A", 1), txt("a1"), txt("a2"),
		hdr("B", 1), txt("b1"),
		hdr("C", 1), txt("c1"), txt("c2"), txt("c3"),
	})
	if len(secs) != 3 {
		t.Fatalf("want 3 sections, got %d", len(secs))
	}
	if len(secs[0].Blocks) != 2 {
		t.Errorf("section A: want 2 blocks, got %d", len(secs[0].Blocks))
	}
	if len(secs[2].Blocks) != 3 {
		t.Errorf("section C: want 3 blocks, got %d", len(secs[2].Blocks))
	}
}

// ── buildPath ────────────────────────────────────────────────────────────────

func TestBuildPath_SkipsMissingLevels(t *testing.T) {
	path := make([]string, 7)
	path[2] = "H2 only" // no H1
	got := buildPath(path, 2)
	if len(got) != 1 || got[0] != "H2 only" {
		t.Errorf("got %v", got)
	}
}

func TestBuildPath_FullHierarchy(t *testing.T) {
	path := make([]string, 7)
	path[1] = "Root"
	path[2] = "Child"
	path[3] = "Grandchild"
	got := buildPath(path, 3)
	if len(got) != 3 {
		t.Fatalf("want 3 elements, got %v", got)
	}
	if got[0] != "Root" || got[1] != "Child" || got[2] != "Grandchild" {
		t.Errorf("wrong path: %v", got)
	}
}
