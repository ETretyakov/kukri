package chunker

import (
	"context"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/ETretyakov/kukri/scorer"
	"github.com/ETretyakov/kukri/types"
)

type fixedScorer []float32

func (s fixedScorer) Score(context.Context, []string) ([]float32, error) { return s, nil }

func TestCumulativeBudgetAndModes(t *testing.T) {
	var sec types.Section
	for i := 0; i < 3; i++ {
		sec.Blocks = append(sec.Blocks, types.Block{Type: types.BlockText, Content: strings.TrimSpace(strings.Repeat("word ", 60)), LineStart: i + 1, LineEnd: i + 1})
	}
	for _, tc := range []struct {
		mode      types.MergeMode
		scr       scorer.Scorer
		threshold float32
		want      []int
	}{
		{types.MergeSize, nil, 0, []int{120, 60}},
		{types.MergeNone, nil, 0, []int{60, 60, 60}},
		{types.MergeSimilarity, fixedScorer{0.9, 0.1}, 0.5, []int{120, 60}},
	} {
		chunks, err := New(types.Config{MaxTokens: 128, MergeMode: tc.mode, Threshold: tc.threshold}, tc.scr).Chop(context.Background(), []types.Section{sec})
		if err != nil {
			t.Fatal(err)
		}
		var got []int
		for _, ch := range chunks {
			got = append(got, ch.Size)
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("%s: %v, want %v", tc.mode, got, tc.want)
		}
	}
}

func TestLongSentencePreservesAllWords(t *testing.T) {
	text := strings.TrimSpace(strings.Repeat("слово ", 300)) + "."
	chunks, err := New(types.Config{MaxTokens: 128}, nil).Chop(context.Background(), []types.Section{textSection("", text)})
	if err != nil {
		t.Fatal(err)
	}
	var parts []string
	for _, c := range chunks {
		if c.Size > 128 || c.Oversized {
			t.Fatalf("unbounded text: %+v", c)
		}
		parts = append(parts, c.Text)
	}
	if len(chunks) != 3 || strings.Join(parts, " ") != text {
		t.Fatal("lost or duplicated words")
	}
}

func TestInvalidConfigurationAndScorer(t *testing.T) {
	configs := []types.Config{
		{MaxTokens: 0}, {MaxTokens: -1}, {MaxTokens: 128, Threshold: 0.5},
		{MaxTokens: 128, MergeMode: types.MergeSimilarity}, {MaxTokens: 128, MergeMode: "unknown"},
		{MaxTokens: 128, OversizePolicy: "unknown"}, {MaxTokens: 128, LengthFunc: func(s string) int { return len(s) }},
	}
	for _, cfg := range configs {
		if _, err := New(cfg, nil).Chop(context.Background(), nil); err == nil {
			t.Fatalf("accepted %+v", cfg)
		}
	}
	sec := types.Section{Blocks: []types.Block{{Type: types.BlockText, Content: "alpha"}, {Type: types.BlockText, Content: "beta"}}}
	for _, scores := range []fixedScorer{nil, {0.1, 0.2}, {float32(math.NaN())}, {2}} {
		_, err := New(types.Config{MaxTokens: 128, MergeMode: types.MergeSimilarity}, scores).Chop(context.Background(), []types.Section{sec})
		if err == nil {
			t.Fatalf("accepted scores %v", scores)
		}
	}
}

func TestCustomLengthMeasuresSeparators(t *testing.T) {
	sec := types.Section{Blocks: []types.Block{{Type: types.BlockText, Content: "abc"}, {Type: types.BlockText, Content: "def"}}}
	chunks, err := New(types.Config{MaxTokens: 7, LengthFunc: func(s string) int { return len(s) }, LengthUnit: "bytes"}, nil).Chop(context.Background(), []types.Section{sec})
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 2 {
		t.Fatal("separator bytes were omitted from budget")
	}
}

func TestOversizePolicies(t *testing.T) {
	for _, kind := range []types.BlockType{types.BlockCode, types.BlockList, types.BlockTable} {
		sec := types.Section{Blocks: []types.Block{{Type: kind, Content: strings.Repeat("word ", 12)}}}
		chunks, err := New(types.Config{MaxTokens: 3}, nil).Chop(context.Background(), []types.Section{sec})
		if err != nil {
			t.Fatal(err)
		}
		if len(chunks) != 1 || !chunks[0].Oversized || chunks[0].OversizeReason == "" {
			t.Fatalf("unmarked %s", kind)
		}
		if _, err := New(types.Config{MaxTokens: 3, OversizePolicy: types.OversizeError}, nil).Chop(context.Background(), []types.Section{sec}); err == nil {
			t.Fatalf("accepted oversized %s", kind)
		}
	}
}

func TestBoundariesAndPrefix(t *testing.T) {
	sections := []types.Section{{Title: "Root", Path: []string{"Root"}, Blocks: []types.Block{
		{Type: types.BlockText, Content: "before"}, {Type: types.BlockCode, Content: "code"}, {Type: types.BlockText, Content: "after"},
	}}, {Title: "Next", Path: []string{"Next"}, Blocks: []types.Block{{Type: types.BlockText, Content: "last"}}}}
	chunks, err := New(types.Config{MaxTokens: 128, PrependSectionPath: true}, nil).Chop(context.Background(), sections)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 4 || chunks[0].EmbedText != "Root\n\nbefore" || chunks[0].Text != "before" || chunks[3].Section != "Next" {
		t.Fatal(chunks)
	}
}
