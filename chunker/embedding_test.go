package chunker

import (
	"context"
	"errors"
	"github.com/ETretyakov/kukri/types"
	"reflect"
	"strings"
	"testing"
)

func byteEmbeddingBudget(n int) *types.EmbeddingBudget {
	return &types.EmbeddingBudget{MaxTokens: n, Tokenizer: "test-byte-plus-specials", CountTokens: func(s string) (int, error) { return len(s) + 2, nil }}
}

func TestCompleteEmbeddingBudget(t *testing.T) {
	cfg := types.Config{MaxTokens: 128, PrependSectionPath: true, EmbeddingBudget: byteEmbeddingBudget(14)}
	sec := textSection("Topic", "aa bb cc dd ee")
	sec.Path = []string{"Topic"}
	chunks, err := New(cfg, nil).Chop(context.Background(), []types.Section{sec})
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 3 {
		t.Fatalf("want 3 chunks with heading + two special tokens; got %+v", chunks)
	}
	var texts []string
	for _, ch := range chunks {
		if ch.EmbeddingTokens != len(ch.EmbedText)+2 || ch.EmbeddingTokens > 14 || ch.Oversized || ch.EmbeddingOversized {
			t.Fatal(ch)
		}
		texts = append(texts, ch.Text)
	}
	if strings.Join(texts, " ") != "aa bb cc dd ee" {
		t.Fatal("content lost")
	}
}

func TestEmbeddingAtomicOverflowAndCounterErrors(t *testing.T) {
	for _, kind := range []types.BlockType{types.BlockText, types.BlockCode, types.BlockTable, types.BlockList, types.BlockImage} {
		sec := types.Section{Blocks: []types.Block{{Type: kind, Content: "unbreakable"}}}
		cfg := types.Config{MaxTokens: 128, EmbeddingBudget: byteEmbeddingBudget(5)}
		chunks, err := New(cfg, nil).Chop(context.Background(), []types.Section{sec})
		if err != nil || len(chunks) != 1 || !chunks[0].EmbeddingOversized || chunks[0].Oversized || chunks[0].EmbeddingOversizeReason == "" {
			t.Fatalf("%s: %+v %v", kind, chunks, err)
		}
		cfg.OversizePolicy = types.OversizeError
		if _, err = New(cfg, nil).Chop(context.Background(), []types.Section{sec}); err == nil {
			t.Fatal("accepted embedding overflow")
		}
	}
	sentinel := errors.New("counter unavailable")
	for _, counter := range []func(string) (int, error){
		func(s string) (int, error) {
			if strings.Contains(s, "bb") {
				return 0, sentinel
			}
			return len(s), nil
		},
		func(s string) (int, error) {
			if strings.Contains(s, "bb") {
				return -1, nil
			}
			return len(s), nil
		},
	} {
		cfg := types.Config{MaxTokens: 128, EmbeddingBudget: &types.EmbeddingBudget{MaxTokens: 20, Tokenizer: "test", CountTokens: counter}}
		if _, err := New(cfg, nil).Chop(context.Background(), []types.Section{textSection("", "aa bb")}); err == nil {
			t.Fatal("lost counter failure")
		}
	}
	cfg := types.Config{MaxTokens: 128, PrependSectionPath: true, EmbeddingBudget: byteEmbeddingBudget(5)}
	sec := textSection("Heading too long", "ok")
	sec.Path = []string{sec.Title}
	ch, err := New(cfg, nil).Chop(context.Background(), []types.Section{sec})
	if err != nil || ch[0].EmbeddingOversizeReason != "embedding_prefix_or_special_tokens" {
		t.Fatal(ch, err)
	}
}

func TestEmbeddingBudgetIncludesRepeatedTableHeader(t *testing.T) {
	sec := types.Section{Path: []string{"T"}, Blocks: []types.Block{{Type: types.BlockTable, Content: "| A |\n|---|\n| x |\n| y |"}}}
	cfg := types.Config{MaxTokens: 128, PrependSectionPath: true, EmbeddingBudget: byteEmbeddingBudget(22)}
	chunks, err := New(cfg, nil).Chop(context.Background(), []types.Section{sec})
	if err != nil || len(chunks) != 2 {
		t.Fatal(chunks, err)
	}
	for _, ch := range chunks {
		if !strings.HasPrefix(ch.Text, "| A |\n|---|\n") || ch.EmbeddingTokens > 22 {
			t.Fatal(ch)
		}
	}
}

func TestRepackTailAndPreserveBoundaries(t *testing.T) {
	blocks := []types.Block{{Type: types.BlockText, Content: "a b c d e f g", LineStart: 1, LineEnd: 1}, {Type: types.BlockText, Content: "h i", LineStart: 3, LineEnd: 3}, {Type: types.BlockCode, Content: "code", LineStart: 5, LineEnd: 7}, {Type: types.BlockText, Content: "j", LineStart: 9, LineEnd: 9}}
	for _, tc := range []struct {
		repack bool
		want   []string
	}{{false, []string{"a b c d e", "f g", "h i", "code", "j"}}, {true, []string{"a b c d e", "f g\n\nh i", "code", "j"}}} {
		cfg := types.Config{MaxTokens: 5, RepackText: tc.repack}
		chunks, err := New(cfg, nil).Chop(context.Background(), []types.Section{{Blocks: blocks}})
		if err != nil {
			t.Fatal(err)
		}
		var got []string
		for i, ch := range chunks {
			got = append(got, ch.Text)
			if i > 0 && ch.PrevID != chunks[i-1].ID {
				t.Fatal("links")
			}
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("%v: %q", tc.repack, got)
		}
		if tc.repack && chunks[1].SourceLines != [2]int{1, 3} {
			t.Fatal(chunks[1])
		}
	}
	if err := New(types.Config{MaxTokens: 5, RepackText: true, MergeMode: types.MergeNone}, nil).Validate(); err == nil {
		t.Fatal("repacking none allowed")
	}
}

func TestEmbeddingCandidatesAreMeasuredAsWhole(t *testing.T) {
	// A non-additive counter: separators cost more than individual bodies.
	b := &types.EmbeddingBudget{MaxTokens: 6, Tokenizer: "nonadditive", CountTokens: func(s string) (int, error) { return len(s) + 2*strings.Count(s, "\n"), nil }}
	cfg := types.Config{MaxTokens: 128, EmbeddingBudget: b, RepackText: true}
	sec := types.Section{Blocks: []types.Block{{Type: types.BlockText, Content: "aa"}, {Type: types.BlockText, Content: "bb"}}}
	chunks, err := New(cfg, nil).Chop(context.Background(), []types.Section{sec})
	if err != nil || len(chunks) != 2 {
		t.Fatal(chunks, err)
	}
}

func TestCompactTableEmbeddingPreservesSourceAndEscapes(t *testing.T) {
	text := "| Name         | Value            |\n|:-------------|-----------------:|\n| a\\|b         | x                |\n| c            | y                |"
	sec := types.Section{Blocks: []types.Block{{Type: types.BlockTable, Content: text}}}
	cfg := types.Config{MaxTokens: 128, EmbeddingBudget: byteEmbeddingBudget(100), CompactTableEmbedding: true}
	chunks, err := New(cfg, nil).Chop(context.Background(), []types.Section{sec})
	if err != nil || len(chunks) != 1 {
		t.Fatal(chunks, err)
	}
	ch := chunks[0]
	if ch.Text != text || ch.EmbeddingOversized || !strings.Contains(ch.EmbedText, "a\\|b") || !strings.Contains(ch.EmbedText, ":--- | ---:") {
		t.Fatal(ch)
	}
	cfg.CompactTableEmbedding = false
	raw, err := New(cfg, nil).Chop(context.Background(), []types.Section{sec})
	if err != nil || len(raw) != 2 || !raw[0].EmbeddingOversized {
		t.Fatal(raw, err)
	}
}

func TestOversizedPrefixDoesNotExplodeIntoWordChunks(t *testing.T) {
	sec := textSection("long heading", strings.Repeat("word ", 50))
	sec.Path = []string{sec.Title}
	cfg := types.Config{MaxTokens: 20, EmbeddingBudget: byteEmbeddingBudget(5), PrependSectionPath: true, RepackText: true}
	chunks, err := New(cfg, nil).Chop(context.Background(), []types.Section{sec})
	if err != nil || len(chunks) != 3 {
		t.Fatal(len(chunks), err)
	}
	for _, ch := range chunks {
		if ch.Size > 20 || ch.EmbeddingOversizeReason != "embedding_prefix_or_special_tokens" {
			t.Fatal(ch)
		}
	}
}

func TestEmbeddingOnlyBudget(t *testing.T) {
	cfg := types.Config{EmbeddingBudget: byteEmbeddingBudget(10), RepackText: true}
	chunks, err := New(cfg, nil).Chop(context.Background(), []types.Section{textSection("", "aa bb cc dd ee")})
	if err != nil || len(chunks) != 2 {
		t.Fatal(chunks, err)
	}
	for _, ch := range chunks {
		if ch.Oversized || ch.EmbeddingOversized || ch.Size == 0 {
			t.Fatal(ch)
		}
	}
}
