package chunker

import (
	"context"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/ETretyakov/kukri/types"
)

func budgetSections() []types.Section {
	return []types.Section{{Title: "API", Path: []string{"API"}, Blocks: []types.Block{
		{Type: types.BlockText, Content: "HTTP requests use Alpha Gateway. Alpha Gateway handles HTTP requests.", LineStart: 2, LineEnd: 2},
		{Type: types.BlockCode, Content: "```go\nclient.Send()\n```", LineStart: 4, LineEnd: 6},
		{Type: types.BlockText, Content: "TLS encryption is enabled. Do not disable TLS encryption.", LineStart: 8, LineEnd: 8},
	}}, {Title: "Other", Path: []string{"Other"}, Blocks: []types.Block{{Type: types.BlockText, Content: "Retries are forbidden for writes.", LineStart: 11, LineEnd: 11}}}}
}

func TestDocumentBudgetLimitsAndIdentity(t *testing.T) {
	ctx := context.Background()
	sections := budgetSections()
	cfg := types.Config{MaxTokens: 20, PrependSectionPath: true, EmbeddingBudget: byteEmbeddingBudget(200)}
	base, err := New(cfg, nil).ChopWithReport(ctx, sections)
	if err != nil || base.Compression != nil {
		t.Fatal(base, err)
	}
	for _, strategy := range []types.SelectionStrategy{types.SelectUniform, types.SelectTerms, types.SelectTermsContext} {
		for _, fraction := range []float64{0.01, 0.25, 0.5, 0.9, 1} {
			cfg.DocumentBudget = &types.DocumentBudget{RetainFraction: fraction, Strategy: strategy}
			got, err := New(cfg, nil).ChopWithReport(ctx, sections)
			if err != nil {
				t.Fatal(err)
			}
			r := got.Compression
			sum := 0
			last := -1
			if r.SelectedChunks != len(got.Chunks) || r.OriginalChunks != len(base.Chunks) || r.OmittedChunks+r.SelectedChunks != r.OriginalChunks {
				t.Fatal(r)
			}
			for i, ch := range got.Chunks {
				sum += ch.EmbeddingTokens
				found := -1
				for j, old := range base.Chunks {
					if ch.ID == old.ID {
						found = j
						old.PrevID = ch.PrevID
						old.NextID = ch.NextID
						if !reflect.DeepEqual(ch, old) {
							t.Fatal("source chunk changed")
						}
						break
					}
				}
				if found <= last {
					t.Fatal("identity/order changed")
				}
				last = found
				if i == 0 && ch.PrevID != "" {
					t.Fatal("dangling previous")
				}
				if i == len(got.Chunks)-1 && ch.NextID != "" {
					t.Fatal("dangling next")
				}
				if i > 0 && (ch.PrevID != got.Chunks[i-1].ID || got.Chunks[i-1].NextID != ch.ID) {
					t.Fatal("broken links")
				}
			}
			if sum != r.SelectedTokens || sum > r.TokenLimit {
				t.Fatal(r)
			}
			if fraction == 1 && !reflect.DeepEqual(got.Chunks, base.Chunks) {
				t.Fatal("full budget must be identity")
			}
			again, _ := New(cfg, nil).ChopWithReport(ctx, sections)
			if !reflect.DeepEqual(got, again) {
				t.Fatal("nondeterministic selection")
			}
		}
	}
	cfg.DocumentBudget = &types.DocumentBudget{MaxChunks: 1, MaxTokens: 80, RetainFraction: 0.5}
	got, err := New(cfg, nil).ChopWithReport(ctx, sections)
	if err != nil || len(got.Chunks) > 1 || got.Compression.SelectedTokens > 80 {
		t.Fatal(got, err)
	}
	cfg.EmbeddingBudget = nil
	cfg.DocumentBudget = &types.DocumentBudget{MaxChunks: 2}
	got, err = New(cfg, nil).ChopWithReport(ctx, sections)
	if err != nil || len(got.Chunks) > 2 || got.Compression.TokenBudgetEnabled {
		t.Fatal(got, err)
	}
}

func TestDocumentBudgetValidationAndCopy(t *testing.T) {
	for _, b := range []types.DocumentBudget{{}, {MaxTokens: -1}, {MaxChunks: -1}, {RetainFraction: -.1}, {RetainFraction: 1.1}, {RetainFraction: math.NaN()}, {RetainFraction: math.Inf(1)}, {MaxChunks: 1, Strategy: "invented"}, {MaxTokens: 1}, {RetainFraction: 1}} {
		if _, err := New(types.Config{MaxTokens: 10, DocumentBudget: &b}, nil).Chop(context.Background(), nil); err == nil {
			t.Fatal(b)
		}
	}
	b := &types.DocumentBudget{MaxChunks: 1}
	c := New(types.Config{MaxTokens: 20, DocumentBudget: b}, nil)
	b.MaxChunks = 100
	r, err := c.ChopWithReport(context.Background(), budgetSections())
	if err != nil || len(r.Chunks) != 1 {
		t.Fatal(r, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := c.ChopWithReport(ctx, nil); err != context.Canceled {
		t.Fatal(err)
	}
}

func TestDocumentContextAndNoFit(t *testing.T) {
	sec := types.Section{Title: "H", Path: []string{"H"}, Blocks: []types.Block{{Type: types.BlockText, Content: "Short description.", LineStart: 2, LineEnd: 2}, {Type: types.BlockCode, Content: "HTTP HTTP HTTP HTTP HTTP", LineStart: 4, LineEnd: 4}}}
	cfg := types.Config{MaxTokens: 100, DocumentBudget: &types.DocumentBudget{MaxChunks: 1, Strategy: types.SelectTerms}}
	a, err := New(cfg, nil).Chop(context.Background(), []types.Section{sec})
	if err != nil || len(a) != 1 || a[0].BlockType != types.BlockCode {
		t.Fatal(a, err)
	}
	cfg.DocumentBudget.Strategy = types.SelectTermsContext
	b, err := New(cfg, nil).Chop(context.Background(), []types.Section{sec})
	if err != nil || len(b) != 1 || b[0].BlockType != types.BlockText {
		t.Fatal(b, err)
	}
	// A new heading with the same title is still a boundary.
	separate := []types.Section{{Title: "H", Path: sec.Path, Blocks: sec.Blocks[:1]}, {Title: "H", Path: sec.Path, Blocks: sec.Blocks[1:]}}
	b, err = New(cfg, nil).Chop(context.Background(), separate)
	if err != nil || b[0].BlockType != types.BlockCode {
		t.Fatal(b, err)
	}
	cfg.EmbeddingBudget = byteEmbeddingBudget(100)
	cfg.DocumentBudget = &types.DocumentBudget{MaxTokens: 1}
	r, err := New(cfg, nil).ChopWithReport(context.Background(), []types.Section{sec})
	if err != nil || len(r.Chunks) != 0 || !r.Compression.NoCandidateFits {
		t.Fatal(r, err)
	}
	// Selection must not hide the caller's strict per-chunk oversize policy.
	cfg.MaxTokens = 1
	cfg.OversizePolicy = types.OversizeError
	if _, err := New(cfg, nil).Chop(context.Background(), []types.Section{sec}); err == nil {
		t.Fatal("oversize hidden")
	}
}

func TestDocumentTermsAndSourceFrequencies(t *testing.T) {
	terms := lexicalTerms("The HTTP request reaches Alice Smith. HTTP request uses Alpha Gateway. Alpha Gateway works. Не отключайте TLS.")
	if _, ok := terms["the"]; ok {
		t.Fatal("stop word scored")
	}
	if !terms["http"].abbreviation || !terms["alice"].proper || terms["http request"].frequency != 2 || terms["alpha gateway"].frequency != 2 {
		t.Fatal(terms)
	}
	if _, ok := lexicalTerms("alpha and beta")["alpha beta"]; ok {
		t.Fatal("invented phrase across stop word")
	}
	if _, ok := lexicalTerms("alpha. beta")["alpha beta"]; ok {
		t.Fatal("invented phrase across sentence")
	}
	sec := budgetSections()
	chunks, err := New(types.Config{MaxTokens: 20}, nil).Chop(context.Background(), sec)
	if err != nil {
		t.Fatal(err)
	}
	_, a := documentFeatures(chunks, sec)
	_, b := documentFeatures(append(chunks, chunks...), sec)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("repeated output inflated source weights")
	}
	cfg := types.Config{MaxTokens: 100, DocumentBudget: &types.DocumentBudget{MaxChunks: 10}}
	r, err := New(cfg, nil).ChopWithReport(context.Background(), sec)
	if err != nil || !strings.Contains(r.Chunks[2].Text, "Do not disable") {
		t.Fatal("negation changed", err)
	}
}
