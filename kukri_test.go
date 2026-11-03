package kukri

import (
	"context"
	"testing"
)

func FuzzChop(f *testing.F) {
	for _, s := range []string{"####### ordinary text", "# Header\n\nbody", "```\n# code\n```", "| a |\n|---|\n| b |", "~~~\n~~~", "- item\n  nested"} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, text string) {
		chunks, err := New(Config{MaxTokens: 32}, nil).Chop(context.Background(), text)
		if err != nil {
			t.Fatal(err)
		}
		for i, ch := range chunks {
			if ch.Size > 32 && !ch.Oversized {
				t.Fatal("unreported oversize")
			}
			if i > 0 && ch.PrevID != chunks[i-1].ID {
				t.Fatal("broken chain")
			}
		}
	})
}
