package kukri

import (
	"context"
	"testing"
)

func TestDocumentBudgetPublicAPI(t *testing.T) {
	c := New(Config{MaxTokens: 20, DocumentBudget: &DocumentBudget{MaxChunks: 1, Strategy: SelectTerms}}, nil)
	r, err := c.ChopWithReport(context.Background(), "# First\nHTTP requests use HTTP.\n\n# Second\nAnother topic.")
	if err != nil || len(r.Chunks) != 1 || r.Compression == nil || r.Compression.OmittedChunks != 1 {
		t.Fatal(r, err)
	}
}
