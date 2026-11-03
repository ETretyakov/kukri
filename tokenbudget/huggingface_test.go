package tokenbudget

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func TestHuggingFaceReferenceCounts(t *testing.T) {
	c, err := FromFile("testdata/bert.json")
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("testdata/counts.json")
	if err != nil {
		t.Fatal(err)
	}
	var refs struct {
		Cases []struct {
			Text  string
			Count int
		}
	}
	if err = json.Unmarshal(raw, &refs); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for _, ref := range refs.Cases {
		wg.Add(1)
		go func() {
			defer wg.Done()
			n, err := c.Count(ref.Text)
			if err != nil || n != ref.Count {
				t.Errorf("%q: %d %v; want %d", ref.Text, n, err, ref.Count)
			}
		}()
	}
	wg.Wait()
	n, _ := c.Count(strings.Repeat("hello ", 100))
	if n != 102 {
		t.Fatal("truncated/padded", n)
	}
	if _, err := c.Count(string([]byte{0xff})); err == nil {
		t.Fatal("invalid UTF8 accepted")
	}
}
func TestUnsupportedTokenizerFailsClosed(t *testing.T) {
	raw, _ := os.ReadFile("testdata/bert.json")
	for _, s := range []string{"null", "{}", strings.Replace(string(raw), `"WordPiece"`, `"BPE"`, 1), strings.Replace(string(raw), `"strip_accents": null`, `"strip_accents": false`, 1)} {
		p := filepath.Join(t.TempDir(), "tokenizer.json")
		if err := os.WriteFile(p, []byte(s), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := FromFile(p); err == nil {
			t.Fatalf("accepted unsupported tokenizer: %s", s)
		}
	}
}
