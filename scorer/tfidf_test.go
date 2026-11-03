package scorer

import (
	"context"
	"reflect"
	"testing"
)

func TestTFIDF(t *testing.T) {
	texts := []string{"Alpha alpha beta", "Alpha alpha beta", "gamma delta", ""}
	got, err := (TFIDF{}).Score(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[0] < 0.99999 || got[1] != 0 || got[2] != 0 {
		t.Fatal(got)
	}
	for i := 0; i < 20; i++ {
		again, _ := (TFIDF{}).Score(context.Background(), texts)
		if !reflect.DeepEqual(got, again) {
			t.Fatal("nondeterministic")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := (TFIDF{}).Score(ctx, texts); err == nil {
		t.Fatal("ignored cancellation")
	}
}
