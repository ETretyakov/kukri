package main

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/ETretyakov/kukri/types"
)

func TestBatchRejectsInvalidRecords(t *testing.T) {
	for _, input := range []string{"not json", `{"id":"a"}`, `{"id":"","text":"ok"}`, "{\"id\":\"a\",\"text\":\"ok\"}\n{\"id\":\"a\",\"text\":\"duplicate\"}"} {
		var out bytes.Buffer
		if err := runBatch(context.Background(), types.Config{MaxTokens: 128}, nil, strings.NewReader(input), &out); err == nil {
			t.Fatalf("accepted %s", input)
		}
	}
}

func TestSevenHashesAndEmptyDocument(t *testing.T) {
	var out bytes.Buffer
	input := "{\"id\":\"a\",\"text\":\"####### ordinary text\"}\n{\"id\":\"b\",\"text\":\"\"}"
	if err := runBatch(context.Background(), types.Config{MaxTokens: 128}, nil, strings.NewReader(input), &out); err != nil {
		t.Fatal(err)
	}
	dec := json.NewDecoder(&out)
	var first, second struct {
		ID     string
		Chunks []types.Chunk
	}
	if err := dec.Decode(&first); err != nil {
		t.Fatal(err)
	}
	if err := dec.Decode(&second); err != nil {
		t.Fatal(err)
	}
	if first.Chunks[0].Text != "####### ordinary text" || second.ID != "b" || second.Chunks == nil || len(second.Chunks) != 0 {
		t.Fatal(first, second)
	}
}
