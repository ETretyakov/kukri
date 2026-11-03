package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/ETretyakov/kukri/chunker"
	"github.com/ETretyakov/kukri/parser"
	"github.com/ETretyakov/kukri/scorer"
	"github.com/ETretyakov/kukri/segmenter"
	"github.com/ETretyakov/kukri/tokenbudget"
	"github.com/ETretyakov/kukri/types"
)

func main() {
	input := flag.String("input", "", "Markdown file, or JSONL in batch mode; - for stdin")
	output := flag.String("output", "", "output file (default stdout)")
	maxWords := 128
	flag.IntVar(&maxWords, "max-words", 128, "chunk Text word budget; 0 requires an embedding budget")
	flag.IntVar(&maxWords, "max-tokens", 128, "deprecated alias of -max-words; these are words, not model tokens")
	mode := flag.String("merge", "size", "text merging: none, size, similarity")
	threshold := flag.Float64("threshold", 0, "similarity threshold in [0,1]; requires -merge similarity")
	scoring := flag.String("scorer", "", "similarity backend: tfidf (required for -merge similarity)")
	oversize := flag.String("oversize", "keep", "indivisible elements over budget: keep (mark), error")
	batch := flag.Bool("batch", false, "read JSONL {id,text}; any invalid record fails the run")
	prefix := flag.Bool("prepend-section-path", false, "prepend heading path to EmbedText; included in embedding token budget when enabled")
	repack := flag.Bool("repack-text", false, "split long paragraphs before packing fragments; requires -merge size")
	tokenizerPath := flag.String("embedding-tokenizer", "", "local BERT WordPiece tokenizer.json; requires -max-embedding-tokens")
	embeddingLimit := flag.Int("max-embedding-tokens", 0, "complete EmbedText limit including headings and special tokens")
	compactTables := flag.Bool("compact-table-embedding", false, "compact table Markdown in EmbedText only")
	docTokens := flag.Int("document-max-tokens", 0, "total complete EmbedText token budget per document; requires embedding tokenizer")
	docChunks := flag.Int("document-max-chunks", 0, "maximum retained chunks per document (lossy selection)")
	docFraction := flag.Float64("document-retain", 0, "retain at most this fraction of original embedding tokens, in (0,1]")
	docStrategy := flag.String("document-selection", "", "document selection: uniform, terms, terms-context (default)")
	flag.Parse()
	if flag.NArg() != 0 {
		fatalf("unexpected positional arguments: %v", flag.Args())
	}
	var scr scorer.Scorer
	switch *scoring {
	case "":
	case "tfidf":
		scr = scorer.TFIDF{}
	default:
		fatalf("unknown scorer %q", *scoring)
	}
	cfg := types.Config{MaxTokens: maxWords, MergeMode: types.MergeMode(*mode), Threshold: float32(*threshold), OversizePolicy: types.OversizePolicy(*oversize), PrependSectionPath: *prefix, RepackText: *repack, CompactTableEmbedding: *compactTables}
	if *tokenizerPath != "" || *embeddingLimit != 0 {
		if *tokenizerPath == "" || *embeddingLimit <= 0 {
			fatalf("-embedding-tokenizer and positive -max-embedding-tokens are required together")
		}
		counter, err := tokenbudget.FromFile(*tokenizerPath)
		if err != nil {
			fatalf("tokenizer: %v", err)
		}
		cfg.EmbeddingBudget = counter.Budget(*embeddingLimit)
	}
	if *docTokens != 0 || *docChunks != 0 || *docFraction != 0 || *docStrategy != "" {
		cfg.DocumentBudget = &types.DocumentBudget{MaxTokens: *docTokens, MaxChunks: *docChunks, RetainFraction: *docFraction, Strategy: types.SelectionStrategy(*docStrategy)}
	}
	if err := chunker.New(cfg, scr).Validate(); err != nil {
		fatalf("configuration: %v", err)
	}
	if *input == "" && !*batch {
		fatalf("-input is required in single-file mode")
	}
	in := os.Stdin
	if *input != "" && *input != "-" {
		var err error
		in, err = os.Open(*input)
		if err != nil {
			fatalf("open input: %v", err)
		}
		defer in.Close()
	}
	out := os.Stdout
	if *output != "" && *output != "-" {
		// Prevent truncating the input, including through a symlink/hard link.
		if a, err := in.Stat(); err == nil {
			if b, err := os.Stat(*output); err == nil && os.SameFile(a, b) {
				fatalf("input and output must be different files")
			}
		}
		if *tokenizerPath != "" {
			if a, err := os.Stat(*tokenizerPath); err == nil {
				if b, err := os.Stat(*output); err == nil && os.SameFile(a, b) {
					fatalf("tokenizer and output must be different files")
				}
			}
		}
		var err error
		out, err = os.Create(*output)
		if err != nil {
			fatalf("create output: %v", err)
		}
		defer out.Close()
	}
	ctx := context.Background()
	if *batch {
		if err := runBatch(ctx, cfg, scr, in, out); err != nil {
			fatalf("batch: %v", err)
		}
	} else {
		raw, err := io.ReadAll(in)
		if err != nil {
			fatalf("read input: %v", err)
		}
		cfg.Source = *input
		result, err := chopReport(ctx, cfg, scr, string(raw))
		if err != nil {
			fatalf("chop: %v", err)
		}
		enc := json.NewEncoder(out)
		enc.SetIndent("", "  ")
		if err := enc.Encode(result); err != nil {
			fatalf("encode: %v", err)
		}
	}
	if out != os.Stdout {
		if err := out.Close(); err != nil {
			fatalf("close output: %v", err)
		}
	}
}

// A failed batch may leave partial output; consumers must check the exit code.
// No malformed document is silently skipped or converted to an empty result.
func runBatch(ctx context.Context, cfg types.Config, scr scorer.Scorer, in io.Reader, out io.Writer) error {
	enc := json.NewEncoder(out)
	scanner := bufio.NewScanner(in)
	scanner.Buffer(make([]byte, 64*1024), 64*1024*1024)
	seen := map[string]bool{}
	line := 0
	for scanner.Scan() {
		line++
		var doc struct {
			ID   string  `json:"id"`
			Text *string `json:"text"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &doc); err != nil {
			return fmt.Errorf("line %d: %w", line, err)
		}
		if doc.ID == "" || doc.Text == nil {
			return fmt.Errorf("line %d: nonempty id and text field required", line)
		}
		if seen[doc.ID] {
			return fmt.Errorf("line %d: duplicate id %q", line, doc.ID)
		}
		seen[doc.ID] = true
		cfg.Source = doc.ID
		result, err := chopReport(ctx, cfg, scr, *doc.Text)
		if err != nil {
			return fmt.Errorf("document %q: %w", doc.ID, err)
		}
		if err := enc.Encode(struct {
			ID          string                   `json:"id"`
			Chunks      []types.Chunk            `json:"chunks"`
			Compression *types.CompressionReport `json:"compression,omitempty"`
		}{doc.ID, result.Chunks, result.Compression}); err != nil {
			return err
		}
	}
	return scanner.Err()
}

func chop(ctx context.Context, cfg types.Config, scr scorer.Scorer, text string) ([]types.Chunk, error) {
	return chunker.New(cfg, scr).Chop(ctx, segmenter.Segment(parser.Parse(text)))
}

func chopReport(ctx context.Context, cfg types.Config, scr scorer.Scorer, text string) (types.Result, error) {
	return chunker.New(cfg, scr).ChopWithReport(ctx, segmenter.Segment(parser.Parse(text)))
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "error: "+format+"\n", args...)
	os.Exit(1)
}
