package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/ETretyakov/kukri/chunker"
	"github.com/ETretyakov/kukri/embedder"
	"github.com/ETretyakov/kukri/parser"
	"github.com/ETretyakov/kukri/segmenter"
	"github.com/ETretyakov/kukri/types"
)

func main() {
	input     := flag.String("input",      "",    "path to markdown file (single-file mode)")
	output    := flag.String("output",     "",    "path to output file (default: stdout)")
	modelPath := flag.String("model",      "",    "path to sentence-transformer model directory (required)")
	maxTokens := flag.Int("max-tokens",    512,   "soft upper limit for chunk size in tokens")
	threshold := flag.Float64("threshold", 0.7,   "minimum cosine similarity to keep sentences together [0,1]")
	batch     := flag.Bool("batch",        false, "batch mode: read JSONL {\"id\":\"...\",\"text\":\"...\"} from -input, write JSONL per doc")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "error: -model is required")
		flag.Usage()
		os.Exit(1)
	}

	ctx := context.Background()

	fmt.Fprintf(os.Stderr, "loading model from %s...\n", *modelPath)
	emb, err := embedder.New(ctx, *modelPath)
	if err != nil {
		fatalf("load embedder: %v", err)
	}
	defer emb.Close()

	cfg := types.Config{
		MaxTokens: *maxTokens,
		Threshold: float32(*threshold),
		ModelPath: *modelPath,
	}

	if *batch {
		runBatch(ctx, cfg, emb, *input, *output)
	} else {
		runSingle(ctx, cfg, emb, *input, *output)
	}
}

// runSingle processes one markdown file.
func runSingle(ctx context.Context, cfg types.Config, emb *embedder.Embedder, inputPath, outputPath string) {
	if inputPath == "" {
		fmt.Fprintln(os.Stderr, "error: -input is required in single-file mode")
		os.Exit(1)
	}

	raw, err := os.ReadFile(inputPath)
	if err != nil {
		fatalf("read input: %v", err)
	}

	cfg.Source = inputPath
	chunks, err := chop(ctx, cfg, emb, string(raw))
	if err != nil {
		fatalf("chop: %v", err)
	}

	out := openOutput(outputPath)
	defer out.Close()

	enc := json.NewEncoder(out)
	enc.SetIndent("", "  ")
	if err := enc.Encode(types.Result{Chunks: chunks}); err != nil {
		fatalf("encode: %v", err)
	}
	fmt.Fprintf(os.Stderr, "done: %d chunks\n", len(chunks))
}

// runBatch reads JSONL {"id":"...","text":"..."} and writes one JSONL line per doc.
// Output line format: {"id":"...","chunks":[...]}
// The embedding model is loaded once for the whole corpus.
func runBatch(ctx context.Context, cfg types.Config, emb *embedder.Embedder, inputPath, outputPath string) {
	var in *os.File
	if inputPath == "" || inputPath == "-" {
		in = os.Stdin
	} else {
		f, err := os.Open(inputPath)
		if err != nil {
			fatalf("open input: %v", err)
		}
		defer f.Close()
		in = f
	}

	out := openOutput(outputPath)
	defer out.Close()

	enc := json.NewEncoder(out)
	scanner := bufio.NewScanner(in)
	scanner.Buffer(make([]byte, 10*1024*1024), 10*1024*1024) // 10 MB per line

	type batchInput struct {
		ID   string `json:"id"`
		Text string `json:"text"`
	}
	type batchOutput struct {
		ID     string        `json:"id"`
		Chunks []types.Chunk `json:"chunks"`
	}

	total, docs := 0, 0
	for scanner.Scan() {
		var doc batchInput
		if err := json.Unmarshal(scanner.Bytes(), &doc); err != nil {
			fmt.Fprintf(os.Stderr, "skip malformed line: %v\n", err)
			continue
		}

		cfg.Source = doc.ID
		chunks, err := chop(ctx, cfg, emb, doc.Text)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error processing %s: %v\n", doc.ID, err)
			continue
		}

		if err := enc.Encode(batchOutput{ID: doc.ID, Chunks: chunks}); err != nil {
			fatalf("encode: %v", err)
		}
		total += len(chunks)
		docs++
	}
	if err := scanner.Err(); err != nil {
		fatalf("read input: %v", err)
	}
	fmt.Fprintf(os.Stderr, "done: %d docs → %d chunks\n", docs, total)
}

// chop runs the full parse→segment→chunk pipeline on a text.
func chop(ctx context.Context, cfg types.Config, emb *embedder.Embedder, text string) ([]types.Chunk, error) {
	c := chunker.New(cfg, emb)
	return c.Chop(ctx, segmenter.Segment(parser.Parse(text)))
}

func openOutput(path string) *os.File {
	if path == "" || path == "-" {
		return os.Stdout
	}
	f, err := os.Create(path)
	if err != nil {
		fatalf("create output: %v", err)
	}
	return f
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "error: "+format+"\n", args...)
	os.Exit(1)
}
