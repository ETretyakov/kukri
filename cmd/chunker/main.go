package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/ETretyakov/kukri/chunker"
	"github.com/ETretyakov/kukri/parser"
	"github.com/ETretyakov/kukri/segmenter"
	"github.com/ETretyakov/kukri/types"
)

func main() {
	input             := flag.String("input",               "",     "path to markdown file (single-file mode)")
	output            := flag.String("output",              "",     "path to output file (default: stdout)")
	maxTokens         := flag.Int("max-tokens",             128,    "soft upper limit for chunk size in tokens")
	threshold         := flag.Float64("threshold",          0,      "minimum similarity to merge adjacent blocks [0,1]; 0 = always merge up to max-tokens")
	batch             := flag.Bool("batch",                 false,  "batch mode: read JSONL {\"id\":\"...\",\"text\":\"...\"} from -input")
	prependSectionPath := flag.Bool("prepend-section-path", false,  "prepend section breadcrumb to embed_text (for BGE/E5 models)")
	flag.Parse()

	cfg := types.Config{
		MaxTokens:          *maxTokens,
		Threshold:          float32(*threshold),
		PrependSectionPath: *prependSectionPath,
	}

	ctx := context.Background()
	if *batch {
		runBatch(ctx, cfg, *input, *output)
	} else {
		runSingle(ctx, cfg, *input, *output)
	}
}

func runSingle(ctx context.Context, cfg types.Config, inputPath, outputPath string) {
	if inputPath == "" {
		fmt.Fprintln(os.Stderr, "error: -input is required in single-file mode")
		os.Exit(1)
	}
	raw, err := os.ReadFile(inputPath)
	if err != nil {
		fatalf("read input: %v", err)
	}
	cfg.Source = inputPath
	chunks, err := chop(ctx, cfg, string(raw))
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

func runBatch(ctx context.Context, cfg types.Config, inputPath, outputPath string) {
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
	scanner.Buffer(make([]byte, 10*1024*1024), 10*1024*1024)

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
		chunks, err := chop(ctx, cfg, doc.Text)
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

func chop(ctx context.Context, cfg types.Config, text string) ([]types.Chunk, error) {
	c := chunker.New(cfg, nil)
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
