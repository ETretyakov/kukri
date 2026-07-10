# kukri

A Go library and CLI for **structure-aware chunking** of markdown documents. Kukri parses document structure (headers, tables, code blocks, lists), groups content into sections with breadcrumb paths, and merges adjacent blocks greedily by size — producing compact, structurally-coherent chunks with zero ML dependencies.

Embedding happens outside the chunker, in your retrieval system. Kukri provides a `Scorer` interface as an extension point if you want to influence merge decisions with similarity, but the default (size-based merge with `nil` scorer) already outperforms pure-semantic and fixed-token alternatives.

---

## Why kukri?

Benchmark on 886 QA pairs from 248 IAEA technical publications (95% CI ≈ ±1.6 pp). Fair comparison: same model (`BAAI/bge-small-en-v1.5`), same `max-tokens=128`:

| Strategy          | Coverage↑ | Hit@1↑    | Hit@3↑    | Hit@5↑    | AvgTok | Chunks  |
|-------------------|-----------|-----------|-----------|-----------|--------|---------|
| **kukri**         | **95.4%** | **56.0%** | **75.1%** | **80.9%** | 70.5   | 149 357 |
| markdown splitter | 79.6%     | 46.1%     | 59.6%     | 64.2%     | 41.2   | 244 037 |

- **Coverage** — % of gold answer spans contained in at least one chunk (higher = less information loss)
- **Hit@k** — % of questions where the correct chunk appears in the top-k by cosine similarity to the question embedding (the real RAG metric)

Kukri leads across every metric: **+15.8 pp Coverage**, **+9.9 pp Hit@1**, **+16.7 pp Hit@5**, with **40% fewer chunks**.

---

## How it works

The pipeline has three stages:

```
Markdown text
    │
    ▼
[1] Parser        — line-by-line scan → typed blocks
                    (text, table, code, diagram, list, image, header)
    │
    ▼
[2] Segmenter     — group blocks under headings → sections with breadcrumb paths
    │
    ▼
[3] Chunker       — greedy block merge within each section
                    adjacent blocks merge while combined tokens ≤ max-tokens
                    (and optional: while similarity ≥ threshold via Scorer)
                    oversized single blocks split at sentence boundaries
                    tables split row-by-row keeping header
                    code/diagrams kept whole
    │
    ▼
[]Chunk{ID, PrevID, NextID, Text, EmbedText, Section, SectionPath, BlockType, SourceLines, TokenCount, Source}
```

Each chunk carries two text fields:
- **`Text`** — the display content (what you store and show to users)
- **`EmbedText`** — the text to embed (equals `Text` by default; with `-prepend-section-path` it prepends the breadcrumb path `"Ch2 > Safety\n\nText..."`)

---

## Installation

```bash
go get github.com/ETretyakov/kukri
```

No CGo, no system libraries, zero dependencies.

---

## CLI

### Single file

```bash
go run ./cmd/chunker \
  -input   document.md \
  -output  chunks.json \
  -max-tokens 128
```

Output is a JSON object with a `chunks` array.

### Batch mode (corpus)

Reads JSONL from `-input` (`{"id":"...","text":"..."}` per line), writes one JSONL line per document to `-output`:

```bash
go run ./cmd/chunker \
  -batch \
  -input   corpus.jsonl \
  -output  chunks.jsonl \
  -max-tokens 128
```

Output format: `{"id":"...","chunks":[...]}` per line.

### Flags

| Flag                    | Default | Description                                                        |
|-------------------------|---------|--------------------------------------------------------------------|
| `-input`                |         | Input markdown file (single) or JSONL corpus (batch)               |
| `-output`               | stdout  | Output path                                                        |
| `-max-tokens`           | `128`   | Max words per chunk (word count, not BPE tokens)                   |
| `-threshold`            | `0`     | Min scorer similarity to merge adjacent blocks (0 = size-only)     |
| `-prepend-section-path` | `false` | Prepend breadcrumb path to `embed_text` (useful for BGE/E5 models) |
| `-batch`                | `false` | Enable batch JSONL mode                                            |

> **Note on token counting.** kukri counts whitespace-delimited words. English text averages ~1.3–1.5 BPE sub-word tokens per word, so `max-tokens=128` corresponds to roughly **165–190 actual LLM tokens**.

---

## Go API

### Quick start

```go
import "github.com/ETretyakov/kukri"

func chunkDocument(ctx context.Context, text string) ([]kukri.Chunk, error) {
	c := kukri.New(kukri.Config{
		MaxTokens: 128,
		Threshold: 0,
		Source:    "my-doc.md",
	}, nil) // nil scorer → pure size-based merge

	return c.Chop(ctx, text)
}
```

### With a custom scorer

```go
import (
    "github.com/ETretyakov/kukri"
    "github.com/ETretyakov/kukri/scorer"
)

// Implement scorer.Scorer to influence merge decisions:
//
//   type Scorer interface {
//       Score(ctx context.Context, texts []string) ([]float32, error)
//   }
//
// Score receives N block texts and must return N-1 pairwise similarity scores.
// Blocks i and i+1 are merged when sims[i] >= threshold AND combined tokens <= MaxTokens.

c := kukri.New(kukri.Config{MaxTokens: 256, Threshold: 0.6}, myScorer)
```

### Individual pipeline stages

```go
import (
    "github.com/ETretyakov/kukri/parser"
    "github.com/ETretyakov/kukri/segmenter"
    "github.com/ETretyakov/kukri/chunker"
    "github.com/ETretyakov/kukri/types"
)

blocks   := parser.Parse(markdownText)
sections := segmenter.Segment(blocks)

c := chunker.New(types.Config{MaxTokens: 128, Threshold: 0}, nil)
chunks, _ := c.Chop(ctx, sections)
```

### Chunk schema

```go
type Chunk struct {
	ID          string    // "introduction_0", "methods_3", …
	PrevID      string    // ID of preceding chunk (empty for first)
	NextID      string    // ID of following chunk (empty for last)
	Text        string    // display content
	EmbedText   string    // text to embed (may include section-path prefix)
	Section     string    // nearest heading title
	SectionPath []string  // breadcrumb: ["Chapter 1", "1.2 Methods"]
	BlockType   BlockType // "text" | "table" | "code" | "diagram" | "list" | "image"
	SourceLines [2]int    // [start, end] line numbers in source markdown
	TokenCount  int       // word count of Text
	Source      string    // source filename / doc ID
}
```

---

## Benchmarks

### Dataset

886 QA pairs generated by `claude-haiku-4-5` from 248 IAEA technical publications.  
Each pair: a verbatim answer span extracted from one document + a question about it.

### Full results

All rows at `max-tokens=128`. `BGE-small` = `BAAI/bge-small-en-v1.5`, `MiniLM` = `sentence-transformers/all-MiniLM-L6-v2`.

| config                           | Coverage↑ | Hit@1↑    | Hit@3↑    | Hit@5↑    | AvgTok | Chunks  |
|----------------------------------|-----------|-----------|-----------|-----------|--------|---------|
| **kukri · BGE-small**            | **95.4%** | **56.0%** | **75.1%** | **80.9%** | 70.5   | 149 357 |
| kukri · BGE-small + section-path | 95.4%     | 56.5%     | 74.5%     | 81.5%     | 70.5   | 149 357 |
| markdown splitter · BGE-small    | 79.6%     | 46.1%     | 59.6%     | 64.2%     | 41.2   | 244 037 |
| kukri · MiniLM                   | 95.4%     | 39.3%     | 60.3%     | 67.4%     | 70.5   | 149 357 |
| baseline (no merge) · MiniLM     | 94.9%     | 36.6%     | 56.7%     | 66.8%     | 53.3   | 183 113 |
| markdown splitter · MiniLM       | 99.6%     | 37.8%     | 60.1%     | 69.5%     | 134.0  | 75 396  |

### Running the benchmark

```bash
cd kukri

# Set up Python env
python3 -m venv bench/.venv
bench/.venv/bin/pip install -r bench/requirements.txt

# Build corpus from parsed markdown files
bench/.venv/bin/python3 bench/make_corpus.py \
  --input  ../parsing/output \
  --output corpus.jsonl

# Generate gold QA pairs (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=sk-... bench/.venv/bin/python3 bench/generate_gold.py \
  --input  corpus.jsonl \
  --output bench/gold.jsonl \
  --pairs-per-doc 5

# Sweep kukri thresholds and measure Hit@k (with markdown splitter baseline)
KUKRI_BIN=./kukri bench/.venv/bin/python3 bench/sweep_threshold.py \
  --gold         bench/gold.jsonl \
  --corpus       corpus.jsonl \
  --max-tokens   128 \
  --thresholds   0.0 \
  --embed-model  BAAI/bge-small-en-v1.5 \
  --baseline
```

Build the binary first: `go build -o kukri ./cmd/chunker`

---

## Design decisions

### Why there is no embedding model inside the chunker

The original design embedded sentences with `all-MiniLM-L6-v2` via hugot (pure-Go ONNX, CPU-only) and used cosine similarity to merge adjacent sentences. Three problems surfaced:

1. **The fast path killed semantic merging.** IAEA documents average ~53 tokens per paragraph block. The code path was: if the block fits in `MaxTokens`, emit it as-is and skip the embedder. With `MaxTokens=512`, 100% of blocks hit the fast path — the embedder was never called.

2. **hugot on CPU is 10–20× slower than GPU inference.** Even if the embedder ran, chunking 248 documents took minutes on CPU. The retrieval system already runs a GPU-accelerated embedder for indexing; running a second, slower embedder in the chunker adds latency and model version skew.

3. **TF-IDF-based similarity didn't improve over size-based merge.** A sweep across threshold values (0.0 to 0.8) with TF-IDF cosine similarity showed identical Hit@k to `threshold=0` across all configurations. Adjacent IAEA paragraphs are already topically coherent; similarity-based gating just prevented valid merges at high thresholds.

**Conclusion:** embedding belongs in the retrieval system, not the chunker. The chunker's job is structure preservation and size control. The `Scorer` interface remains as an extension point for future experimentation.

### Why BGE-small outperforms MiniLM by +16.7 pp Hit@1

`BAAI/bge-small-en-v1.5` is trained with hard negative mining for asymmetric retrieval (question → passage). It uses a query prefix to distinguish query-side embeddings from passage-side embeddings. `all-MiniLM-L6-v2` is trained for symmetric sentence similarity.

For RAG the retrieval task is asymmetric: a user question is matched against document passages. BGE's training objective aligns directly with this use case; MiniLM's does not. The difference is +16.7 pp Hit@1 and +14.8 pp Hit@3 on identical chunks.

### Why the section-path prefix is opt-in

`-prepend-section-path` prepends the breadcrumb (`"Ch2 > Safety\n\n"`) to `EmbedText` before encoding. This works for models trained to incorporate document structure (E5, BGE with instructed variants). For `all-MiniLM-L6-v2` it hurts: IAEA documents have generic section names ("Introduction", "Background", "Conclusions") that add noise without adding discriminative signal.

### PrevID / NextID as context hooks

Every chunk carries `PrevID` and `NextID`. Retrieval systems can use these to expand the retrieved context window at query time ("sentence-window retrieval") without storing the expansion in the chunk itself. This keeps chunks compact while allowing the reader to see surrounding context.

---

## Repository layout

```
kukri/
├── kukri.go            — root package: Chunker, Scorer type alias
├── types/              — shared types: Chunk, Config, Block, Section, BlockType
├── parser/             — markdown → []Block
├── segmenter/          — []Block → []Section (header grouping + breadcrumbs)
├── scorer/             — Scorer interface (extension point; no built-in implementations)
├── chunker/            — []Section → []Chunk (greedy block merge + type-aware splitting)
├── cmd/
│   └── chunker/        — CLI entry point (single-file and batch modes)
└── bench/
    ├── sweep_threshold.py — Hit@k + Coverage benchmark sweep
    ├── generate_gold.py   — LLM-based gold QA pair generator
    ├── make_corpus.py     — corpus.jsonl builder from markdown files
    └── requirements.txt
```

---

## Conclusion

Kukri outperforms the markdown splitter on every metric when the comparison is fair (same model, same `max-tokens`):

| metric   | kukri   | markdown splitter | delta        |
|----------|---------|-------------------|--------------|
| Coverage | 95.4%   | 79.6%             | **+15.8 pp** |
| Hit@1    | 56.0%   | 46.1%             | **+9.9 pp**  |
| Hit@3    | 75.1%   | 59.6%             | **+15.5 pp** |
| Hit@5    | 80.9%   | 64.2%             | **+16.7 pp** |
| Chunks   | 149 357 | 244 037           | **−39%**     |

The coverage gap tells the core story: `RecursiveCharacterTextSplitter` cuts on character count and routinely splits answer spans across chunk boundaries. Kukri respects document structure — paragraphs, tables, lists — so spans stay intact. This also produces 40% fewer, more cohesive chunks, which shrinks the index and speeds up retrieval.

Two independent levers control retrieval quality:

1. **Chunking strategy** (this library) — determines Coverage and chunk coherence. Kukri's block-aware merge gives +15.8 pp Coverage over character splitting.
2. **Embedding model** (your retrieval system) — determines how well queries match chunks. Switching from MiniLM to BGE-small gives +16.7 pp Hit@1 on the same chunks.

Both levers matter. A good chunker with a weak model, or a strong model with character-split chunks, both leave significant performance on the table.

---

## License

MIT
