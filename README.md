# kukri

A Go library and CLI for **hybrid markdown-aware + semantic chunking** of documents. Kukri understands your document's structure before it touches the text — respecting headers, tables, code blocks, and diagrams — then applies cosine-similarity-based semantic grouping to keep related sentences together.

---

## Why kukri?

Most chunkers treat a document as a flat stream of tokens. Kukri treats it as a structured artifact:

| Strategy | Coverage↑ | Avg tokens | Notes |
|---|---|---|---|
| **kukri** | **93.6%** | **53** | structural + semantic |
| semantic | 88.4% | 61 | semantic only, no structure |
| fixed_token | 60.6% | 62 | splits blindly |
| sentence | 98.9% | 212 | large chunks, poor precision |
| recursive | 99.6% | 185 | large chunks, poor precision |

**Coverage** = % of gold answer spans fully contained within a single chunk (measured on 886 QA pairs from 248 IAEA technical documents, 95% CI ≈ ±1.6 pp, see [Benchmarks](#benchmarks)).

Sentence and recursive splitters win on coverage only because their chunks are 3–4× larger. At comparable chunk sizes, kukri delivers **+36 pp over fixed_token** and **+4 pp over pure semantic** — while producing precise, retrieval-friendly chunks.

---

## How it works

The pipeline has four stages:

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
[3] Embedder      — sentence-level ONNX inference (all-MiniLM-L6-v2, CPU)
    │
    ▼
[4] Chunker       — greedy semantic grouping within sections
                    (tables split row-by-row keeping header,
                     code/diagrams kept whole, IDs assigned and linked)
    │
    ▼
[]Chunk{ID, PrevID, NextID, Text, Section, SectionPath, BlockType, SourceLines, TokenCount, Source}
```

Each block type is chunked differently:
- **text** — sentences are embedded; adjacent sentences are merged while cosine similarity ≥ `threshold` and combined token count ≤ `max-tokens`
- **table** — fits in one chunk if small; otherwise split row-by-row with the header repeated in every part
- **list** — split on top-level item boundaries
- **code / diagram / image** — always emitted as a single chunk regardless of size

---

## Installation

### Prerequisites

- Go 1.22+
- No CGo, no system libraries — kukri uses [hugot](https://github.com/knights-analytics/hugot) with a pure-Go ONNX backend (GoMLX)

### Get the module

```bash
go get github.com/ETretyakov/kukri
```

### Download the embedding model

```bash
go run ./cmd/download -model KnightsAnalytics/all-MiniLM-L6-v2 -dest ./models
```

This downloads `all-MiniLM-L6-v2` (384-dim, CPU-friendly) from HuggingFace into `./models/`.

---

## CLI

### Single file

```bash
go run ./cmd/chunker \
  -input   document.md \
  -model   ./models/KnightsAnalytics_all-MiniLM-L6-v2 \
  -output  chunks.json \
  -max-tokens 512 \
  -threshold  0.7
```

Output is a JSON object with a `chunks` array.

### Batch mode (corpus)

Reads JSONL from `-input` (`{"id":"...","text":"..."}` per line), writes one JSONL line per document to `-output`. The model is loaded once for the entire corpus.

```bash
go run ./cmd/chunker \
  -batch \
  -input   corpus.jsonl \
  -output  chunks.jsonl \
  -model   ./models/KnightsAnalytics_all-MiniLM-L6-v2 \
  -max-tokens 512 \
  -threshold  0.7
```

Output format: `{"id":"...","chunks":[...]}` per line.

### Flags

| Flag | Default | Description |
|---|---|---|
| `-input` | | Input markdown file (single) or JSONL corpus (batch) |
| `-output` | stdout | Output path |
| `-model` | | Path to ONNX model directory **(required)** |
| `-max-tokens` | `512` | Soft upper limit per chunk (word count) |
| `-threshold` | `0.7` | Min cosine similarity to merge adjacent sentences |
| `-batch` | `false` | Enable batch JSONL mode |

---

## Go API

### Quick start

```go
import "github.com/ETretyakov/kukri"

func chunkDocument(ctx context.Context, modelPath, text string) ([]kukri.Chunk, error) {
    emb, err := kukri.NewEmbedder(ctx, modelPath)
    if err != nil {
        return nil, err
    }
    defer emb.Close()

    c := kukri.New(kukri.Config{
        MaxTokens: 512,
        Threshold: 0.7,
        ModelPath: modelPath,
        Source:    "my-doc.md",
    }, emb)

    return c.Chop(ctx, text)
}
```

### Advanced: individual pipeline stages

```go
import (
    "github.com/ETretyakov/kukri/parser"
    "github.com/ETretyakov/kukri/segmenter"
    "github.com/ETretyakov/kukri/embedder"
    "github.com/ETretyakov/kukri/chunker"
    "github.com/ETretyakov/kukri/types"
)

blocks   := parser.Parse(markdownText)          // []types.Block
sections := segmenter.Segment(blocks)           // []types.Section

emb, _   := embedder.New(ctx, modelPath)
defer emb.Close()

c      := chunker.New(types.Config{MaxTokens: 512, Threshold: 0.7}, emb)
chunks, _ := c.Chop(ctx, sections)             // []types.Chunk
```

### Chunk schema

```go
type Chunk struct {
    ID          string    // "introduction_0", "methods_3", …
    PrevID      string    // ID of the preceding chunk (empty for first)
    NextID      string    // ID of the following chunk (empty for last)
    Text        string    // chunk content
    Section     string    // nearest heading title
    SectionPath []string  // breadcrumb: ["Chapter 1", "1.2 Methods"]
    BlockType   BlockType // "text" | "table" | "code" | "diagram" | "list" | "image"
    SourceLines [2]int    // [start, end] line numbers in the source markdown
    TokenCount  int       // word count of Text
    Source      string    // source filename / doc ID
}
```

---

## Benchmarks

### Answer Coverage Rate

Measures what % of gold answer spans (exact verbatim quotes from source documents) are fully contained within a single chunk. Higher is better; evaluated at similar chunk sizes.

**Dataset**: 886 QA pairs generated by `claude-haiku-4-5` from 248 IAEA technical publications (95% CI ≈ ±1.6 pp).

| Strategy | Coverage↑ | Avg tokens | Coverage/token↑ | Chunks |
|---|---|---|---|---|
| **kukri** | **93.6%** | **53** | **1.77** | 183 113 |
| semantic | 88.4% | 61 | 1.45 | 163 612 |
| fixed_token | 60.6% | 62 | 0.98 | 182 614 |
| sentence | 98.9% | 212 | 0.47 | 50 432 |
| recursive | 99.6% | 185 | 0.54 | 54 823 |

**Coverage/token** = coverage% ÷ avg_tokens × 100 — measures how efficiently a strategy packs answer spans into its chunks. Sentence and recursive score high on raw coverage only because their chunks are 3–4× larger; at equal chunk size budgets they drop below kukri.

### Running the coverage benchmark

```bash
# 1. Build corpus from parsed markdown files
bench/.venv/bin/python3 bench/make_corpus.py \
  --input  ../parsing/output \
  --output corpus.jsonl \
  --max-docs 50

# 2. Generate gold QA pairs (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=sk-... bench/.venv/bin/python3 bench/generate_gold.py \
  --input  corpus.jsonl \
  --output bench/gold.jsonl \
  --pairs-per-doc 5

# 3. Measure coverage across all strategies
bench/.venv/bin/python3 bench/coverage_bench.py \
  --gold   bench/gold.jsonl \
  --corpus corpus.jsonl
```

---

## Repository layout

```
kukri/
├── kukri.go            — root package: Chunker, NewEmbedder, type aliases
├── types/              — shared types: Chunk, Config, Block, Section, BlockType
├── parser/             — markdown → []Block
├── segmenter/          — []Block → []Section (header grouping + breadcrumbs)
├── embedder/           — ONNX sentence embeddings via hugot (pure-Go backend)
├── chunker/            — []Section → []Chunk (semantic + type-aware splitting)
├── cmd/
│   ├── chunker/        — CLI entry point (single-file and batch modes)
│   └── download/       — HuggingFace model downloader
├── bench/
│   ├── coverage_bench.py — Answer Coverage Rate benchmark
│   ├── generate_gold.py  — LLM-based gold QA pair generator
│   ├── make_corpus.py    — corpus.jsonl builder from markdown files
│   ├── metrics.py        — intrinsic metrics (coherence, contrast, boundary respect)
│   ├── run.sh            — end-to-end benchmark runner
│   └── requirements.txt
└── models/             — downloaded ONNX model weights (git-ignored)
```

---

## Configuration guide

| Parameter | Recommended range | Effect |
|---|---|---|
| `max-tokens` | 256–1024 | Larger → fewer, richer chunks; smaller → more precise retrieval |
| `threshold` | 0.6–0.85 | Lower → more aggressive merging; higher → finer splits |

For technical documents with dense, self-contained paragraphs, `max-tokens=512 threshold=0.7` works well. For conversational or narrative text, try `threshold=0.65`.

> **Note on token counting.** kukri counts whitespace-delimited words, not BPE sub-word tokens. English text averages ~1.3–1.5 BPE tokens per word, so `max-tokens=512` corresponds to roughly **665–780 actual LLM tokens**. If your downstream model has a strict context window, set `max-tokens` to `target_bpe_tokens / 1.4`.

---

## License

MIT
