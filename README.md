# kukri

Kukri is a Go library for structure-aware chunking of Markdown, intended for retrieval pipelines. It parses typed blocks, groups them by heading hierarchy, and packs text within sections. The core requires no embedding model or external runtime.

An optional document budget performs extractive compression before indexing: it
selects whole chunks under total token and chunk limits using fixed lexical
weights and local context rules. Compression requires no neural-model inference.
Choose a budget by measuring retrieval quality on the documents and queries
used by your application.

## Library

```go
c := kukri.New(kukri.Config{
    MaxTokens: 128, // historical field name; the default unit is WORDS
    MergeMode: kukri.MergeSize,
}, nil)
chunks, err := c.Chop(ctx, markdownText)
```

Always check `err`. Configuration errors are reported by `Chop`, including on empty input.

Three explicit merge modes are available:

| Mode | Behavior |
| --- | --- |
| `MergeSize` (default) | Merge adjacent text blocks while the complete candidate fits the budget |
| `MergeNone` | Do not merge text blocks; oversized paragraphs still undergo splitting |
| `MergeSimilarity` | Require a Scorer and a threshold in `[0,1]`, as well as the size constraint |

Merging never crosses a heading or a non-text block. The budget applies to the complete accumulated group, not just the last pair. A Scorer receives one **contiguous text run** at a time, not all text scattered through a section. It must return exactly `len(texts)-1` finite scores in `[-1,1]`.

```go
import "github.com/ETretyakov/kukri/scorer"

c := kukri.New(kukri.Config{
    MaxTokens: 128,
    MergeMode: kukri.MergeSimilarity,
    Threshold: 0.4,
}, scorer.TFIDF{})
```

The built-in TF-IDF reference uses raw term frequency and smooth IDF `log((1+n)/(1+df))+1`, fitted per contiguous text run. Terms are lowercase Unicode letter/digit sequences; there is no stemming or stopword removal. This is a lexical comparator, not an embedding model. Its usefulness must be measured on the intended corpus.

### Size and indivisible elements

The default counter is `len(strings.Fields(text))`. `MaxTokens` and JSON `token_count` retain their historical names; they do **not** denote a model's BPE/WordPiece tokens. `token_count` always counts words. `size` and `size_unit` describe the configured length measure.

- Text: pack sentences; split a sentence at word boundaries if it is too long.
- Tables: split on rows and repeat the first two header/separator lines.
- Lists: split at item markers at the start of a line; indented lines stay within their item.
- Code, diagrams and image placeholders: preserve the complete block.

A single table row with its header, a list item or an atomic code block can exceed the budget. Such chunks carry `oversized: true` and an `oversize_reason`. `OversizePolicy: kukri.OversizeError` rejects them instead. The default is `OversizeKeep`.

Use `LengthFunc func(string) int` and `LengthUnit` to supply another deterministic counter. It must return zero for empty text and a positive count for nonempty text. Complete candidates, including separators, are measured; additivity is not assumed. A word that alone exceeds a custom tokenizer budget is also reported as oversized.

### Complete embedding inputs and repacking

`EmbeddingBudget` adds a separate limit on the complete `EmbedText`, including heading paths and tokenizer special tokens. `CountTokens func(string) (int, error)` must be deterministic and count without padding or truncation; an empty string may consume special tokens. Counter errors abort processing. The counter must be safe for concurrent use when the chunker is shared. `MaxTokens: 0` disables the body limit only when an embedding budget is configured.

```go
import "github.com/ETretyakov/kukri/tokenbudget"

counter, err := tokenbudget.FromFile("/path/to/tokenizer.json")
if err != nil { /* handle error */ }
c := kukri.New(kukri.Config{
    MaxTokens: 0, // use the complete embedding-token budget alone
    EmbeddingBudget: counter.Budget(256),
    RepackText: true,
    PrependSectionPath: true,
    CompactTableEmbedding: true,
    OversizePolicy: kukri.OversizeError,
}, nil)
```

The optional local adapter supports the uncased BERT WordPiece profile used by BGE and MiniLM; see [its supported configuration and compatibility checks](tokenbudget/README.md). Other tokenizers can provide the callback directly. Input limits belong to the embedding model: a tokenizer file does not by itself establish that limit.

`RepackText` first splits oversized paragraphs and then greedily packs the fragments, preserving paragraph separators and structural boundaries. It is opt-in and requires `MergeSize`. `CompactTableEmbedding` removes padding and shortens Markdown delimiter rows in the embedding view only; `Text` stays unchanged. Both options default to false.

When the embedding budget is enabled, chunks report `embedding_tokens`, `embedding_tokenizer`, `embedding_oversized` and `embedding_oversize_reason`. A long heading prefix, indivisible word, row or atomic block can still exceed the limit under `keep`. `error` rejects violations of either active budget, so a successful call under that policy contains no measured overflow. Kukri never silently truncates these elements. If metadata alone is too long, text splitting retains ordinary body-budget behavior instead of exploding into one chunk per word. Callers must handle these exceptions or revise their context policy before embedding.

### Total document budget and extractive compression

`DocumentBudget` enables **lossy selection of whole finished chunks**. For
example, keep at most half of the original embedding-input tokens, and no more
than 12 chunks:

```go
counter, err := tokenbudget.FromFile("/path/to/tokenizer.json")
if err != nil { /* handle error */ }
c := kukri.New(kukri.Config{
    MaxTokens: 0,
    EmbeddingBudget: counter.Budget(256),
    RepackText: true,
    PrependSectionPath: true,
    DocumentBudget: &kukri.DocumentBudget{
        RetainFraction: 0.5,
        MaxChunks: 12,
        Strategy: kukri.SelectTermsContext,
    },
}, nil)
result, err := c.ChopWithReport(ctx, markdownText)
// result.Chunks contains the retained source chunks.
// result.Compression reports original/selected chunks and tokens, omissions,
// effective limits, and whether no candidate fitted.
```

`DocumentBudget.MaxTokens` is an optional absolute total **model-token** limit,
unlike the historical word-count field `Config.MaxTokens`. It and `RetainFraction`
require `EmbeddingBudget`. The cost is the sum of the measured complete
`EmbedText` inputs, including every prefix, repeated header and special token.
It does not include any extra windows, text or tokens added downstream.
`RetainFraction` is in `(0,1]`; the effective cap is rounded down. All specified
limits apply together. Zero-valued fields disable individual limits; an empty
`DocumentBudget` is an error. A nil budget preserves existing behavior and JSON.

| Strategy | Selection |
| --- | --- |
| `SelectUniform` / `uniform` | Spread candidates by repeatedly bisecting source-order intervals; accept those that fit |
| `SelectTerms` / `terms` | Recompute weighted term coverage per added token, reducing the contribution of repetitions |
| `SelectTermsContext` / `terms-context` (default) | The same selection, with preceding text context retained for table/list/code/diagram chunks in the same section |

Term weights use log-frequency in the original parsed blocks, recurring adjacent
2/3-word phrases, uppercase abbreviations, and name candidates detected by
capitalization. The fixed EN/RU stop list affects scoring only. This is a lexical
heuristic without stemming or a named-entity model; capitalized words are not
guaranteed to be names, and repeated phrases are not guaranteed to be terms.
Repeated output headers do not inflate source frequencies. Each feature receives
`log(1 + frequency) * (1 + 0.5 * (words - 1) + 0.5 * abbreviation + 0.25 * proper)`.
The last two indicators apply only to single-word features. Selected chunks reduce
the contribution of a repeated feature to `weight / (1 + previous chunks)`.

Context links are an adjacency heuristic, not a guarantee of semantic coherence:
the immediately preceding text source group is retained with a structural block.
Split table/list chunks share that requirement. It needs valid source line ranges,
which the built-in parser supplies. It does not resolve arbitrary references or
all natural-language dependencies.
The selector is greedy; it does not guarantee an optimal summary or nested
selections across different budgets.

Content, including stop words and negations, stays unchanged in retained chunks.
IDs and source order are preserved; `PrevID`/`NextID` now link the retained sequence
and can skip omitted source content. Keep the original document for provenance.
There is no implicit source expansion after selection. A tiny budget can produce
an empty chunk list with `no_candidate_fits: true`; the library does not break a
code block or exceed the budget to force one result. Per-chunk `OversizeError`
still rejects violations before selection. Under `keep`, an oversized chunk may
be retained if it fits the document budget and still needs downstream handling.

The original `Chop` API applies the same selection but returns only chunks.
`ChopWithReport` and CLI output include the optional `compression` report.
Selection is performed per document in batch mode, never across the corpus.

Compression drops source material. Validate the selected budget against your own
retrieval queries and compare it with the same index without compression. The
reported token and chunk reductions do not guarantee preserved answer quality
or a proportional reduction in runtime, memory or cost.

### Metadata and provenance

Chunks expose `ID`, `PrevID`, `NextID`, `Text`, `EmbedText`, `Section`, `SectionPath`, `BlockType`, `SourceLines`, `TokenCount`, `Size`, `SizeUnit`, `Oversized`, `OversizeReason`, and `Source`.

IDs are unique **within one document**. Assign a unique `Source` to each document and use `(Source, ID)` when indexing multiple documents. `SourceLines` is an enclosing range of the original source blocks, not an exact substring coordinate: split parts may share a range and tables repeat headers. Paragraph whitespace may be normalized.

`PrependSectionPath` adds the heading path to `EmbedText`, leaving `Text` unchanged. This prefix is outside the Text size budget. Enable `EmbeddingBudget` to count it with the body, or check the tokenizer limit downstream; whether the prefix helps is an empirical question for each model and dataset.

### Supported Markdown

The parser targets the block subset commonly found in converted technical documents: ATX headings H1-H6, fenced code (backticks or tildes, matched delimiter length), pipe-prefixed tables, consecutive list items with indented continuation lines, Docling-style image placeholders, and paragraphs. It is not a complete CommonMark/GFM parser. Setext headings, arbitrary HTML, block quotes and more complex nested block constructs are not reconstructed as a full Markdown AST. Seven or more heading hashes are ordinary text, not a valid heading level.

## CLI

```sh
go build -o kukri ./cmd/chunker
./kukri -input document.md -max-words 128 -merge size
./kukri -batch -input corpus.jsonl -output chunks.jsonl -max-words 128 -merge none
./kukri -input document.md -max-words 128 -merge similarity -scorer tfidf -threshold 0.4
./kukri -input document.md -max-words 0 -embedding-tokenizer /path/to/tokenizer.json \
  -max-embedding-tokens 256 -repack-text -prepend-section-path \
  -compact-table-embedding -oversize error
./kukri -input document.md -max-words 0 -embedding-tokenizer /path/to/tokenizer.json \
  -max-embedding-tokens 256 -repack-text -prepend-section-path \
  -document-retain 0.5 -document-max-chunks 12 -document-selection terms-context
```

Batch input contains one `{"id":"document-id","text":"..."}` record per line. IDs must be nonempty and unique, and `text` must be a string. Empty documents produce an empty chunk list. Invalid input or processing errors cause a nonzero exit; no failed document is silently skipped. A failed batch can leave partial output, which callers must discard.

`-max-tokens` remains an alias for `-max-words`. The former `-model` and `-merge-blocks` flags are not supported.

### Migration from 7464582

- Replace positive `Threshold` with nil Scorer, formerly used to disable merging, with `MergeMode: MergeNone` and threshold zero.
- Set `MergeMode: MergeSimilarity` explicitly when supplying a Scorer.
- Large single sentences now split at word boundaries. Chunk boundaries and IDs can therefore change; rebuild the index.
- Do not interpret `SourceLines` as precise offsets of a split fragment.

## Tests

```sh
go test -race ./...
go vet ./...
go test -fuzz=FuzzChop -fuzztime=10s .
```

Tests cover cumulative and document budgets, token counting, retained source
content, oversize handling, invalid scorer responses, parser boundaries, batch
failures and randomized Markdown inputs.
