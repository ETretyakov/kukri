# BERT uncased embedding token counts

`FromFile` loads a local Hugging Face `tokenizer.json`. The supported profile is WordPiece with BertNormalizer (`clean_text`, `handle_chinese_chars`, `lowercase` enabled; accents stripped), BertPreTokenizer and a single-sequence TemplateProcessing template. Added tokens must be literal, unnormalized special tokens already in the vocabulary. Other profiles fail with an error; supply `kukri.EmbeddingBudget.CountTokens` for another tokenizer family.

The counter counts the full string and the template's special tokens. Saved padding and truncation settings are deliberately ignored. It exposes counts only, not IDs or offsets. Callers must pass the same `EmbedText` to their model and avoid adding an unbudgeted prompt. The core Kukri API does not import this adapter; this package uses `golang.org/x/text` for canonical decomposition.

The supported profile is used by BGE-small-en-v1.5 and all-MiniLM-L6-v2.
The test suite includes a compact WordPiece fixture and reference counts from
Hugging Face tokenizers. Compatibility is limited to the validated profile; an
unsupported tokenizer configuration returns an error.

HF's `unicode_categories` 0.1.1 uses older category tables than Go. In particular, private-use characters are removed, unassigned characters survive, and newer combining marks are not necessarily stripped. `unicode_tables.go` preserves the reference classifier, including punctuation. The upstream MIT notice is retained in `UNICODE_LICENSE`; regenerate with `internal/generate_unicode.py PATH_TO_CRATE/src/tables.rs unicode_tables.go`. The source file checksum is embedded in the generated file.

Algorithm references: [BertNormalizer](https://github.com/huggingface/tokenizers/blob/v0.22.2/tokenizers/src/normalizers/bert.rs), [WordPiece](https://github.com/huggingface/tokenizers/blob/v0.22.2/tokenizers/src/models/wordpiece/mod.rs), [unicode_categories 0.1.1](https://crates.io/crates/unicode_categories/0.1.1). This adapter is independently implemented; the category ranges are derived under the retained MIT license.

Run the adapter tests with `go test ./tokenbudget`. Supply another tokenizer
family through the `EmbeddingBudget.CountTokens` callback.
