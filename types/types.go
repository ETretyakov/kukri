// Package types defines the shared data structures passed between the
// parser, segmenter, and chunker stages of the kukri pipeline.
package types

// BlockType classifies a parsed Markdown element.
type BlockType string

const (
	BlockText    BlockType = "text"
	BlockTable   BlockType = "table"
	BlockCode    BlockType = "code"
	BlockDiagram BlockType = "diagram" // mermaid / plantuml / dot fenced blocks
	BlockList    BlockType = "list"
	BlockImage   BlockType = "image" // Docling > **[IMAGE …]** placeholder
	BlockHeader  BlockType = "header"
)

// Block is a single typed unit parsed from a Markdown document.
type Block struct {
	Type      BlockType
	Content   string
	Level     int // ATX header level 1–6; 0 for all other block types
	LineStart int // 1-based line number of the first line in the source
	LineEnd   int // 1-based line number of the last line in the source
}

// Section groups the content blocks that fall under a single heading.
// Blocks that precede the first heading are placed in a preamble section
// (Title == "", Level == 0).
type Section struct {
	Title  string   // heading text, empty for the preamble
	Level  int      // ATX heading level 1–6; 0 for the preamble
	Path   []string // breadcrumb from root to this section, e.g. ["Ch1", "1.2 Methods"]
	Blocks []Block  // content blocks (never contains BlockHeader)
}

// Chunk is the atomic output unit of the kukri pipeline, ready for
// embedding and insertion into a vector store.
type Chunk struct {
	// ID is a human-readable unique identifier within the document,
	// built from the section slug and a counter: "introduction_0", "methods_3".
	ID string `json:"id"`

	// PrevID and NextID link adjacent chunks for context retrieval (empty
	// on the first and last chunk respectively).
	PrevID string `json:"prev_id,omitempty"`
	NextID string `json:"next_id,omitempty"`

	// Text preserves content order; paragraph whitespace may be normalized.
	Text string `json:"text"`

	// EmbedText is the text to pass to the embedding model.
	// By default equals Text. When Config.PrependSectionPath is true,
	// it is prefixed with the section breadcrumb:
	//   "Chapter 2 > Safety Analysis\n\n[chunk text]"
	// Downstream consumers should embed EmbedText but display Text.
	EmbedText string `json:"embed_text,omitempty"`

	// Section is the title of the heading this chunk falls under.
	Section string `json:"section"`

	// SectionPath is the full heading breadcrumb, e.g. ["Chapter 1", "1.2 Methods"].
	SectionPath []string `json:"section_path"`

	// BlockType records which Markdown element produced this chunk.
	BlockType BlockType `json:"block_type"`

	// SourceLines is the [start, end] 1-based line range in the source
	// Markdown enclosing the source block(s). Split parts may share this range;
	// it is not an exact substring location after normalization or header repetition.
	SourceLines [2]int `json:"source_lines"`

	// TokenCount is the whitespace-delimited word count of Text.
	TokenCount int `json:"token_count"`

	// Size is the length of Text under Config.LengthFunc (words by default).
	Size     int    `json:"size"`
	SizeUnit string `json:"size_unit"`
	// Oversized identifies an indivisible element that exceeds MaxTokens.
	Oversized      bool   `json:"oversized,omitempty"`
	OversizeReason string `json:"oversize_reason,omitempty"`

	// EmbeddingTokens measures the complete EmbedText, including tokenizer special
	// tokens, when Config.EmbeddingBudget is enabled. No truncation is applied.
	EmbeddingTokens         int    `json:"embedding_tokens,omitempty"`
	EmbeddingTokenizer      string `json:"embedding_tokenizer,omitempty"`
	EmbeddingOversized      bool   `json:"embedding_oversized,omitempty"`
	EmbeddingOversizeReason string `json:"embedding_oversize_reason,omitempty"`

	// Source is the filename or document ID set via Config.Source.
	Source string `json:"source"`
}

// Result is the top-level envelope written to JSON in single-file mode.
type Result struct {
	Chunks      []Chunk            `json:"chunks"`
	Compression *CompressionReport `json:"compression,omitempty"`
}

// SelectionStrategy selects whole output chunks under a document budget.
type SelectionStrategy string

const (
	SelectUniform      SelectionStrategy = "uniform"
	SelectTerms        SelectionStrategy = "terms"
	SelectTermsContext SelectionStrategy = "terms-context"
)

// DocumentBudget enables lossy, query-independent selection after chunking.
// Zero fields are disabled. At least one limit must be set. Token limits require
// EmbeddingBudget and sum its counts over complete EmbedText inputs, including
// repeated headers and special tokens. They do not count downstream windows.
type DocumentBudget struct {
	MaxTokens int
	MaxChunks int
	// RetainFraction optionally caps tokens at floor(fraction * original tokens).
	// Values are in (0,1]; when MaxTokens is also set, the smaller limit applies.
	RetainFraction float64
	// Defaults to SelectTermsContext. Uniform is a control, not a salience model.
	Strategy SelectionStrategy
}

// CompressionReport makes omissions and budget utilization explicit.
type CompressionReport struct {
	Strategy           SelectionStrategy `json:"strategy"`
	OriginalChunks     int               `json:"original_chunks"`
	SelectedChunks     int               `json:"selected_chunks"`
	OmittedChunks      int               `json:"omitted_chunks"`
	OriginalTokens     int               `json:"original_tokens,omitempty"`
	SelectedTokens     int               `json:"selected_tokens,omitempty"`
	TokenBudgetEnabled bool              `json:"token_budget_enabled"`
	TokenLimit         int               `json:"token_limit"`
	ChunkLimit         int               `json:"chunk_limit,omitempty"`
	ContextAddedChunks int               `json:"context_added_chunks,omitempty"`
	NoCandidateFits    bool              `json:"no_candidate_fits,omitempty"`
}

// MergeMode selects how adjacent text blocks are combined.
type MergeMode string

const (
	MergeSize       MergeMode = "size"
	MergeNone       MergeMode = "none"
	MergeSimilarity MergeMode = "similarity"
)

// OversizePolicy controls indivisible words, rows, list items and code blocks.
type OversizePolicy string

const (
	OversizeKeep  OversizePolicy = "keep"
	OversizeError OversizePolicy = "error"
)

// EmbeddingBudget constrains the complete model input independently of the body
// budget. CountTokens must count without truncation or padding, include the model's
// special tokens and be deterministic. Empty inputs may have a nonzero count.
// For concurrent use of Chunker, the callback must also be safe for concurrent use.
type EmbeddingBudget struct {
	MaxTokens   int
	Tokenizer   string // identifying name or immutable tokenizer hash
	CountTokens func(string) (int, error)
}

// Config holds the parameters that control the chunking pipeline.
type Config struct {
	// MaxTokens limits Text length. The historical name is retained for API
	// compatibility: the default unit is whitespace-delimited words, not BPE.
	// Zero disables the body limit only when EmbeddingBudget is configured.
	// Atomic elements that cannot fit are marked or rejected by OversizePolicy.
	MaxTokens int
	// MergeMode defaults to MergeSize. Similarity requires a non-nil Scorer.
	MergeMode MergeMode
	// Threshold in [0,1] is used only by MergeSimilarity. Other modes require 0.
	Threshold float32
	// LengthFunc defaults to whitespace word count. A custom counter must be
	// deterministic, return 0 for empty text and a positive count otherwise.
	// Full candidate text is measured: additivity is not assumed.
	LengthFunc func(string) int
	// LengthUnit is required for a custom counter, e.g. "cl100k_base_tokens".
	LengthUnit string
	// OversizePolicy defaults to OversizeKeep; exceptions carry a reason.
	OversizePolicy OversizePolicy
	Source         string
	// RepackText splits oversized paragraphs before greedily packing their
	// fragments within each uninterrupted text run. Opt-in; requires MergeSize.
	RepackText bool
	// CompactTableEmbedding removes table padding and shortens Markdown
	// delimiter rows in EmbedText only; cell values and Text are retained.
	CompactTableEmbedding bool
	// EmbeddingBudget optionally limits full EmbedText as well as Text.
	EmbeddingBudget *EmbeddingBudget
	// DocumentBudget optionally selects a subset of the finished chunks.
	// Omitted content is not recoverable from that subset; retain the source.
	DocumentBudget *DocumentBudget
	// PrependSectionPath affects EmbedText only. The prefix is outside the Text
	// budget, but is included in EmbeddingBudget when enabled.
	PrependSectionPath bool
}
