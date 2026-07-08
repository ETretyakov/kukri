// Package types defines the shared data structures passed between the
// parser, segmenter, embedder, and chunker stages of the kukri pipeline.
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
// It is an intermediate representation consumed by the segmenter; headers
// are not present in the final []Chunk output.
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

	// Text is the raw chunk content as it appears in the source document.
	Text string `json:"text"`

	// Section is the title of the heading this chunk falls under.
	Section string `json:"section"`

	// SectionPath is the full heading breadcrumb, e.g. ["Chapter 1", "1.2 Methods"].
	SectionPath []string `json:"section_path"`

	// BlockType records which Markdown element produced this chunk.
	BlockType BlockType `json:"block_type"`

	// SourceLines is the [start, end] 1-based line range in the source
	// Markdown that this chunk was derived from.
	SourceLines [2]int `json:"source_lines"`

	// TokenCount is the whitespace-delimited word count of Text.
	TokenCount int `json:"token_count"`

	// Source is the filename or document ID set via Config.Source.
	Source string `json:"source"`
}

// Result is the top-level envelope written to JSON in single-file mode.
type Result struct {
	Chunks []Chunk `json:"chunks"`
}

// Config holds the parameters that control the chunking pipeline.
type Config struct {
	// MaxTokens is the soft upper limit on chunk size (word count).
	// Text blocks exceeding this are split; code/image/diagram blocks are
	// always emitted whole regardless of size.
	MaxTokens int

	// Threshold is the minimum cosine similarity [0, 1] required to merge
	// two adjacent sentences into the same chunk.  Lower values produce
	// larger, more topically diverse chunks; higher values produce finer,
	// more homogeneous chunks.
	Threshold float32

	// ModelPath is the path to the ONNX sentence-transformer model directory.
	ModelPath string

	// Source is the filename or document ID attached to every chunk's Source field.
	Source string
}
