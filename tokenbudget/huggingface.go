// Package tokenbudget counts complete inputs for local BERT uncased WordPiece
// tokenizers. It supports the BGE-small-en-v1.5 and all-MiniLM-L6-v2 configuration;
// other tokenizer families can use a caller-supplied EmbeddingBudget callback.
package tokenbudget

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/ETretyakov/kukri/types"
	"golang.org/x/text/unicode/norm"
)

// Counter is immutable and safe for concurrent use. Counts include template
// special tokens, with no padding or truncation. This is a count-only adapter,
// not a tokenizer API: it does not expose token IDs or character offsets.
type Counter struct {
	id           string
	vocab        map[string]int
	added        []string
	maxChars     int
	continuation string
	specialCount int
}

// FromFile accepts only the explicitly supported BERT uncased configuration.
// Unsupported normalization, added-token matching and templates fail closed.
func FromFile(path string) (*Counter, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg struct {
		Model struct {
			Type       string
			UnkToken   string `json:"unk_token"`
			Continuing string `json:"continuing_subword_prefix"`
			MaxChars   int    `json:"max_input_chars_per_word"`
			Vocab      map[string]int
		} `json:"model"`
		Normalizer struct {
			Type    string
			Clean   bool  `json:"clean_text"`
			Chinese bool  `json:"handle_chinese_chars"`
			Lower   bool  `json:"lowercase"`
			Strip   *bool `json:"strip_accents"`
		} `json:"normalizer"`
		PreTokenizer  struct{ Type string } `json:"pre_tokenizer"`
		PostProcessor struct {
			Type    string
			Single  []map[string]json.RawMessage
			Special map[string]struct {
				IDs    []int    `json:"ids"`
				Tokens []string `json:"tokens"`
			} `json:"special_tokens"`
		} `json:"post_processor"`
		Added []struct {
			ID         int
			Content    string
			SingleWord bool `json:"single_word"`
			LStrip     bool `json:"lstrip"`
			RStrip     bool `json:"rstrip"`
			Normalized bool
			Special    bool
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, err
	}
	n := cfg.Normalizer
	if cfg.Model.Type != "WordPiece" || n.Type != "BertNormalizer" || !n.Clean || !n.Chinese || !n.Lower || (n.Strip != nil && !*n.Strip) || cfg.PreTokenizer.Type != "BertPreTokenizer" || cfg.PostProcessor.Type != "TemplateProcessing" {
		return nil, fmt.Errorf("supported tokenizer requires uncased WordPiece, BertNormalizer (clean text, Chinese splitting, accent stripping), BertPreTokenizer and TemplateProcessing")
	}
	if cfg.Model.MaxChars <= 0 || cfg.Model.Continuing == "" || len(cfg.Model.Vocab) == 0 {
		return nil, fmt.Errorf("invalid WordPiece vocabulary/configuration")
	}
	if _, ok := cfg.Model.Vocab[cfg.Model.UnkToken]; !ok {
		return nil, fmt.Errorf("missing unknown token")
	}
	c := &Counter{id: fmt.Sprintf("bert-uncased-v1:sha256:%x", sha256.Sum256(raw)), vocab: cfg.Model.Vocab, maxChars: cfg.Model.MaxChars, continuation: cfg.Model.Continuing}
	for _, a := range cfg.Added {
		id, ok := c.vocab[a.Content]
		if a.Content == "" || !ok || id != a.ID || !a.Special || a.Normalized || a.SingleWord || a.LStrip || a.RStrip {
			return nil, fmt.Errorf("unsupported added-token matching configuration")
		}
		c.added = append(c.added, a.Content)
	}
	// Longest literal wins at a shared position, as in HF AddedVocabulary.
	sort.Slice(c.added, func(i, j int) bool { return len(c.added[i]) > len(c.added[j]) })
	sequences := 0
	for _, item := range cfg.PostProcessor.Single {
		if len(item) != 1 {
			return nil, fmt.Errorf("invalid single-input template")
		}
		if data, ok := item["Sequence"]; ok {
			var s struct {
				ID     string
				TypeID int `json:"type_id"`
			}
			if err := json.Unmarshal(data, &s); err != nil {
				return nil, err
			}
			if s.ID != "A" || s.TypeID != 0 {
				return nil, fmt.Errorf("unsupported single-input sequence")
			}
			sequences++
		} else if data, ok := item["SpecialToken"]; ok {
			var s struct {
				ID     string
				TypeID int `json:"type_id"`
			}
			if err := json.Unmarshal(data, &s); err != nil {
				return nil, err
			}
			def, ok := cfg.PostProcessor.Special[s.ID]
			if !ok || len(def.IDs) != len(def.Tokens) || len(def.IDs) == 0 || s.TypeID != 0 {
				return nil, fmt.Errorf("invalid template special token")
			}
			for i, t := range def.Tokens {
				if id, ok := c.vocab[t]; !ok || id != def.IDs[i] {
					return nil, fmt.Errorf("template token missing from vocabulary")
				}
			}
			c.specialCount += len(def.IDs)
		} else {
			return nil, fmt.Errorf("unsupported template item")
		}
	}
	if sequences != 1 {
		return nil, fmt.Errorf("single-input template must contain sequence A exactly once")
	}
	return c, nil
}

func (c *Counter) Count(text string) (int, error) {
	if !utf8.ValidString(text) {
		return 0, fmt.Errorf("tokenizer input is not valid UTF-8")
	}
	count := c.specialCount
	for len(text) > 0 {
		at, token := -1, ""
		for _, a := range c.added {
			if i := strings.Index(text, a); i >= 0 && (at < 0 || i < at) {
				at, token = i, a
			}
		}
		if at < 0 {
			count += c.countOrdinary(text)
			break
		}
		count += c.countOrdinary(text[:at]) + 1
		text = text[at+len(token):]
	}
	return count, nil
}

func (c *Counter) countOrdinary(text string) int {
	var cleaned strings.Builder
	for _, r := range text {
		switch {
		case r == 0 || r == '\uFFFD' || ((inRanges(r, bertOther) || r >= 0xE000 && r <= 0xF8FF || r >= 0xF0000 && r <= 0xFFFFD || r >= 0x100000 && r <= 0x10FFFD) && r != '\t' && r != '\n' && r != '\r'):
			continue
		case unicode.IsSpace(r):
			cleaned.WriteByte(' ')
		case isChinese(r):
			cleaned.WriteByte(' ')
			cleaned.WriteRune(r)
			cleaned.WriteByte(' ')
		default:
			cleaned.WriteRune(r)
		}
	}
	// Lowercase expansion of U+0130 is equivalent to i after accent stripping.
	normalized := norm.NFD.String(cleaned.String())
	var word []rune
	count := 0
	flush := func() {
		if len(word) > 0 {
			count += c.countWord(word)
			word = word[:0]
		}
	}
	for _, r := range normalized {
		if inRanges(r, bertMark) {
			continue
		}
		if unicode.IsSpace(r) {
			flush()
			continue
		}
		if isPunctuation(r) {
			flush()
			count += c.countWord([]rune{r})
			continue
		}
		word = append(word, unicode.ToLower(r))
	}
	flush()
	return count
}

func (c *Counter) countWord(word []rune) int {
	if len(word) > c.maxChars {
		return 1
	}
	count := 0
	for start := 0; start < len(word); {
		end := len(word)
		for ; end > start; end-- {
			part := string(word[start:end])
			if start > 0 {
				part = c.continuation + part
			}
			if _, ok := c.vocab[part]; ok {
				break
			}
		}
		if end == start {
			return 1
		} // Entire pre-token becomes [UNK], not partial pieces.
		count++
		start = end
	}
	return count
}
func isPunctuation(r rune) bool {
	return r >= 33 && r <= 47 || r >= 58 && r <= 64 || r >= 91 && r <= 96 || r >= 123 && r <= 126 || inRanges(r, bertPunctuation)
}
func isChinese(r rune) bool {
	return r >= 0x4E00 && r <= 0x9FFF || r >= 0x3400 && r <= 0x4DBF || r >= 0x20000 && r <= 0x2A6DF || r >= 0x2A700 && r <= 0x2B73F || r >= 0x2B740 && r <= 0x2B81F || r >= 0x2B920 && r <= 0x2CEAF || r >= 0xF900 && r <= 0xFAFF || r >= 0x2F800 && r <= 0x2FA1F
}
func (c *Counter) Budget(maxTokens int) *types.EmbeddingBudget {
	return &types.EmbeddingBudget{MaxTokens: maxTokens, Tokenizer: c.id, CountTokens: c.Count}
}
