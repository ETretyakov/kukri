// Package parser converts raw Markdown text into a flat slice of typed
// [types.Block] values.
//
// The scanner is purely line-by-line (O(n) time and memory) and recognises:
//   - Fenced code blocks (``` … ```) and diagram blocks (```mermaid / plantuml / dot)
//   - Pipe tables (lines starting with |)
//   - ATX headers (# … ######)
//   - Unordered lists (- / *) and ordered lists (1. 2. …) with indented continuations
//   - Docling-style image placeholders (> **[IMAGE …]**)
//   - Plain text paragraphs (everything else)
//
// Block line numbers are 1-based so they can be reported directly to users.
package parser

import (
	"strings"

	"github.com/ETretyakov/kukri/types"
)

// Parse converts raw markdown text into a flat slice of typed blocks.
// Blocks are emitted in document order; headers themselves are included
// as [types.BlockHeader] so the segmenter can reconstruct the hierarchy.
func Parse(md string) []types.Block {
	lines := strings.Split(md, "\n")
	var blocks []types.Block

	i := 0
	for i < len(lines) {
		line := lines[i]

		// ── Fenced code or diagram block ────────────────────────────────────
		// Opened by a line that starts with ```.  The language tag (if any)
		// distinguishes diagrams (mermaid / plantuml / dot) from code.
		if tag, ok := strings.CutPrefix(line, "```"); ok {
			tag = strings.TrimSpace(strings.ToLower(tag))
			start := i
			i++
			buf := []string{line}
			for i < len(lines) {
				buf = append(buf, lines[i])
				if strings.HasPrefix(lines[i], "```") {
					i++
					break
				}
				i++
			}
			bt := types.BlockCode
			if tag == "mermaid" || tag == "plantuml" || tag == "dot" {
				bt = types.BlockDiagram
			}
			blocks = append(blocks, types.Block{
				Type:      bt,
				Content:   strings.Join(buf, "\n"),
				LineStart: start + 1,
				LineEnd:   i,
			})
			continue
		}

		// ── Docling image placeholder ────────────────────────────────────────
		// Docling renders images as:  > **[IMAGE | caption]**
		if strings.HasPrefix(line, "> **[IMAGE") {
			blocks = append(blocks, types.Block{
				Type:      types.BlockImage,
				Content:   line,
				LineStart: i + 1,
				LineEnd:   i + 1,
			})
			i++
			continue
		}

		// ── Pipe table ───────────────────────────────────────────────────────
		// Accumulate all consecutive pipe-prefixed lines into one block so
		// the chunker can split rows while repeating the header.
		if strings.HasPrefix(strings.TrimSpace(line), "|") {
			start := i
			var buf []string
			for i < len(lines) && strings.HasPrefix(strings.TrimSpace(lines[i]), "|") {
				buf = append(buf, lines[i])
				i++
			}
			blocks = append(blocks, types.Block{
				Type:      types.BlockTable,
				Content:   strings.Join(buf, "\n"),
				LineStart: start + 1,
				LineEnd:   i,
			})
			continue
		}

		// ── ATX header ───────────────────────────────────────────────────────
		// Require one space after the hash run: "# Title", "## Sub", etc.
		// Lines like "#tag" (no space) fall through to the text paragraph.
		if level, ok := atxHeaderLevel(line); ok {
			blocks = append(blocks, types.Block{
				Type:      types.BlockHeader,
				Content:   strings.TrimSpace(line[level+1:]),
				Level:     level,
				LineStart: i + 1,
				LineEnd:   i + 1,
			})
			i++
			continue
		}

		// ── List block ───────────────────────────────────────────────────────
		// Collect the starting list item and any indented continuation lines.
		if isListLine(line) {
			start := i
			var buf []string
			for i < len(lines) && (isListLine(lines[i]) || isContinuation(lines[i])) {
				buf = append(buf, lines[i])
				i++
			}
			blocks = append(blocks, types.Block{
				Type:      types.BlockList,
				Content:   strings.Join(buf, "\n"),
				LineStart: start + 1,
				LineEnd:   i,
			})
			continue
		}

		// ── Text paragraph ───────────────────────────────────────────────────
		// Accumulate consecutive non-empty lines that do not open any other
		// block type; join them with a single space (soft-wrap).
		// We test isATXHeader (requires a space after the hashes) rather than
		// HasPrefix("#") so that "#notaheader" is treated as plain text instead
		// of causing an infinite loop (no branch would consume the line).
		if strings.TrimSpace(line) != "" {
			start := i
			var buf []string
			for i < len(lines) &&
				strings.TrimSpace(lines[i]) != "" &&
				!isATXHeader(lines[i]) &&
				!strings.HasPrefix(strings.TrimSpace(lines[i]), "|") &&
				!strings.HasPrefix(lines[i], "```") &&
				!strings.HasPrefix(lines[i], "> **[IMAGE") &&
				!isListLine(lines[i]) {
				buf = append(buf, lines[i])
				i++
			}
			if len(buf) > 0 {
				blocks = append(blocks, types.Block{
					Type:      types.BlockText,
					Content:   strings.Join(buf, " "),
					LineStart: start + 1,
					LineEnd:   i,
				})
			}
			continue
		}

		i++ // blank line — advance and try the next
	}

	return blocks
}

// atxHeaderLevel returns (level, true) when line is a valid ATX header
// ("# …", "## …", …), or (0, false) otherwise.
// A valid ATX header requires at least one space after the hash run.
func atxHeaderLevel(line string) (int, bool) {
	level := 0
	for level < len(line) && line[level] == '#' {
		level++
	}
	if level == 0 || level >= len(line) || line[level] != ' ' {
		return 0, false
	}
	return level, true
}

// isATXHeader reports whether line is a valid ATX header.
func isATXHeader(line string) bool {
	_, ok := atxHeaderLevel(line)
	return ok
}

// isListLine reports whether line begins a list item (unordered or ordered).
func isListLine(line string) bool {
	t := strings.TrimSpace(line)
	if strings.HasPrefix(t, "- ") || strings.HasPrefix(t, "* ") {
		return true
	}
	// Ordered: "1. " "12. " etc.
	for j, ch := range t {
		if ch >= '0' && ch <= '9' {
			continue
		}
		if ch == '.' && j > 0 && j+1 < len(t) && t[j+1] == ' ' {
			return true
		}
		break
	}
	return false
}

// isContinuation reports whether line is an indented continuation of the
// preceding list item (two or more leading spaces, non-blank).
func isContinuation(line string) bool {
	return strings.HasPrefix(line, "  ") && strings.TrimSpace(line) != ""
}
