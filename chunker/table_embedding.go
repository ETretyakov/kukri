package chunker

import "strings"

// compactTableEmbedding removes Markdown table padding and shortens the
// delimiter row. Cell values, escaped pipes and column alignment are retained.
// Text/source provenance remain unchanged: only the embedding view is compact.
func compactTableEmbedding(text string) string {
	lines := strings.Split(text, "\n")
	for row, line := range lines {
		var cells []string
		start, slashes := 0, 0
		for i, r := range line {
			if r == '|' && slashes%2 == 0 {
				cells = append(cells, strings.TrimSpace(line[start:i]))
				start = i + 1
			}
			if r == '\\' {
				slashes++
			} else {
				slashes = 0
			}
		}
		cells = append(cells, strings.TrimSpace(line[start:]))
		if row == 1 {
			for i, cell := range cells {
				if cell == "" {
					continue
				}
				body := strings.Trim(cell, ":")
				if len(body) >= 3 && strings.Trim(body, "-") == "" {
					replacement := "---"
					if strings.HasPrefix(cell, ":") {
						replacement = ":" + replacement
					}
					if strings.HasSuffix(cell, ":") {
						replacement += ":"
					}
					cells[i] = replacement
				}
			}
		}
		lines[row] = strings.Join(cells, " | ")
	}
	return strings.Join(lines, "\n")
}
