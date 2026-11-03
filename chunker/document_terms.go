package chunker

import (
	"math"
	"sort"
	"strings"
	"unicode"

	"github.com/ETretyakov/kukri/types"
)

// Stop words affect scoring only. Source content, negations and punctuation are
// never removed from returned chunks. This small EN/RU list is not a stemmer.
var documentStopWords = func() map[string]bool {
	m := map[string]bool{}
	for _, w := range strings.Fields("a an the and or but if then else as at by for from in into of on to with without is are was were be been being this that these those it its they their we our you your he she his her not no only also can could may might must should would will do does did have has had than so such which who what when where how all any each some other there here through over under between about и или а но если то как к ко в во на по из от до с со за для при без о об у над под между что это этот эта эти тот та те он она оно они мы вы я его ее их наш ваш свой не ни да же бы ли лишь только также быть был была были есть будет будут который которая которые когда где чем все всё уже ещё") {
		m[w] = true
	}
	return m
}()

type termEvidence struct {
	frequency    int
	words        int
	abbreviation bool
	proper       bool
}
type lexicalWord struct {
	text                       string
	start, end                 int
	stop, abbreviation, proper bool
}

func lexicalTerms(text string) map[string]termEvidence {
	words := []lexicalWord{}
	start := -1
	flush := func(end int) {
		if start < 0 {
			return
		}
		raw := text[start:end]
		upper, lower, letters := 0, 0, 0
		for _, r := range raw {
			if unicode.IsLetter(r) {
				letters++
			}
			if unicode.IsUpper(r) {
				upper++
			}
			if unicode.IsLower(r) {
				lower++
			}
		}
		first := []rune(raw)[0]
		initial := len(words) == 0
		if !initial {
			initial = strings.ContainsAny(text[words[len(words)-1].end:start], ".!?\n")
		}
		w := lexicalWord{text: strings.ToLower(raw), start: start, end: end, abbreviation: letters >= 2 && letters <= 12 && upper == letters, proper: unicode.IsUpper(first) && lower > 0 && !initial}
		w.stop = documentStopWords[w.text] || len([]rune(w.text)) < 2
		words = append(words, w)
		start = -1
	}
	for i, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' {
			if start < 0 {
				start = i
			}
		} else {
			flush(i)
		}
	}
	flush(len(text))
	out := map[string]termEvidence{}
	for i, w := range words {
		if w.stop {
			continue
		}
		e := out[w.text]
		e.frequency++
		e.words = 1
		e.abbreviation = e.abbreviation || w.abbreviation
		e.proper = e.proper || w.proper
		out[w.text] = e
		phrase := w.text
		for width := 2; width <= 3 && i+width <= len(words); width++ {
			prev, next := words[i+width-2], words[i+width-1]
			gap := text[prev.end:next.start]
			if next.stop || strings.TrimSpace(gap) != "" || strings.Contains(gap, "\n\n") {
				break
			}
			phrase += " " + next.text
			e := out[phrase]
			e.frequency++
			e.words = width
			out[phrase] = e
		}
	}
	return out
}

func documentFeatures(chunks []types.Chunk, sections []types.Section) ([][]int, []float64) {
	// Source blocks, before header repetition/repacking, determine frequency.
	stats := map[string]termEvidence{}
	for _, sec := range sections {
		for _, block := range sec.Blocks {
			for term, e := range lexicalTerms(block.Content) {
				x := stats[term]
				x.frequency += e.frequency
				x.words = e.words
				x.abbreviation = x.abbreviation || e.abbreviation
				x.proper = x.proper || e.proper
				stats[term] = x
			}
		}
	}
	keys := []string{}
	for term, e := range stats {
		if e.words == 1 || e.frequency >= 2 {
			keys = append(keys, term)
		}
	}
	sort.Strings(keys)
	ids := make(map[string]int, len(keys))
	weights := make([]float64, len(keys))
	for i, term := range keys {
		ids[term] = i
		e := stats[term]
		bonus := 1.0 + 0.5*float64(e.words-1)
		if e.abbreviation {
			bonus += 0.5
		}
		if e.proper {
			bonus += 0.25
		}
		weights[i] = math.Log1p(float64(e.frequency)) * bonus
	}
	features := make([][]int, len(chunks))
	for i, ch := range chunks {
		for term := range lexicalTerms(ch.Text) {
			if id, ok := ids[term]; ok {
				features[i] = append(features[i], id)
			}
		}
		sort.Ints(features[i])
	}
	return features, weights
}
