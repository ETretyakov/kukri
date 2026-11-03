package scorer

import (
	"context"
	"math"
	"sort"
	"strings"
	"unicode"
)

// TFIDF is a deterministic lexical reference scorer. IDF is fitted separately
// to each contiguous text run supplied to Score: log((1+n)/(1+df))+1. Terms are
// lowercase Unicode letter/digit sequences; TF is the raw occurrence count.
// There is no stopword removal, stemming, pretrained model or corpus fitting.
type TFIDF struct{}

func (TFIDF) Score(ctx context.Context, texts []string) ([]float32, error) {
	if len(texts) < 2 {
		return nil, nil
	}
	counts := make([]map[string]float64, len(texts))
	df := map[string]int{}
	for i, text := range texts {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		counts[i] = map[string]float64{}
		for _, term := range strings.FieldsFunc(strings.ToLower(text), func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) }) {
			counts[i][term]++
		}
		for term := range counts[i] {
			df[term]++
		}
	}
	terms := make([]string, 0, len(df))
	for term := range df {
		terms = append(terms, term)
	}
	sort.Strings(terms) // stable accumulation, independent of map iteration order
	norms := make([]float64, len(texts))
	for _, term := range terms {
		idf := math.Log(float64(1+len(texts))/float64(1+df[term])) + 1
		for i := range counts {
			if tf := counts[i][term]; tf != 0 {
				counts[i][term] = tf * idf
				norms[i] += tf * tf * idf * idf
			}
		}
	}
	scores := make([]float32, len(texts)-1)
	for i := range scores {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if norms[i] == 0 || norms[i+1] == 0 {
			continue
		}
		dot := 0.0
		for _, term := range terms {
			dot += counts[i][term] * counts[i+1][term]
		}
		scores[i] = float32(math.Min(1, dot/math.Sqrt(norms[i]*norms[i+1])))
	}
	return scores, nil
}
