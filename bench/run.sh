#!/usr/bin/env bash
set -euo pipefail

INPUT=${1:?"Usage: run.sh <input.md> [output_dir]"}
OUTDIR=${2:-/tmp/kukri_bench}
MODEL=${MODEL:-"./models/KnightsAnalytics_all-MiniLM-L6-v2"}
MAX_TOKENS=${MAX_TOKENS:-512}
THRESHOLD=${THRESHOLD:-0.7}

PYTHON="$(dirname "$0")/.venv/bin/python3"
if [[ ! -x "$PYTHON" ]]; then
  echo "error: venv not found at bench/.venv — run: python3 -m venv bench/.venv && bench/.venv/bin/pip install -r bench/requirements.txt"
  exit 1
fi

mkdir -p "$OUTDIR"

echo "==> [1/5] kukri (Go hybrid)"
go run ./cmd/chunker \
  -input "$INPUT" \
  -model "$MODEL" \
  -output "$OUTDIR/chunks_kukri.json" \
  -max-tokens "$MAX_TOKENS" \
  -threshold "$THRESHOLD"

echo "==> [2/5] chonkie semantic"
"$PYTHON" bench/chunkers/chonkie_semantic.py \
  --input "$INPUT" \
  --output "$OUTDIR/chunks_chonkie_sem.json" \
  --max-tokens "$MAX_TOKENS" \
  --threshold "$THRESHOLD"

echo "==> [3/5] chonkie token"
"$PYTHON" bench/chunkers/chonkie_token.py \
  --input "$INPUT" \
  --output "$OUTDIR/chunks_chonkie_tok.json" \
  --max-tokens "$MAX_TOKENS"

echo "==> [4/5] langchain markdown"
"$PYTHON" bench/chunkers/langchain_md.py \
  --input "$INPUT" \
  --output "$OUTDIR/chunks_langchain.json" \
  --max-tokens "$MAX_TOKENS"

echo "==> [5/5] llamaindex sentence"
"$PYTHON" bench/chunkers/llamaindex_sentence.py \
  --input "$INPUT" \
  --output "$OUTDIR/chunks_llama.json" \
  --max-tokens "$MAX_TOKENS"

echo ""
echo "==> computing metrics..."
"$PYTHON" bench/metrics.py \
  --model all-MiniLM-L6-v2 \
  --output "$OUTDIR/results.json" \
  "kukri:$OUTDIR/chunks_kukri.json" \
  "chonkie_semantic:$OUTDIR/chunks_chonkie_sem.json" \
  "chonkie_token:$OUTDIR/chunks_chonkie_tok.json" \
  "langchain_md:$OUTDIR/chunks_langchain.json" \
  "llamaindex_sentence:$OUTDIR/chunks_llama.json"

echo ""
echo "results saved to $OUTDIR/results.json"

# ---------------------------------------------------------------------------
# Optional: gold-based coverage benchmark (requires ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  GOLD=${GOLD:-"$OUTDIR/gold.jsonl"}
  CORPUS=${CORPUS:-""}

  if [[ -n "$CORPUS" ]]; then
    echo ""
    echo "==> [gold] generating QA pairs from $CORPUS..."
    "$PYTHON" bench/generate_gold.py \
      --input   "$CORPUS" \
      --output  "$GOLD" \
      --model   "${GOLD_MODEL:-claude-haiku-4-5}" \
      --pairs-per-doc "${GOLD_PAIRS:-5}"

    echo ""
    echo "==> [coverage] measuring Answer Coverage Rate..."
    "$PYTHON" bench/coverage_bench.py \
      --gold      "$GOLD" \
      --corpus    "$CORPUS" \
      --output    "$OUTDIR/coverage_results.json" \
      --max-tokens "$MAX_TOKENS" \
      --threshold  "$THRESHOLD"

    echo "coverage results saved to $OUTDIR/coverage_results.json"
  else
    echo ""
    echo "note: set CORPUS=path/to/corpus.jsonl to run the coverage benchmark"
  fi
else
  echo ""
  echo "note: set ANTHROPIC_API_KEY to run the gold-based coverage benchmark"
fi
