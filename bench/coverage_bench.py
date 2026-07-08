#!/usr/bin/env python3
"""
Answer Coverage Rate benchmark: what % of gold answer_spans are fully
contained within a single chunk, per chunking strategy.

A high coverage rate means the strategy keeps answer-bearing passages
intact rather than splitting them across chunk boundaries.

Usage:
  python bench/coverage_bench.py \
    --gold bench/gold.jsonl \
    --corpus corpus.jsonl \
    --output /tmp/coverage_results.json
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
from tabulate import tabulate

from chonkie import SemanticChunker, TokenChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import SentenceSplitter


# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MODEL_PATH = os.path.join(_REPO_ROOT, "models", "KnightsAnalytics_all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return " ".join(s.split())


def _contains(chunk_text: str, span: str) -> bool:
    """True if span is (approximately) contained in chunk_text."""
    if span in chunk_text:
        return True
    return _norm(span) in _norm(chunk_text)


# ---------------------------------------------------------------------------
# Kukri batch chunker
# ---------------------------------------------------------------------------

def chunk_kukri_batch(docs: dict[str, str], max_tokens: int, threshold: float) -> dict[str, list[str]]:
    """Run kukri in batch mode; return doc_id → list[chunk_text]."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for doc_id, text in docs.items():
            f.write(json.dumps({"id": doc_id, "text": text}) + "\n")
        jsonl_in = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        jsonl_out = f.name

    result = subprocess.run(
        [
            "go", "run", "./cmd/chunker",
            "-batch",
            "-input",      jsonl_in,
            "-output",     jsonl_out,
            "-model",      _MODEL_PATH,
            "-max-tokens", str(max_tokens),
            "-threshold",  str(threshold),
        ],
        capture_output=True, text=True, cwd=_REPO_ROOT,
    )
    os.unlink(jsonl_in)

    if result.returncode != 0:
        print(f"  [kukri] subprocess error: {result.stderr[:300]}", file=sys.stderr)
        os.unlink(jsonl_out)
        return {}

    chunks: dict[str, list[str]] = {}
    with open(jsonl_out) as f:
        for line in f:
            row = json.loads(line)
            chunks[row["id"]] = [c["text"] for c in row.get("chunks", []) if c.get("text")]
    os.unlink(jsonl_out)
    return chunks


# ---------------------------------------------------------------------------
# Python chunkers
# ---------------------------------------------------------------------------

def chunk_python(docs: dict[str, str], chunk_fn) -> dict[str, list[str]]:
    return {doc_id: [c for c in chunk_fn(text) if c.strip()] for doc_id, text in docs.items()}


# ---------------------------------------------------------------------------
# Coverage calculation
# ---------------------------------------------------------------------------

def coverage_rate(
    gold: list[dict],
    chunks_by_doc: dict[str, list[str]],
) -> dict:
    """
    For each (doc_id, answer_span) in gold, check if answer_span is
    fully contained in at least one chunk of the same doc.
    """
    total = 0
    covered = 0
    uncovered_examples = []

    for item in gold:
        doc_id = item["doc_id"]
        span = item["answer_span"]
        doc_chunks = chunks_by_doc.get(doc_id, [])
        total += 1
        if any(_contains(ch, span) for ch in doc_chunks):
            covered += 1
        elif len(uncovered_examples) < 3:
            uncovered_examples.append(span[:80])

    rate = covered / total if total > 0 else 0.0
    return {
        "total_gold":       total,
        "covered":          covered,
        "coverage_rate":    round(rate, 4),
        "uncovered_sample": uncovered_examples,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Answer Coverage Rate benchmark")
    ap.add_argument("--gold",       required=True, help="gold.jsonl from generate_gold.py")
    ap.add_argument("--corpus",     required=True,
                    help="corpus.jsonl  ({\"id\":...,\"text\":...}) — same docs used for gold generation")
    ap.add_argument("--max-tokens", type=int,   default=512)
    ap.add_argument("--threshold",  type=float, default=0.7)
    ap.add_argument("--overlap",    type=int,   default=50)
    ap.add_argument("--max-docs",   type=int,   default=None)
    ap.add_argument("--output",     default="-")
    ap.add_argument("--strategies", nargs="+",
                    default=["kukri", "fixed_token", "semantic", "sentence", "recursive"],
                    choices=["kukri", "fixed_token", "semantic", "sentence", "recursive"])
    args = ap.parse_args()

    # --- load gold pairs --------------------------------------------------------
    gold: list[dict] = []
    with open(args.gold) as f:
        for line in f:
            gold.append(json.loads(line))
    print(f"gold pairs: {len(gold)}", file=sys.stderr)

    # --- load corpus ------------------------------------------------------------
    docs: dict[str, str] = {}
    with open(args.corpus) as f:
        for line in f:
            row = json.loads(line)
            doc_id = str(row["id"])
            docs[doc_id] = row["text"]
    if args.max_docs:
        doc_ids = list(docs)[:args.max_docs]
        docs = {k: docs[k] for k in doc_ids}
    print(f"corpus: {len(docs)} docs", file=sys.stderr)

    # Filter gold to docs in corpus
    gold = [g for g in gold if g["doc_id"] in docs]
    print(f"gold pairs after corpus filter: {len(gold)}", file=sys.stderr)

    # --- init Python chunkers ---------------------------------------------------
    mt = args.max_tokens
    token_chunker    = TokenChunker(chunk_size=mt, chunk_overlap=args.overlap)
    semantic_chunker = SemanticChunker(chunk_size=mt, threshold=args.threshold)
    sentence_splitter = SentenceSplitter(chunk_size=mt, chunk_overlap=args.overlap)
    recursive_spl    = RecursiveCharacterTextSplitter(chunk_size=mt * 4, chunk_overlap=100)

    strategy_fns = {
        "fixed_token": lambda t: [c.text for c in token_chunker.chunk(t)],
        "semantic":    lambda t: [c.text for c in semantic_chunker.chunk(t)],
        "sentence":    lambda t: sentence_splitter.split_text(t),
        "recursive":   lambda t: recursive_spl.split_text(t),
    }

    # --- run strategies ---------------------------------------------------------
    results = []
    for name in args.strategies:
        print(f"\n{'='*50}\nStrategy: {name}\n{'='*50}", file=sys.stderr)

        if name == "kukri":
            chunks_by_doc = chunk_kukri_batch(docs, mt, args.threshold)
        else:
            fn = strategy_fns[name]
            chunks_by_doc = chunk_python(docs, fn)

        total_chunks = sum(len(v) for v in chunks_by_doc.values())
        avg_len = float(np.mean([
            len(ch.split())
            for clist in chunks_by_doc.values()
            for ch in clist
        ])) if total_chunks > 0 else 0.0

        print(f"  chunks: {total_chunks}, avg_tokens: {avg_len:.1f}", file=sys.stderr)

        metrics = coverage_rate(gold, chunks_by_doc)
        metrics["strategy"]    = name
        metrics["total_chunks"] = total_chunks
        metrics["avg_tokens"]  = round(avg_len, 1)
        results.append(metrics)

        print(
            f"  coverage: {metrics['covered']}/{metrics['total_gold']} "
            f"= {metrics['coverage_rate']:.1%}",
            file=sys.stderr,
        )

    # --- print table -----------------------------------------------------------
    headers = ["Strategy", "Coverage↑", "Covered/Total", "Chunks", "AvgTok"]
    rows = [
        [
            r["strategy"],
            f"{r['coverage_rate']:.1%}",
            f"{r['covered']}/{r['total_gold']}",
            r["total_chunks"],
            r["avg_tokens"],
        ]
        for r in results
    ]
    print("\n" + tabulate(rows, headers=headers, tablefmt="github"), file=sys.stderr)

    out = open(args.output, "w") if args.output != "-" else sys.stdout
    json.dump(results, out, indent=2)
    if args.output != "-":
        print(f"\nresults saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
