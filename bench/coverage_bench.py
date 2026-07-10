#!/usr/bin/env python3
"""
Answer Coverage Rate + Hit Rate@k benchmark.

Coverage: % of gold answer_spans fully contained in at least one chunk.
Hit Rate@k: % of questions whose top-k most similar chunks contain the answer,
  simulating a real retrieval step with all-MiniLM-L6-v2 embeddings.

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

import numpy as np
from sentence_transformers import SentenceTransformer
from tabulate import tabulate

from chonkie import SemanticChunker, TokenChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from llama_index.core.node_parser import SentenceSplitter


# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MODEL_PATH = os.path.join(_REPO_ROOT, "models", "KnightsAnalytics_all-MiniLM-L6-v2")
_ST_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"

_HIT_KS = (1, 3, 5)


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

def chunk_kukri_batch(docs: dict[str, str], max_tokens: int, threshold: float, merge_blocks: bool = False) -> dict[str, list[str]]:
    """Run kukri in batch mode; return doc_id → list[chunk_text]."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for doc_id, text in docs.items():
            f.write(json.dumps({"id": doc_id, "text": text}) + "\n")
        jsonl_in = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        jsonl_out = f.name

    cmd = [
        "go", "run", "./cmd/chunker",
        "-batch",
        "-input",      jsonl_in,
        "-output",     jsonl_out,
        "-model",      _MODEL_PATH,
        "-max-tokens", str(max_tokens),
        "-threshold",  str(threshold),
    ]
    if merge_blocks:
        cmd.append("-merge-blocks")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO_ROOT)
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

def coverage_rate(gold: list[dict], chunks_by_doc: dict[str, list[str]]) -> dict:
    """% of answer_spans fully contained in at least one chunk."""
    total, covered = 0, 0
    uncovered_examples: list[str] = []

    for item in gold:
        doc_chunks = chunks_by_doc.get(item["doc_id"], [])
        total += 1
        if any(_contains(ch, item["answer_span"]) for ch in doc_chunks):
            covered += 1
        elif len(uncovered_examples) < 3:
            uncovered_examples.append(item["answer_span"][:80])

    rate = covered / total if total > 0 else 0.0
    return {
        "total_gold":       total,
        "covered":          covered,
        "coverage_rate":    round(rate, 4),
        "uncovered_sample": uncovered_examples,
    }


# ---------------------------------------------------------------------------
# Hit Rate@k calculation
# ---------------------------------------------------------------------------

def hit_rate_at_k(
    gold: list[dict],
    chunks_by_doc: dict[str, list[str]],
    q_embeddings: np.ndarray,
    st_model: SentenceTransformer,
    ks: tuple[int, ...] = _HIT_KS,
) -> dict[str, float]:
    """
    For each gold pair, embed all chunks of the same doc, find top-k by
    cosine similarity to the question, and check if any contain answer_span.

    Embeddings are normalised to unit length, so cosine sim = dot product.
    """
    # Build per-doc chunk embeddings lazily (only for docs that appear in gold).
    needed_docs = {item["doc_id"] for item in gold if item["doc_id"] in chunks_by_doc}
    doc_chunk_texts: dict[str, list[str]] = {}
    doc_chunk_embs:  dict[str, np.ndarray] = {}

    print(f"  embedding chunks for {len(needed_docs)} docs...", file=sys.stderr)
    for doc_id in needed_docs:
        texts = chunks_by_doc[doc_id]
        if not texts:
            continue
        doc_chunk_texts[doc_id] = texts
        doc_chunk_embs[doc_id] = st_model.encode(
            texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False
        )

    hits = {k: 0 for k in ks}
    total = 0

    for i, item in enumerate(gold):
        doc_id = item["doc_id"]
        if doc_id not in doc_chunk_embs:
            continue

        total += 1
        q_emb      = q_embeddings[i]                   # (384,) normalised
        chunk_embs = doc_chunk_embs[doc_id]             # (n, 384) normalised
        chunk_texts = doc_chunk_texts[doc_id]

        sims = chunk_embs @ q_emb                       # cosine similarities

        for k in ks:
            top_k_idx = np.argpartition(sims, -min(k, len(sims)))[-k:]
            if any(_contains(chunk_texts[j], item["answer_span"]) for j in top_k_idx):
                hits[k] += 1

    return {f"hit@{k}": round(hits[k] / total, 4) if total > 0 else 0.0 for k in ks}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Answer Coverage Rate + Hit Rate@k benchmark")
    ap.add_argument("--gold",         required=True)
    ap.add_argument("--corpus",       required=True)
    ap.add_argument("--max-tokens",   type=int,   default=512)
    ap.add_argument("--threshold",    type=float, default=0.7)
    ap.add_argument("--overlap",      type=int,   default=50)
    ap.add_argument("--max-docs",     type=int,   default=None)
    ap.add_argument("--output",       default="-")
    ap.add_argument("--no-retrieval",  action="store_true",
                    help="skip Hit Rate@k (faster, no sentence-transformers inference)")
    ap.add_argument("--merge-blocks",  action="store_true",
                    help="pass -merge-blocks to kukri (cross-block semantic merging)")
    ap.add_argument("--strategies",   nargs="+",
                    default=["kukri", "fixed_token", "semantic", "sentence", "recursive", "markdown"],
                    choices=["kukri", "fixed_token", "semantic", "sentence", "recursive", "markdown"])
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
            docs[str(row["id"])] = row["text"]
    if args.max_docs:
        doc_ids = list(docs)[: args.max_docs]
        docs = {k: docs[k] for k in doc_ids}
    print(f"corpus: {len(docs)} docs", file=sys.stderr)

    gold = [g for g in gold if g["doc_id"] in docs]
    print(f"gold pairs after corpus filter: {len(gold)}", file=sys.stderr)

    # --- embed questions once (shared across all strategies) --------------------
    st_model: SentenceTransformer | None = None
    q_embeddings: np.ndarray | None = None
    if not args.no_retrieval:
        print("\nloading sentence-transformers model...", file=sys.stderr)
        st_model = SentenceTransformer(_ST_MODEL)
        questions = [g["question"] for g in gold]
        print(f"embedding {len(questions)} questions...", file=sys.stderr)
        q_embeddings = st_model.encode(
            questions, batch_size=256, normalize_embeddings=True, show_progress_bar=True
        )

    # --- init Python chunkers ---------------------------------------------------
    mt = args.max_tokens
    token_chunker    = TokenChunker(chunk_size=mt, chunk_overlap=args.overlap)
    semantic_chunker = SemanticChunker(chunk_size=mt, threshold=args.threshold)
    sentence_splitter = SentenceSplitter(chunk_size=mt, chunk_overlap=args.overlap)
    recursive_spl    = RecursiveCharacterTextSplitter(chunk_size=mt * 4, chunk_overlap=100)

    md_header_spl = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    md_char_spl = RecursiveCharacterTextSplitter(chunk_size=mt * 4, chunk_overlap=100)

    def _chunk_markdown_splitter(text: str) -> list[str]:
        header_docs = md_header_spl.split_text(text)
        split_docs = md_char_spl.split_documents(header_docs)
        return [doc.page_content for doc in split_docs]

    strategy_fns = {
        "fixed_token": lambda t: [c.text for c in token_chunker.chunk(t)],
        "semantic":    lambda t: [c.text for c in semantic_chunker.chunk(t)],
        "sentence":    lambda t: sentence_splitter.split_text(t),
        "recursive":   lambda t: recursive_spl.split_text(t),
        "markdown":    _chunk_markdown_splitter,
    }

    # --- run strategies ---------------------------------------------------------
    results = []
    for name in args.strategies:
        print(f"\n{'='*50}\nStrategy: {name}\n{'='*50}", file=sys.stderr)

        if name == "kukri":
            chunks_by_doc = chunk_kukri_batch(docs, mt, args.threshold, merge_blocks=args.merge_blocks)
        else:
            chunks_by_doc = chunk_python(docs, strategy_fns[name])

        total_chunks = sum(len(v) for v in chunks_by_doc.values())
        avg_len = float(np.mean([
            len(ch.split())
            for clist in chunks_by_doc.values()
            for ch in clist
        ])) if total_chunks > 0 else 0.0

        print(f"  chunks: {total_chunks}, avg_tokens: {avg_len:.1f}", file=sys.stderr)

        metrics = coverage_rate(gold, chunks_by_doc)
        metrics["strategy"]     = name
        metrics["total_chunks"] = total_chunks
        metrics["avg_tokens"]   = round(avg_len, 1)

        print(
            f"  coverage: {metrics['covered']}/{metrics['total_gold']} "
            f"= {metrics['coverage_rate']:.1%}",
            file=sys.stderr,
        )

        if st_model is not None and q_embeddings is not None:
            hr = hit_rate_at_k(gold, chunks_by_doc, q_embeddings, st_model)
            metrics.update(hr)
            print(
                "  " + "  ".join(f"{k}: {v:.1%}" for k, v in hr.items()),
                file=sys.stderr,
            )

        results.append(metrics)

    # --- print table -----------------------------------------------------------
    has_retrieval = "hit@1" in results[0] if results else False
    if has_retrieval:
        headers = ["Strategy", "Coverage↑", "Hit@1↑", "Hit@3↑", "Hit@5↑", "AvgTok", "Chunks"]
        rows = [
            [
                r["strategy"],
                f"{r['coverage_rate']:.1%}",
                f"{r.get('hit@1', 0):.1%}",
                f"{r.get('hit@3', 0):.1%}",
                f"{r.get('hit@5', 0):.1%}",
                r["avg_tokens"],
                r["total_chunks"],
            ]
            for r in results
        ]
    else:
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
