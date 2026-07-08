#!/usr/bin/env python3
"""
BEIR benchmark: compare chunking strategies on retrieval quality.

Downloads a BEIR dataset (default: scifact), chunks the corpus with each
strategy, embeds everything, runs retrieval per query, reports NDCG@10 / Recall@10 / MRR.

Usage:
  python bench/beir_bench.py --dataset scifact --output /tmp/beir_results.json
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tabulate import tabulate

from chonkie import SemanticChunker, TokenChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import SentenceSplitter


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ndcg_at_k(relevant: set, ranked: list, k: int) -> float:
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, doc_id in enumerate(ranked[:k])
        if doc_id in relevant
    )
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def recall_at_k(relevant: set, ranked: list, k: int) -> float:
    hits = len(set(ranked[:k]) & relevant)
    return hits / len(relevant) if relevant else 0.0


def mrr(relevant: set, ranked: list) -> float:
    for i, doc_id in enumerate(ranked):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def build_index(corpus: dict, chunk_fn, max_docs: int | None, label: str):
    """Chunk corpus docs, return (chunk_texts, chunk_doc_ids)."""
    doc_ids = list(corpus.keys())
    if max_docs:
        doc_ids = doc_ids[:max_docs]

    chunk_texts, chunk_doc_ids = [], []
    for doc_id in doc_ids:
        doc = corpus[doc_id]
        text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
        for chunk in chunk_fn(text):
            if chunk.strip():
                chunk_texts.append(chunk)
                chunk_doc_ids.append(doc_id)

    print(f"  [{label}] {len(chunk_texts)} chunks from {len(doc_ids)} docs", file=sys.stderr)
    return chunk_texts, chunk_doc_ids


def retrieve(query_emb: np.ndarray, chunk_embs: np.ndarray, chunk_doc_ids: list, top_k: int = 100) -> list:
    """Return ranked doc_ids by best chunk score."""
    sims = chunk_embs @ query_emb
    doc_scores: dict[str, float] = defaultdict(float)
    for j, doc_id in enumerate(chunk_doc_ids):
        if sims[j] > doc_scores[doc_id]:
            doc_scores[doc_id] = float(sims[j])
    return [d for d, _ in sorted(doc_scores.items(), key=lambda x: -x[1])][:top_k]


def evaluate(
    name: str,
    corpus: dict,
    queries: dict,
    qrels: dict,
    model: SentenceTransformer,
    chunk_fn,
    max_docs: int | None,
) -> dict:
    chunk_texts, chunk_doc_ids = build_index(corpus, chunk_fn, max_docs, name)

    print(f"  [{name}] embedding {len(chunk_texts)} chunks...", file=sys.stderr)
    chunk_embs = model.encode(
        chunk_texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True
    )

    query_ids = [qid for qid in queries if qid in qrels]
    query_texts = [queries[qid] for qid in query_ids]

    print(f"  [{name}] evaluating {len(query_ids)} queries...", file=sys.stderr)
    query_embs = model.encode(
        query_texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True
    )

    ndcg_scores, recall_scores, mrr_scores = [], [], []
    for i, qid in enumerate(query_ids):
        relevant = {doc_id for doc_id, score in qrels[qid].items() if score > 0}
        if not relevant:
            continue
        ranked = retrieve(query_embs[i], chunk_embs, chunk_doc_ids)
        ndcg_scores.append(ndcg_at_k(relevant, ranked, 10))
        recall_scores.append(recall_at_k(relevant, ranked, 10))
        mrr_scores.append(mrr(relevant, ranked))

    avg_tokens = float(np.mean([len(t.split()) for t in chunk_texts]))

    return {
        "strategy":        name,
        "total_chunks":    len(chunk_texts),
        "chunks_per_doc":  round(len(chunk_texts) / max(1, len(set(chunk_doc_ids))), 1),
        "avg_tokens":      round(avg_tokens, 1),
        "ndcg@10":         round(float(np.mean(ndcg_scores)),   4),
        "recall@10":       round(float(np.mean(recall_scores)), 4),
        "mrr":             round(float(np.mean(mrr_scores)),    4),
    }


# ---------------------------------------------------------------------------
# Kukri batch chunker
# ---------------------------------------------------------------------------

# Project root is two levels up from bench/
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MODEL_PATH = os.path.join(_REPO_ROOT, "models", "KnightsAnalytics_all-MiniLM-L6-v2")


def make_kukri_chunker(max_tokens: int, threshold: float):
    """
    Returns a function (corpus: dict) -> dict[doc_id, list[str]] that
    calls the kukri Go binary in batch mode (one subprocess, model loaded once).
    The returned callable accepts a single text str for API consistency,
    but we pre-build the whole corpus index on first call via a closure trick.

    Since kukri loads the model once per process and processes all docs in one
    JSONL pass, we run it over the full corpus in evaluate() via a side-channel.
    The per-text lambda is used as a compatibility shim.
    """
    def chunk_text(text: str) -> list[str]:
        # Single-text fallback (used for small tests outside evaluate()).
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(text)
            tmp = f.name
        try:
            result = subprocess.run(
                ["go", "run", "./cmd/chunker",
                 "-input", tmp,
                 "-model", _MODEL_PATH,
                 "-max-tokens", str(max_tokens),
                 "-threshold", str(threshold)],
                capture_output=True, text=True, cwd=_REPO_ROOT,
            )
            if result.returncode != 0:
                return [text]
            data = json.loads(result.stdout)
            return [c["text"] for c in data.get("chunks", [])]
        finally:
            os.unlink(tmp)

    return chunk_text


def evaluate_kukri_batch(
    corpus: dict,
    queries: dict,
    qrels: dict,
    model: SentenceTransformer,
    max_tokens: int,
    threshold: float,
    max_docs: int | None,
) -> dict:
    """Run kukri over entire corpus in one subprocess call via batch JSONL mode."""
    doc_ids = list(corpus.keys())
    if max_docs:
        doc_ids = doc_ids[:max_docs]

    # Write JSONL input
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for doc_id in doc_ids:
            doc = corpus[doc_id]
            text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
            f.write(json.dumps({"id": doc_id, "text": text}) + "\n")
        jsonl_in = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        jsonl_out = f.name

    print(f"  [kukri] running batch over {len(doc_ids)} docs...", file=sys.stderr)
    result = subprocess.run(
        ["go", "run", "./cmd/chunker",
         "-batch",
         "-input",      jsonl_in,
         "-output",     jsonl_out,
         "-model",      _MODEL_PATH,
         "-max-tokens", str(max_tokens),
         "-threshold",  str(threshold)],
        capture_output=True, text=True, cwd=_REPO_ROOT,
    )
    os.unlink(jsonl_in)

    if result.returncode != 0:
        print(f"  [kukri] error: {result.stderr}", file=sys.stderr)
        os.unlink(jsonl_out)
        return {}

    # Parse JSONL output
    chunk_texts, chunk_doc_ids = [], []
    with open(jsonl_out) as f:
        for line in f:
            row = json.loads(line)
            for chunk in row.get("chunks", []):
                if chunk["text"].strip():
                    chunk_texts.append(chunk["text"])
                    chunk_doc_ids.append(row["id"])
    os.unlink(jsonl_out)

    print(f"  [kukri] {len(chunk_texts)} chunks, embedding...", file=sys.stderr)
    chunk_embs = model.encode(
        chunk_texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True
    )

    query_ids   = [qid for qid in queries if qid in qrels]
    query_texts = [queries[qid] for qid in query_ids]
    print(f"  [kukri] evaluating {len(query_ids)} queries...", file=sys.stderr)
    query_embs = model.encode(
        query_texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True
    )

    ndcg_scores, recall_scores, mrr_scores = [], [], []
    for i, qid in enumerate(query_ids):
        relevant = {doc_id for doc_id, score in qrels[qid].items() if score > 0}
        if not relevant:
            continue
        ranked = retrieve(query_embs[i], chunk_embs, chunk_doc_ids)
        ndcg_scores.append(ndcg_at_k(relevant, ranked, 10))
        recall_scores.append(recall_at_k(relevant, ranked, 10))
        mrr_scores.append(mrr(relevant, ranked))

    avg_tokens = float(np.mean([len(t.split()) for t in chunk_texts])) if chunk_texts else 0

    return {
        "strategy":       "kukri",
        "total_chunks":   len(chunk_texts),
        "chunks_per_doc": round(len(chunk_texts) / max(1, len(doc_ids)), 1),
        "avg_tokens":     round(avg_tokens, 1),
        "ndcg@10":        round(float(np.mean(ndcg_scores)),   4),
        "recall@10":      round(float(np.mean(recall_scores)), 4),
        "mrr":            round(float(np.mean(mrr_scores)),    4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",    default="scifact",
                    help="BEIR dataset: scifact | nfcorpus | trec-covid | fiqa ...")
    ap.add_argument("--model",      default="all-MiniLM-L6-v2")
    ap.add_argument("--max-tokens", type=int,   default=512)
    ap.add_argument("--threshold",  type=float, default=0.7)
    ap.add_argument("--overlap",    type=int,   default=50)
    ap.add_argument("--max-docs",   type=int,   default=None,
                    help="limit corpus size for quick smoke-test")
    ap.add_argument("--output",     default="-")
    ap.add_argument("--strategies", nargs="+",
                    default=["kukri", "fixed_token", "semantic", "sentence", "recursive"],
                    choices=["kukri", "fixed_token", "semantic", "sentence", "recursive"])
    args = ap.parse_args()

    # --- load dataset -------------------------------------------------------
    print(f"loading BeIR/{args.dataset} from HuggingFace...", file=sys.stderr)
    corpus_ds  = load_dataset(f"BeIR/{args.dataset}",       "corpus",  split="corpus",  trust_remote_code=True)
    queries_ds = load_dataset(f"BeIR/{args.dataset}",       "queries", split="queries", trust_remote_code=True)
    qrels_ds   = load_dataset(f"BeIR/{args.dataset}-qrels",            split="test",    trust_remote_code=True)

    corpus  = {str(row["_id"]): {"title": row.get("title", ""), "text": row["text"]} for row in corpus_ds}
    queries = {str(row["_id"]): row["text"] for row in queries_ds}
    qrels: dict[str, dict] = defaultdict(dict)
    for row in qrels_ds:
        qrels[str(row["query-id"])][str(row["corpus-id"])] = row["score"]

    print(f"corpus: {len(corpus)} docs | queries: {len(queries)} | qrels: {len(qrels)}", file=sys.stderr)

    # --- init chunkers once (models loaded here, not per-doc) ---------------
    print("initialising chunkers...", file=sys.stderr)
    mt = args.max_tokens

    token_chunker    = TokenChunker(chunk_size=mt, chunk_overlap=args.overlap)
    semantic_chunker = SemanticChunker(chunk_size=mt, threshold=args.threshold)
    sentence_chunker = SentenceSplitter(chunk_size=mt, chunk_overlap=args.overlap)
    recursive_spl    = RecursiveCharacterTextSplitter(chunk_size=mt * 4, chunk_overlap=100)

    kukri_fn = make_kukri_chunker(args.max_tokens, args.threshold)

    strategy_fns = {
        "kukri":       kukri_fn,
        "fixed_token": lambda t: [c.text for c in token_chunker.chunk(t)],
        "semantic":    lambda t: [c.text for c in semantic_chunker.chunk(t)],
        "sentence":    lambda t: sentence_chunker.split_text(t),
        "recursive":   lambda t: recursive_spl.split_text(t),
    }

    # --- embedding model ----------------------------------------------------
    print(f"loading embedding model '{args.model}'...", file=sys.stderr)
    model = SentenceTransformer(args.model)

    # --- run -----------------------------------------------------------------
    results = []
    for name in args.strategies:
        print(f"\n{'='*50}\nStrategy: {name}\n{'='*50}", file=sys.stderr)
        if name == "kukri":
            result = evaluate_kukri_batch(
                corpus, queries, qrels, model,
                args.max_tokens, args.threshold, args.max_docs,
            )
        else:
            result = evaluate(name, corpus, queries, qrels, model,
                              strategy_fns[name], args.max_docs)
        if result:
            results.append(result)

    # --- print table ---------------------------------------------------------
    headers = ["Strategy", "Chunks", "Chunks/Doc", "AvgTok",
               "NDCG@10↑", "Recall@10↑", "MRR↑"]
    rows = [[
        r["strategy"], r["total_chunks"], r["chunks_per_doc"], r["avg_tokens"],
        r["ndcg@10"], r["recall@10"], r["mrr"],
    ] for r in results]
    print("\n" + tabulate(rows, headers=headers, tablefmt="github"), file=sys.stderr)

    out = open(args.output, "w") if args.output != "-" else sys.stdout
    json.dump(results, out, indent=2)
    if args.output != "-":
        print(f"\nresults saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
