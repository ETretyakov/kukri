#!/usr/bin/env python3
"""
Sweep kukri threshold to find the sweet spot for Hit Rate@k.
Runs kukri at multiple threshold values and reports Coverage + Hit@k.

Usage:
  python bench/sweep_threshold.py \
    --gold bench/gold.jsonl \
    --corpus corpus.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tabulate import tabulate


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MODEL_PATH = os.path.join(_REPO_ROOT, "models", "KnightsAnalytics_all-MiniLM-L6-v2")
_DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_HIT_KS = (1, 3, 5)

# Models that need a query prefix for asymmetric retrieval.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_QUERY_PREFIX_MODELS = {"BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5"}


def _norm(s: str) -> str:
    return " ".join(s.split())


def _contains(chunk_text: str, span: str) -> bool:
    if span in chunk_text:
        return True
    return _norm(span) in _norm(chunk_text)


def run_markdown_splitter(
    docs: dict[str, str],
    max_tokens: int,
) -> dict[str, list[dict]]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens * 4,  # ~4 chars per word approximation
        chunk_overlap=0,
    )
    result: dict[str, list[dict]] = {}
    for doc_id, text in docs.items():
        header_docs = header_splitter.split_text(text)
        split_docs = char_splitter.split_documents(header_docs)
        chunks = [
            {"text": d.page_content, "embed_text": d.page_content}
            for d in split_docs if d.page_content.strip()
        ]
        result[doc_id] = chunks
    return result


def run_kukri(
    docs: dict[str, str],
    max_tokens: int,
    threshold: float,
    prepend_section_path: bool = False,
) -> dict[str, list[dict]]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for doc_id, text in docs.items():
            f.write(json.dumps({"id": doc_id, "text": text}) + "\n")
        jsonl_in = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        jsonl_out = f.name

    binary = os.environ.get("KUKRI_BIN", "go run ./cmd/chunker")
    base_cmd = binary.split() if " " in binary else [binary]
    cmd = base_cmd + [
        "-batch", "-input", jsonl_in, "-output", jsonl_out,
        "-max-tokens", str(max_tokens), "-threshold", str(threshold)]
    if prepend_section_path:
        cmd.append("-prepend-section-path")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO_ROOT if base_cmd[0] == "go" else None)
    os.unlink(jsonl_in)
    if result.returncode != 0:
        print(f"kukri error: {result.stderr[:200]}", file=sys.stderr)
        os.unlink(jsonl_out)
        return {}

    # chunks_by_doc: {doc_id: [{"text": ..., "embed_text": ...}]}
    chunks: dict[str, list[dict]] = {}
    with open(jsonl_out) as f:
        for line in f:
            row = json.loads(line)
            chunks[row["id"]] = [
                {"text": c["text"], "embed_text": c.get("embed_text") or c["text"]}
                for c in row.get("chunks", []) if c.get("text")
            ]
    os.unlink(jsonl_out)
    return chunks


def compute_metrics(
    gold: list[dict],
    chunks_by_doc: dict[str, list[dict]],
    q_embeddings: np.ndarray,
    st_model: SentenceTransformer,
) -> dict:
    # coverage — checked against display text (no prefix)
    total, covered = 0, 0
    for item in gold:
        total += 1
        if any(_contains(ch["text"], item["answer_span"]) for ch in chunks_by_doc.get(item["doc_id"], [])):
            covered += 1

    # embed chunks using embed_text (includes section-path prefix)
    needed = {item["doc_id"] for item in gold if item["doc_id"] in chunks_by_doc}
    doc_embs: dict[str, np.ndarray] = {}
    doc_texts: dict[str, list[str]] = {}
    for doc_id in needed:
        chunks = chunks_by_doc.get(doc_id, [])
        if chunks:
            doc_texts[doc_id] = [ch["text"] for ch in chunks]
            doc_embs[doc_id] = st_model.encode(
                [ch["embed_text"] for ch in chunks],
                batch_size=256, normalize_embeddings=True, show_progress_bar=False,
            )

    hits = {k: 0 for k in _HIT_KS}
    ret_total = 0
    for i, item in enumerate(gold):
        doc_id = item["doc_id"]
        if doc_id not in doc_embs:
            continue
        ret_total += 1
        sims = doc_embs[doc_id] @ q_embeddings[i]
        for k in _HIT_KS:
            top_k = np.argpartition(sims, -min(k, len(sims)))[-k:]
            if any(_contains(doc_texts[doc_id][j], item["answer_span"]) for j in top_k):
                hits[k] += 1

    total_chunks = sum(len(v) for v in chunks_by_doc.values())
    avg_tok = float(np.mean([
        len(ch["text"].split()) for clist in chunks_by_doc.values() for ch in clist
    ])) if total_chunks > 0 else 0.0

    return {
        "coverage":     round(covered / total, 4) if total else 0.0,
        "avg_tokens":   round(avg_tok, 1),
        "total_chunks": total_chunks,
        **{f"hit@{k}": round(hits[k] / ret_total, 4) if ret_total else 0.0 for k in _HIT_KS},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold",       required=True)
    ap.add_argument("--corpus",     required=True)
    ap.add_argument("--max-tokens", type=int,   default=512)
    ap.add_argument("--thresholds",  nargs="+", type=float,
                    default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--prepend-section-path", action="store_true",
                    help="pass -prepend-section-path to kukri (for BGE/E5 models)")
    ap.add_argument("--embed-model", default=_DEFAULT_EMBED_MODEL,
                    help="HuggingFace embedding model for Hit@k (default: all-MiniLM-L6-v2)")
    ap.add_argument("--max-docs",    type=int, default=None,
                    help="limit corpus to first N docs for quick iteration")
    ap.add_argument("--baseline", action="store_true",
                    help="also run markdown splitter baseline with the same model")
    args = ap.parse_args()

    gold: list[dict] = []
    with open(args.gold) as f:
        for line in f:
            gold.append(json.loads(line))

    docs: dict[str, str] = {}
    with open(args.corpus) as f:
        for line in f:
            row = json.loads(line)
            docs[str(row["id"])] = row["text"]
    if args.max_docs:
        doc_ids = list(docs)[: args.max_docs]
        docs = {k: docs[k] for k in doc_ids}
    gold = [g for g in gold if g["doc_id"] in docs]
    print(f"{len(gold)} gold pairs, {len(docs)} docs", file=sys.stderr)

    print(f"loading {args.embed_model}...", file=sys.stderr)
    st_model = SentenceTransformer(args.embed_model)
    raw_questions = [g["question"] for g in gold]
    if args.embed_model in _QUERY_PREFIX_MODELS:
        raw_questions = [_BGE_QUERY_PREFIX + q for q in raw_questions]
    q_embeddings = st_model.encode(
        raw_questions,
        batch_size=256, normalize_embeddings=True, show_progress_bar=True,
    )

    def _append_row(rows, label, m):
        rows.append([
            label,
            f"{m['coverage']:.1%}",
            f"{m['hit@1']:.1%}",
            f"{m['hit@3']:.1%}",
            f"{m['hit@5']:.1%}",
            m["avg_tokens"],
            m["total_chunks"],
        ])
        print(f"  coverage={m['coverage']:.1%}  hit@1={m['hit@1']:.1%}  "
              f"hit@3={m['hit@3']:.1%}  hit@5={m['hit@5']:.1%}  "
              f"avg_tok={m['avg_tokens']}  chunks={m['total_chunks']}", file=sys.stderr)

    rows = []
    for t in args.thresholds:
        label = f"kukri t={t}"
        print(f"\n{label}", file=sys.stderr)
        chunks_by_doc = run_kukri(
            docs, args.max_tokens, t,
            prepend_section_path=args.prepend_section_path,
        )
        _append_row(rows, label, compute_metrics(gold, chunks_by_doc, q_embeddings, st_model))

    if args.baseline:
        label = f"markdown mt={args.max_tokens}"
        print(f"\n{label}", file=sys.stderr)
        chunks_by_doc = run_markdown_splitter(docs, args.max_tokens)
        _append_row(rows, label, compute_metrics(gold, chunks_by_doc, q_embeddings, st_model))

    print("\n" + tabulate(
        rows,
        headers=["config", "Coverage↑", "Hit@1↑", "Hit@3↑", "Hit@5↑", "AvgTok", "Chunks"],
        tablefmt="github",
    ))


if __name__ == "__main__":
    main()
