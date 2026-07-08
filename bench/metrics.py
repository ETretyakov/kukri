#!/usr/bin/env python3
"""
Compute chunking quality metrics across multiple strategies.

Usage:
  python bench/metrics.py \
    --model all-MiniLM-L6-v2 \
    kukri:chunks_kukri.json \
    chonkie_semantic:chunks_chonkie_sem.json \
    langchain_md:chunks_langchain.json
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tabulate import tabulate


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def intra_coherence(embeddings: list[np.ndarray]) -> float:
    """Mean cosine similarity of all sentence-pair combinations within a chunk."""
    n = len(embeddings)
    if n < 2:
        return 1.0
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += cosine_sim(embeddings[i], embeddings[j])
            count += 1
    return total / count


def split_sentences(text: str) -> list[str]:
    import re
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
    return [p.strip() for p in parts if p.strip()]


def compute_metrics(chunks: list[dict], model: SentenceTransformer) -> dict:
    texts = [c["text"] for c in chunks]
    token_counts = [c.get("token_count", len(c["text"].split())) for c in chunks]

    # --- size stats ---
    avg_tokens = float(np.mean(token_counts))
    std_tokens = float(np.std(token_counts))
    max_tokens = int(np.max(token_counts))

    # --- boundary respect: % chunks that don't cross section boundaries ---
    # A chunk crosses a boundary if its section_path changed from the previous chunk.
    boundary_violations = 0
    for i in range(1, len(chunks)):
        if chunks[i].get("section_path") != chunks[i-1].get("section_path"):
            # This is a boundary — expected. We count violations only when
            # section_path is non-empty and matches but the section title differs.
            pass
    # Simpler proxy: % of chunks with non-empty section_path (structural awareness)
    with_section = sum(1 for c in chunks if c.get("section_path"))
    boundary_respect = with_section / len(chunks) if chunks else 0.0

    # --- semantic coherence & contrast (sample for speed) ---
    sample_size = min(200, len(chunks))
    step = max(1, len(chunks) // sample_size)
    sampled = chunks[::step][:sample_size]

    coherence_scores = []
    contrast_scores  = []

    for i, chunk in enumerate(sampled):
        sentences = split_sentences(chunk["text"])
        if len(sentences) >= 2:
            embs = model.encode(sentences, show_progress_bar=False)
            coherence_scores.append(intra_coherence(embs))

        # inter-chunk contrast with next chunk
        if i + 1 < len(sampled):
            sents_a = split_sentences(chunk["text"])
            sents_b = split_sentences(sampled[i+1]["text"])
            if sents_a and sents_b:
                emb_a = model.encode([sents_a[-1]], show_progress_bar=False)[0]
                emb_b = model.encode([sents_b[0]],  show_progress_bar=False)[0]
                contrast_scores.append(cosine_sim(emb_a, emb_b))

    coherence = float(np.mean(coherence_scores)) if coherence_scores else 0.0
    contrast  = float(np.mean(contrast_scores))  if contrast_scores  else 0.0

    return {
        "chunks":           len(chunks),
        "avg_tokens":       round(avg_tokens, 1),
        "std_tokens":       round(std_tokens, 1),
        "max_tokens":       max_tokens,
        "coherence":        round(coherence, 3),   # higher = better
        "contrast":         round(contrast,  3),   # lower  = better
        "boundary_respect": round(boundary_respect, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default="all-MiniLM-L6-v2", help="sentence-transformers model name or path")
    ap.add_argument("--output", default="-", help="JSON output path (default: stdout)")
    ap.add_argument("files",    nargs="+", metavar="name:path.json",
                    help="Strategy results in name:path format")
    args = ap.parse_args()

    print(f"loading embedding model '{args.model}'...", file=sys.stderr)
    model = SentenceTransformer(args.model)

    results = {}
    for entry in args.files:
        if ":" not in entry:
            print(f"skip malformed entry (expected name:path): {entry}", file=sys.stderr)
            continue
        name, path = entry.split(":", 1)
        data = json.loads(Path(path).read_text())
        chunks = data.get("chunks", data) if isinstance(data, dict) else data

        print(f"computing metrics for '{name}' ({len(chunks)} chunks)...", file=sys.stderr)
        t0 = time.time()
        metrics = compute_metrics(chunks, model)
        metrics["elapsed_s"] = round(time.time() - t0, 2)
        results[name] = metrics

    # pretty table
    headers = ["Strategy", "Chunks", "AvgTok", "StdTok", "MaxTok",
               "Coherence↑", "Contrast↓", "BoundaryRespect↑", "Time(s)"]
    rows = [
        [
            name,
            m["chunks"],
            m["avg_tokens"],
            m["std_tokens"],
            m["max_tokens"],
            m["coherence"],
            m["contrast"],
            m["boundary_respect"],
            m["elapsed_s"],
        ]
        for name, m in results.items()
    ]
    print("\n" + tabulate(rows, headers=headers, tablefmt="github"), file=sys.stderr)

    out = open(args.output, "w") if args.output != "-" else sys.stdout
    json.dump(results, out, indent=2)


if __name__ == "__main__":
    main()
