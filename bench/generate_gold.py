#!/usr/bin/env python3
"""
Generate (question, answer_span) gold pairs from documents using Claude.

answer_span is an exact verbatim quote from the document text, used to
measure Answer Coverage Rate across chunking strategies.

Usage:
  python bench/generate_gold.py \
    --input corpus.jsonl \
    --output bench/gold.jsonl \
    --pairs-per-doc 5 \
    --model claude-haiku-4-5
"""
import argparse
import json
import sys
import time
from pathlib import Path

import anthropic


_SYSTEM = (
    "You are a precise QA pair extractor for technical documents.\n"
    "For each pair you generate:\n"
    "  - question: a specific question answerable from a single passage\n"
    "  - answer_span: an EXACT verbatim copy of 1–4 consecutive sentences "
    "from the document that fully answers the question\n\n"
    "Rules:\n"
    "  - answer_span must be copied CHARACTER-FOR-CHARACTER from the source\n"
    "  - answer_span should be 20–200 words\n"
    "  - questions must require that specific passage — not general knowledge\n"
    "  - cover different parts of the document\n"
    "  - return valid JSON only, no surrounding text"
)


def _build_prompt(text: str, n: int) -> str:
    if len(text) > 60_000:
        text = text[:60_000] + "\n[... document truncated ...]"
    return (
        f'Extract {n} QA pairs. Return a JSON array:\n'
        f'[{{"question": "...", "answer_span": "..."}}]\n\n'
        f"Document:\n{text}"
    )


def _normalize(s: str) -> str:
    return " ".join(s.split())


def generate_pairs(
    client: anthropic.Anthropic,
    doc_id: str,
    text: str,
    n: int,
    model: str,
) -> list[dict]:
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=2048,
            system=_SYSTEM,
            messages=[{"role": "user", "content": _build_prompt(text, n)}],
        )
    except anthropic.APIError as exc:
        print(f"  [{doc_id}] API error: {exc}", file=sys.stderr)
        return []

    raw = msg.content[0].text.strip()
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start == -1 or end == 0:
        print(f"  [{doc_id}] no JSON array in response", file=sys.stderr)
        return []

    try:
        pairs = json.loads(raw[start:end])
    except json.JSONDecodeError as exc:
        print(f"  [{doc_id}] JSON parse error: {exc}", file=sys.stderr)
        return []

    verified = []
    norm_text = _normalize(text)
    for p in pairs:
        if not isinstance(p, dict):
            continue
        question = p.get("question", "").strip()
        span = p.get("answer_span", "").strip()
        if not question or not span:
            continue
        # Accept exact match or whitespace-normalised match
        if span in text or _normalize(span) in norm_text:
            verified.append({"doc_id": doc_id, "question": question, "answer_span": span})
        else:
            print(
                f"  [{doc_id}] span not verbatim, skipping: {span[:70]}...",
                file=sys.stderr,
            )
    return verified


def main():
    ap = argparse.ArgumentParser(description="Generate gold QA pairs with Claude")
    ap.add_argument("--input", required=True,
                    help="corpus.jsonl ({\"id\":...,\"text\":...}) or single .md file")
    ap.add_argument("--output", required=True, help="output gold.jsonl")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--pairs-per-doc", type=int, default=5)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--delay", type=float, default=0.3,
                    help="seconds to wait between API calls")
    args = ap.parse_args()

    client = anthropic.Anthropic()

    inp = Path(args.input)
    docs: list[dict] = []
    if inp.suffix == ".jsonl":
        with open(inp) as f:
            for line in f:
                row = json.loads(line)
                docs.append({"id": str(row["id"]), "text": row["text"]})
    else:
        docs.append({"id": inp.stem, "text": inp.read_text()})

    if args.max_docs:
        docs = docs[: args.max_docs]

    print(
        f"generating gold pairs: {len(docs)} docs, "
        f"{args.pairs_per_doc} pairs/doc, model={args.model}",
        file=sys.stderr,
    )

    total = 0
    with open(args.output, "w") as out:
        for i, doc in enumerate(docs):
            print(f"  [{i+1}/{len(docs)}] {doc['id']} ...", file=sys.stderr)
            pairs = generate_pairs(
                client, doc["id"], doc["text"], args.pairs_per_doc, args.model
            )
            for p in pairs:
                out.write(json.dumps(p, ensure_ascii=False) + "\n")
            total += len(pairs)
            print(f"    → {len(pairs)} verified pairs", file=sys.stderr)
            if i < len(docs) - 1:
                time.sleep(args.delay)

    print(f"done: {total} gold pairs → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
