#!/usr/bin/env python3
"""
Build corpus.jsonl from parsed IAEA markdown documents.

Each line: {"id": "<doc_dir_name>", "text": "<document.md contents>"}

Usage:
  python bench/make_corpus.py \
    --input ../../parsing/output \
    --output corpus.jsonl \
    --max-docs 200
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../../parsing/output",
                    help="directory containing per-doc subdirs with document.md")
    ap.add_argument("--output", default="corpus.jsonl")
    ap.add_argument("--max-docs", type=int, default=None,
                    help="limit number of docs (default: all)")
    ap.add_argument("--min-chars", type=int, default=500,
                    help="skip docs shorter than this (default: 500)")
    args = ap.parse_args()

    root = Path(args.input)
    if not root.exists():
        print(f"error: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    doc_files = sorted(root.glob("*/document.md"))
    if args.max_docs:
        doc_files = doc_files[: args.max_docs]

    written = 0
    skipped = 0
    with open(args.output, "w") as out:
        for path in doc_files:
            text = path.read_text(errors="replace").strip()
            if len(text) < args.min_chars:
                skipped += 1
                continue
            doc_id = path.parent.name
            out.write(json.dumps({"id": doc_id, "text": text}, ensure_ascii=False) + "\n")
            written += 1

    print(f"written: {written} docs → {args.output}  (skipped {skipped} too-short)", file=sys.stderr)


if __name__ == "__main__":
    main()
