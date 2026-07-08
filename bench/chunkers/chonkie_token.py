#!/usr/bin/env python3
"""Chonkie TokenChunker baseline (fixed-size)."""
import argparse, json, sys
from chonkie import TokenChunker

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      required=True)
    ap.add_argument("--output",     default="-")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--overlap",    type=int, default=50)
    args = ap.parse_args()

    text = open(args.input).read()

    chunker = TokenChunker(
        chunk_size=args.max_tokens,
        chunk_overlap=args.overlap,
    )
    raw = chunker.chunk(text)

    chunks = [
        {
            "id":           str(i),
            "text":         c.text,
            "token_count":  c.token_count,
            "block_type":   "text",
            "section":      "",
            "section_path": [],
            "source_lines": [0, 0],
            "source":       args.input,
        }
        for i, c in enumerate(raw)
    ]

    result = {"strategy": "chonkie_token", "chunks": chunks}
    out = open(args.output, "w") if args.output != "-" else sys.stdout
    json.dump(result, out, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
