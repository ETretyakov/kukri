#!/usr/bin/env python3
"""Chonkie SemanticChunker baseline."""
import argparse, json, sys
from chonkie import SemanticChunker

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",     required=True)
    ap.add_argument("--output",    default="-")
    ap.add_argument("--max-tokens", type=int,   default=512)
    ap.add_argument("--threshold",  type=float, default=0.7)
    args = ap.parse_args()

    text = open(args.input).read()

    chunker = SemanticChunker(
        embedding_model="minishlab/potion-base-8M",
        chunk_size=args.max_tokens,
        threshold=args.threshold,
    )
    raw = chunker.chunk(text)

    chunks = [
        {
            "id":          str(i),
            "text":        c.text,
            "token_count": c.token_count,
            "block_type":  "text",
            "section":     "",
            "section_path": [],
            "source_lines": [0, 0],
            "source":      args.input,
        }
        for i, c in enumerate(raw)
    ]

    result = {"strategy": "chonkie_semantic", "chunks": chunks}
    out = open(args.output, "w") if args.output != "-" else sys.stdout
    json.dump(result, out, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
