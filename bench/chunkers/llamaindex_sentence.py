#!/usr/bin/env python3
"""LlamaIndex SentenceSplitter baseline."""
import argparse, json, sys
from llama_index.core.node_parser import SentenceSplitter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      required=True)
    ap.add_argument("--output",     default="-")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--overlap",    type=int, default=50)
    args = ap.parse_args()

    text = open(args.input).read()

    splitter = SentenceSplitter(
        chunk_size=args.max_tokens,
        chunk_overlap=args.overlap,
    )
    nodes = splitter.split_text(text)

    chunks = [
        {
            "id":           str(i),
            "text":         node,
            "token_count":  len(node.split()),
            "block_type":   "text",
            "section":      "",
            "section_path": [],
            "source_lines": [0, 0],
            "source":       args.input,
        }
        for i, node in enumerate(nodes)
    ]

    result = {"strategy": "llamaindex_sentence", "chunks": chunks}
    out = open(args.output, "w") if args.output != "-" else sys.stdout
    json.dump(result, out, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
