#!/usr/bin/env python3
"""LangChain MarkdownHeaderTextSplitter baseline."""
import argparse, json, sys
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      required=True)
    ap.add_argument("--output",     default="-")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    text = open(args.input).read()

    # Step 1: split by markdown headers
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#",   "h1"),
            ("##",  "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )
    header_docs = header_splitter.split_text(text)

    # Step 2: further split large sections by character count
    # ~4 chars per token approximation
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.max_tokens * 4,
        chunk_overlap=100,
    )
    docs = char_splitter.split_documents(header_docs)

    chunks = []
    for i, doc in enumerate(docs):
        meta = doc.metadata
        path = [v for v in [meta.get("h1"), meta.get("h2"), meta.get("h3")] if v]
        chunks.append({
            "id":           str(i),
            "text":         doc.page_content,
            "token_count":  len(doc.page_content.split()),
            "block_type":   "text",
            "section":      path[-1] if path else "",
            "section_path": path,
            "source_lines": [0, 0],
            "source":       args.input,
        })

    result = {"strategy": "langchain_md", "chunks": chunks}
    out = open(args.output, "w") if args.output != "-" else sys.stdout
    json.dump(result, out, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
