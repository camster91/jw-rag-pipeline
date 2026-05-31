#!/usr/bin/env python3
"""
02_chunk.py — Split extracted text into overlapping semantic chunks.

Reads clean .txt files from data/extracted/, splits each into
overlapping chunks using LangChain's RecursiveCharacterTextSplitter,
and writes JSONL files to data/chunks/.

Each chunk includes metadata: source publication, chunk index,
and character offset for provenance tracking.

Output: data/chunks/<pub_name>.jsonl
"""

import json
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


# ── paths ──────────────────────────────────────────────────────────────
TXT_DIR   = Path("data/extracted")
CHUNK_DIR = Path("data/chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# ── splitter config ────────────────────────────────────────────────────
# 500 chars per chunk with 50-char overlap — tuned for scripture paragraphs
# JW text is dense; 500 chars ≈ 3-5 verses or 2-3 paragraphs
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],  # paragraph → sentence → word
    length_function=len,
    is_separator_regex=False,
)


def chunk_text(text: str, pub_name: str) -> list[dict]:
    """Split text into chunks with metadata. Returns list of chunk dicts."""
    chunks = splitter.split_text(text)
    results = []
    offset = 0

    for i, chunk in enumerate(chunks):
        results.append({
            "id": f"{pub_name}_{i:05d}",
            "publication": pub_name,
            "chunk_index": i,
            "char_offset": offset,
            "char_length": len(chunk),
            "text": chunk,
        })
        offset += len(chunk) - CHUNK_OVERLAP  # account for overlap

    return results


# ── main ───────────────────────────────────────────────────────────────

def main():
    txt_files = sorted(TXT_DIR.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {TXT_DIR.resolve()}")
        print("Run 01_extract.py first.")
        sys.exit(1)

    total_chunks = 0

    for txt_path in tqdm(txt_files, desc="Chunking", unit="file"):
        pub_name = txt_path.stem
        text = txt_path.read_text(encoding="utf-8")

        chunks = chunk_text(text, pub_name)
        total_chunks += len(chunks)

        out_path = CHUNK_DIR / f"{pub_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        tqdm.write(f"  {pub_name}: {len(chunks):,} chunks")

    print(f"\nDone. {total_chunks:,} total chunks written to {CHUNK_DIR.resolve()}/")


if __name__ == "__main__":
    main()
