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
import re


# ── OCR quality filter ─────────────────────────────────────────────────
def is_garbage_line(line: str) -> bool:
    """Return True if this line is likely OCR artifact, not real content."""
    s = line.strip()
    if not s:
        return False
    # Isolated 1-2 character non-word symbols
    if len(s) <= 2 and s.lower() not in {
        'i', 'a', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he',
        'if', 'in', 'is', 'it', 'me', 'my', 'no', 'of', 'on', 'or',
        'so', 'to', 'up', 'us', 'we', 'am', 'oh', 'ah', 'hi', 'ok',
    }:
        return True
    # Lines that are purely symbolic/page markers
    if re.match(r'^[\W\d\s°\-─═]+$', s) and len(s) < 10:
        return True
    # Lines where < 50% of characters are alphabetic or space
    if len(s) > 5:
        alpha_ratio = sum(1 for c in s if c.isalpha() or c == ' ') / len(s)
        if alpha_ratio < 0.5:
            return True
    return False


def clean_ocr_text(text: str) -> str:
    """Remove OCR garbage lines and normalize whitespace."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        if not is_garbage_line(line):
            cleaned.append(line)
    result = '\n'.join(cleaned)
    # Collapse 3+ blank lines
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


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

        # Apply OCR quality filter to clean garbage lines
        text_before = len(text)
        text = clean_ocr_text(text)
        removed = text_before - len(text)
        if removed > text_before * 0.05:  # More than 5% removed
            tqdm.write(f"  {pub_name}: stripped {removed/1024:.0f}KB OCR garbage ({(removed/text_before)*100:.0f}%)")

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
