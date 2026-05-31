#!/usr/bin/env python3
"""
01_extract.py — Parse JW EPUB files into clean plain text.

Reads all .epub files from data/raw_epubs/, extracts text content
from XHTML spine documents, strips HTML/CSS/XML markup, and writes
clean .txt files to data/extracted/.

Output: data/extracted/<pub_name>.txt (one file per EPUB)
"""

import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT
from tqdm import tqdm


# ── paths ──────────────────────────────────────────────────────────────
RAW_DIR   = Path("data/raw_epubs")
OUT_DIR   = Path("data/extracted")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────

def clean_html(html_text: str) -> str:
    """Strip all HTML tags and return clean text with sensible whitespace."""
    soup = BeautifulSoup(html_text, "lxml")
    # Remove script, style, nav, and hidden elements
    for tag in soup(["script", "style", "nav", "head", "meta", "link"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    # Remove completely empty lines that are just formatting artifacts
    lines = [l for l in lines if l]
    return "\n".join(lines)


def extract_epub(epub_path: Path) -> str:
    """Extract all text from an EPUB file, returning concatenated clean text."""
    book = epub.read_epub(str(epub_path))
    parts: list[str] = []

    for item in book.get_items_of_type(ITEM_DOCUMENT):
        try:
            content = item.get_content().decode("utf-8")
        except UnicodeDecodeError:
            continue  # skip binary content

        cleaned = clean_html(content)
        if cleaned.strip():
            # Add a document separator for provenance tracking
            file_name = item.get_name()
            parts.append(f"<!-- source: {file_name} -->\n{cleaned}")

    return "\n\n".join(parts)


# ── main ───────────────────────────────────────────────────────────────

def main():
    epub_files = sorted(RAW_DIR.glob("*.epub"))
    if not epub_files:
        print(f"No EPUB files found in {RAW_DIR.resolve()}")
        print("Copy your JW EPUBs there first, e.g.:")
        print("  cp ~/.hermes/jw-library/downloads/*.epub data/raw_epubs/")
        sys.exit(1)

    print(f"Found {len(epub_files)} EPUB(s)\n")

    for epub_path in tqdm(epub_files, desc="Extracting", unit="epub"):
        pub_name = epub_path.stem  # filename without .epub
        try:
            text = extract_epub(epub_path)
        except Exception as e:
            print(f"\n  ERROR: {pub_name}: {e}")
            continue

        out_path = OUT_DIR / f"{pub_name}.txt"
        out_path.write_text(text, encoding="utf-8")

        size_kb = len(text) / 1024
        tqdm.write(f"  {pub_name}.txt → {size_kb:.0f} KB ({len(text):,} chars)")

    print(f"\nDone. Extracted text in {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
