# JW RAG Pipeline

Local Retrieval-Augmented Generation pipeline for JW.org publications. Extracts EPUB files, chunks them semantically, generates embeddings, and stores everything in a local ChromaDB vector database for fast semantic search.

## Tech Stack

| Layer | Technology |
|-------|------------|
| EPUB parsing | EbookLib + BeautifulSoup4 + lxml |
| Text chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB (persistent, local) |
| CLI | Rich + tqdm |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy your JW EPUBs into data/raw_epubs/
cp ~/.hermes/jw-library/downloads/*.epub data/raw_epubs/

# 3. Run the pipeline in order
python src/01_extract.py    # EPUB → clean text files
python src/02_chunk.py      # Text → semantic chunks
python src/03_embed.py      # Chunks → embeddings → ChromaDB
python src/04_query.py      # Interactive search
```

## Pipeline Steps

1. **Extract** (`01_extract.py`) — Parses EPUB files from `data/raw_epubs/`, strips HTML/CSS/XML, outputs clean `.txt` files to `data/extracted/`
2. **Chunk** (`02_chunk.py`) — Splits extracted text into overlapping semantic chunks using LangChain's recursive splitter (500 chars, 50 overlap)
3. **Embed** (`03_embed.py`) — Generates 384-dim embeddings via sentence-transformers, stores in ChromaDB at `data/vector_db/`
4. **Query** (`04_query.py`) — CLI search interface with ranked results, source file references, and context snippets

## Data Flow

```
data/raw_epubs/*.epub
    ↓ 01_extract.py
data/extracted/*.txt
    ↓ 02_chunk.py
data/chunks/*.jsonl
    ↓ 03_embed.py
data/vector_db/ (ChromaDB)
    ↓ 04_query.py
Search results
```

## Notes

- All data files are gitignored — only source code is tracked
- First run downloads the embedding model (~90MB) and caches it locally
- ChromaDB persists to disk — re-indexing is only needed when EPUBs change
- Designed for local use only — no network calls after model download
