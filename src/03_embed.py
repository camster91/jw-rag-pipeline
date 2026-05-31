#!/usr/bin/env python3
"""
03_embed.py — Generate embeddings and store in ChromaDB.

Reads JSONL chunks from data/chunks/, generates 384-dimensional
embeddings using sentence-transformers (all-MiniLM-L6-v2),
and stores them in a persistent ChromaDB vector database.

First run downloads the model (~90MB) and caches it locally.
Subsequent runs are fast — only changed chunks are re-embedded.

Output: data/vector_db/ (ChromaDB persistent directory)
"""

import json
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ── paths & config ─────────────────────────────────────────────────────
CHUNK_DIR  = Path("data/chunks")
VECTOR_DIR = Path("data/vector_db")
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "jw_publications"
BATCH_SIZE = 64  # embed in batches for speed


# ── init ───────────────────────────────────────────────────────────────

def load_model() -> SentenceTransformer:
    """Load the embedding model (downloads on first run)."""
    print(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Model loaded. Dimension: {model.get_sentence_embedding_dimension()}")
    return model


def get_or_create_collection():
    """Get or create the ChromaDB collection."""
    client = chromadb.PersistentClient(
        path=str(VECTOR_DIR.resolve()),
        settings=Settings(anonymized_telemetry=False),
    )

    # Delete existing collection to rebuild fresh
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Dropped existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "JW publications — RAG pipeline embeddings"},
    )
    return collection


# ── main ───────────────────────────────────────────────────────────────

def main():
    jsonl_files = sorted(CHUNK_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl chunks found in {CHUNK_DIR.resolve()}")
        print("Run 02_chunk.py first.")
        sys.exit(1)

    # Load all chunks into memory
    all_chunks: list[dict] = []
    for jf in jsonl_files:
        with open(jf, encoding="utf-8") as f:
            for line in f:
                all_chunks.append(json.loads(line))

    print(f"Loaded {len(all_chunks):,} chunks from {len(jsonl_files)} publication(s)")

    # Load model
    model = load_model()

    # Extract texts
    texts = [c["text"] for c in all_chunks]
    ids = [c["id"] for c in all_chunks]
    metadatas = [
        {
            "publication": c["publication"],
            "chunk_index": c["chunk_index"],
            "char_offset": c["char_offset"],
            "char_length": c["char_length"],
        }
        for c in all_chunks
    ]

    # Generate embeddings in batches
    print(f"\nGenerating embeddings ({len(texts):,} chunks, batch size {BATCH_SIZE})...")
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding", unit="batch"):
        batch = texts[i : i + BATCH_SIZE]
        batch_embeddings = model.encode(
            batch,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embeddings.extend(batch_embeddings.tolist())

    # Store in ChromaDB
    print(f"\nStoring in ChromaDB at {VECTOR_DIR.resolve()}...")
    collection = get_or_create_collection()

    # Add in sub-batches to avoid memory spikes
    for i in tqdm(range(0, len(ids), BATCH_SIZE), desc="Indexing", unit="batch"):
        collection.add(
            ids=ids[i : i + BATCH_SIZE],
            embeddings=embeddings[i : i + BATCH_SIZE],
            documents=texts[i : i + BATCH_SIZE],
            metadatas=metadatas[i : i + BATCH_SIZE],
        )

    count = collection.count()
    print(f"\nDone. {count:,} vectors indexed in '{COLLECTION_NAME}'")
    print(f"Vector DB: {VECTOR_DIR.resolve()}/")


if __name__ == "__main__":
    main()
