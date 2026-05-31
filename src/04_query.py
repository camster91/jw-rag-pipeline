#!/usr/bin/env python3
"""
04_query.py — Semantic search over JW publications via ChromaDB.

Interactive CLI: type a query, get ranked results with source
publication, chunk context, and relevance scores.

Usage:
    python src/04_query.py                    # interactive mode
    python src/04_query.py "faith and works"  # single query
    python src/04_query.py --top 10 "prayer"  # custom result count
"""

import argparse
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from sentence_transformers import SentenceTransformer


# ── config ─────────────────────────────────────────────────────────────
VECTOR_DIR        = Path("data/vector_db")
COLLECTION_NAME   = "jw_publications"
MODEL_NAME        = "all-MiniLM-L6-v2"
DEFAULT_TOP_K     = 5
SNIPPET_CHARS     = 300  # chars to show around the match

console = Console()

# ── init ───────────────────────────────────────────────────────────────

def load_search_engine():
    """Load model and connect to ChromaDB."""
    if not VECTOR_DIR.exists():
        console.print(
            f"\n[red]Vector DB not found at {VECTOR_DIR.resolve()}[/red]\n"
            "Run 03_embed.py first to build the index."
        )
        sys.exit(1)

    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(
        path=str(VECTOR_DIR.resolve()),
        settings=Settings(anonymized_telemetry=False),
    )

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        console.print(
            f"\n[red]Collection '{COLLECTION_NAME}' not found.[/red]\n"
            "Run 03_embed.py first to build the index."
        )
        sys.exit(1)

    return model, collection


def search(query: str, model, collection, top_k: int = DEFAULT_TOP_K):
    """Run semantic search and return formatted results."""
    # Generate query embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return results


def format_results(query: str, results: dict):
    """Pretty-print search results using Rich."""
    distances = results["distances"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not documents:
        console.print("\n[yellow]No results found.[/yellow]")
        return

    # Build results table
    table = Table(
        title=f"Results for: [bold cyan]\"{query}\"[/bold cyan]",
        title_style="bold",
        border_style="dim blue",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Publication", style="bold green", width=15)
    table.add_column("Score", style="yellow", width=7)
    table.add_column("Snippet", width=60)

    for i in range(len(documents)):
        # Cosine distance → similarity score (0-1, higher = better)
        score = max(0, 1 - distances[i]) if distances[i] <= 1 else 0
        pub = metadatas[i].get("publication", "unknown")
        chunk_idx = metadatas[i].get("chunk_index", "?")

        # Truncate snippet for display
        snippet = documents[i].replace("\n", " ").strip()
        if len(snippet) > SNIPPET_CHARS:
            snippet = snippet[:SNIPPET_CHARS] + "..."

        table.add_row(
            str(i + 1),
            f"{pub}\n(chunk {chunk_idx})",
            f"{score:.3f}",
            snippet,
        )

    console.print()
    console.print(table)

    # Show full text for top result
    console.print()
    top_doc = documents[0].strip()
    console.print(Panel(
        top_doc,
        title=f"[bold]Top Result[/bold] — {metadatas[0].get('publication', 'unknown')}, chunk {metadatas[0].get('chunk_index', '?')}",
        title_align="left",
        border_style="green",
    ))


def interactive_loop(model, collection):
    """Run interactive query loop."""
    console.print()
    console.print(Panel(
        "[bold]JW RAG Pipeline — Semantic Search[/bold]\n"
        f"Collection: {collection.count():,} chunks indexed\n"
        "Type a query or 'quit' to exit.",
        border_style="blue",
    ))

    while True:
        try:
            query = console.input("\n[bold cyan]Search >[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            console.print("Goodbye.")
            break

        results = search(query, model, collection)
        format_results(query, results)


# ── main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Semantic search over JW publications using RAG"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Search query (omit for interactive mode)",
    )
    parser.add_argument(
        "--top", "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results (default: {DEFAULT_TOP_K})",
    )
    args = parser.parse_args()

    model, collection = load_search_engine()

    if args.query:
        # Single-shot mode
        results = search(args.query, model, collection, top_k=args.top)
        format_results(args.query, results)
    else:
        # Interactive mode
        interactive_loop(model, collection)


if __name__ == "__main__":
    main()
