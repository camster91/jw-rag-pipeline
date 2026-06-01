"""
JW Study API — RAG-powered scripture search
Serves ChromaDB queries via FastAPI + embedded React frontend
"""
import os
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("JW_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
VECTOR_DB_DIR = os.environ.get("JW_VECTOR_DB", str(DATA_DIR / "vector_db"))
MODEL_NAME = os.environ.get("JW_EMBED_MODEL", "all-MiniLM-L6-v2")
DEFAULT_K = int(os.environ.get("JW_DEFAULT_K", "10"))

# ── Init once at startup ──────────────────────────────
print(f"Loading embedding model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

print(f"Loading ChromaDB from: {VECTOR_DB_DIR}...")
client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
collection = client.get_collection("jw_publications")
count = collection.count()
print(f"Collection ready: {count} vectors")

app = FastAPI(title="JW Study", version="1.0.0")

# Allow cross-origin requests from GH Pages and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://camster91.github.io",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/search")
def search(q: str = Query(..., min_length=1), k: int = Query(default=DEFAULT_K, ge=1, le=50)):
    """Semantic search across JW publications."""
    embedding = model.encode([q]).tolist()
    results = collection.query(query_embeddings=embedding, n_results=k)

    hits = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        hits.append({
            "id": results["ids"][0][i],
            "score": round(1 - results["distances"][0][i], 4),
            "publication": meta.get("publication", "Unknown"),
            "chunk": meta.get("chunk_index", 0),
            "text": results["documents"][0][i][:800],
        })

    return {"query": q, "count": len(hits), "results": hits}


@app.get("/api/stats")
def stats():
    """Collection statistics."""
    return {
        "name": collection.name,
        "vectors": collection.count(),
        "model": MODEL_NAME,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve embedded single-page app."""
    return HTMLResponse(FRONTEND_HTML)


# ── Embedded React SPA ───────────────────────────────
FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>JW Study — Scripture Search</title>
<style>
  :root {
    --bg: #0f172a; --card: #1e293b; --border: #334155;
    --text: #f1f5f9; --muted: #94a3b8; --primary: #4A6FA4;
    --accent: #71BC37; --highlight: #fbbf24;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
  .container { max-width: 800px; margin: 0 auto; padding: 24px 16px; }
  header { text-align: center; padding: 48px 0 32px; }
  header h1 { font-size: 2rem; font-weight: 700; letter-spacing: -0.5px; }
  header p { color: var(--muted); margin-top: 8px; }
  .search-box { position: relative; margin-bottom: 32px; }
  .search-box input {
    width: 100%; padding: 16px 20px; font-size: 1.1rem;
    border: 2px solid var(--border); border-radius: 16px;
    background: var(--card); color: var(--text); outline: none;
    transition: border-color 0.2s;
  }
  .search-box input:focus { border-color: var(--primary); }
  .search-box .hint { color: var(--muted); font-size: 0.8rem; margin-top: 8px; margin-left: 4px; }
  .stats-bar { display: flex; gap: 16px; justify-content: center; margin-bottom: 24px; flex-wrap: wrap; }
  .stat { background: var(--card); border-radius: 12px; padding: 12px 20px; text-align: center; }
  .stat .num { font-size: 1.5rem; font-weight: 700; color: var(--primary); }
  .stat .lbl { font-size: 0.75rem; color: var(--muted); }
  .result { background: var(--card); border-radius: 16px; padding: 20px; margin-bottom: 12px; border: 1px solid var(--border); }
  .result .meta { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
  .result .src { font-size: 0.75rem; font-weight: 600; color: var(--primary); background: rgba(74,111,164,0.15); padding: 3px 10px; border-radius: 6px; }
  .result .ch { font-size: 0.75rem; color: var(--muted); }
  .result .score { font-size: 0.7rem; color: var(--accent); margin-left: auto; }
  .result .txt { font-size: 0.95rem; line-height: 1.6; color: var(--text); }
  .result .txt mark { background: rgba(251,191,36,0.2); color: var(--highlight); border-radius: 2px; padding: 0 2px; }
  .empty { text-align: center; padding: 60px 20px; color: var(--muted); }
  .empty .icon { font-size: 3rem; margin-bottom: 16px; }
  .loading { text-align: center; padding: 40px; color: var(--muted); }
  .spinner { width: 32px; height: 32px; border: 3px solid var(--border); border-top-color: var(--primary); border-radius: 50%; animation: spin 0.8s linear infinite; margin: 0 auto 12px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .error { background: #7f1d1d20; border: 1px solid #7f1d1d; border-radius: 12px; padding: 16px; color: #fca5a5; margin-bottom: 12px; }
  footer { text-align: center; padding: 40px 0; color: var(--muted); font-size: 0.8rem; }
  @media (max-width: 600px) { header h1 { font-size: 1.5rem; } .search-box input { font-size: 1rem; padding: 14px 16px; } }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>📖 JW Study</h1>
    <p>Search 62,000+ passages across 7 JW publications</p>
  </header>

  <div class="search-box">
    <input type="text" id="searchInput" placeholder="What are you studying? e.g. faith without works, God's name, resurrection hope..."
      autofocus autocomplete="off">
    <div class="hint">Semantic search — ask questions or search keywords</div>
  </div>

  <div class="stats-bar" id="statsBar"></div>
  <div id="results"></div>

  <footer>
    Built on JW RAG Pipeline · all-MiniLM-L6-v2 · ChromaDB
  </footer>
</div>

<script>
const input = document.getElementById('searchInput');
const results = document.getElementById('results');
const statsBar = document.getElementById('statsBar');
let debounceTimer;

// Load stats on page load
fetch('/api/stats').then(r => r.json()).then(s => {
  statsBar.innerHTML = `
    <div class="stat"><div class="num">${s.vectors.toLocaleString()}</div><div class="lbl">Vectors</div></div>
    <div class="stat"><div class="num">7</div><div class="lbl">Publications</div></div>
    <div class="stat"><div class="num">${s.model.split('-').slice(0,2).join('-')}</div><div class="lbl">Model</div></div>
  `;
});

function highlight(text, query) {
  if (!query) return text;
  const words = query.split(/\\s+/).filter(w => w.length > 2);
  let result = text;
  words.forEach(w => {
    const re = new RegExp(`(${w.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&')})`, 'gi');
    result = result.replace(re, '<mark>$1</mark>');
  });
  return result;
}

function doSearch() {
  const q = input.value.trim();
  if (!q) { results.innerHTML = ''; return; }

  results.innerHTML = '<div class="loading"><div class="spinner"></div>Searching...</div>';

  fetch(`/api/search?q=${encodeURIComponent(q)}&k=10`)
    .then(r => r.json())
    .then(data => {
      if (!data.results.length) {
        results.innerHTML = '<div class="empty"><div class="icon">🔍</div><p>No results found. Try different wording.</p></div>';
        return;
      }
      results.innerHTML = data.results.map((r, i) => `
        <div class="result">
          <div class="meta">
            <span class="src">${r.publication}</span>
            <span class="score">${Math.round(r.score * 100)}% match</span>
          </div>
          <div class="txt">${highlight(r.text, q)}</div>
        </div>
      `).join('');
    })
    .catch(e => {
      results.innerHTML = `<div class="error">Search failed: ${e.message}</div>`;
    });
}

input.addEventListener('input', () => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(doSearch, 300);
});

// Initial search hint
results.innerHTML = '<div class="empty"><div class="icon">📖</div><p>Type a question or topic above to search scripture.</p></div>';
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
