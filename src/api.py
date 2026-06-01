"""
JW Study API — RAG-powered scripture search
Serves ChromaDB queries via FastAPI + embedded React frontend

Endpoints:
  GET  /api/search           Semantic + keyword search
  GET  /api/scriptures       Extract scripture refs from a passage
  GET  /api/related          Find related passages
  GET  /api/passage/{id}     Get a specific chunk by id
  GET  /api/stats            Corpus statistics
  GET  /                     Embedded SPA
"""
import os
import re
from pathlib import Path
from collections import Counter

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
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

app = FastAPI(title="JW Study", version="2.0.0")

# Allow cross-origin requests from GH Pages, Coolify apps, and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.ashbi.ca",
        "https://camster91.github.io",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=500)


# ── Scripture extraction ──────────────────────────────────────

# Multi-word short pattern: "1 Cor 1:13", "2 Sam 5:3", "1 Thes 4:1"
# Number + 2-3 letter abbreviation
MULTIWORD_PATTERN = re.compile(
    r"\b((?:[1-3])?\s?(?:Cor|Sam|Ki|Kgs|Ch|Chr|Co|Corinthians?|Samuel|Kings?|Chronicles?|Thes(?:s)?|Thess(?:alonians)?|Tim(?:othy)?|Pe|Pet|Peter|Jn|Jo|Joh|John|Joes?))\s+(\d+)(?::(\d+(?:[a-z]?)?(?:\s*[-–]\s*\d+(?:[a-z]?)?)*))?",
    re.IGNORECASE,
)

# Singular forms — handle "Psalm", "Song", "1 Cor" (without s)
SINGULAR_PATTERN = re.compile(
    r"\b((?:[1-3])?\s?(?:Psalm|Song(?:s)?|SongOfSolomon|Gen(?:esis)?|Exo(?:dus)?|Lev(?:iticus)?|Num(?:bers)?|Deu(?:t(?:eronomy)?)?|Jos(?:hua)?|Jdg(?:es)?|Rut(?:h)?|Sam(?:uel)?|Ki(?:ngs?)?|Ch(?:r(?:onicles?)?)?|Ezr(?:a)?|Neh(?:emiah)?|Est(?:her)?|Job|Psa(?:lm)?|Pro(?:verbs?)?|Ecc(?:l(?:esiastes?)?)?|Sng|Isa(?:iah)?|Jer(?:emiah)?|Lam(?:entation(?:s)?)?|Eze(?:k(?:iel)?)?|Dan(?:iel)?|Hos(?:ea)?|Joel?|Amos?|Oba(?:d(?:iah)?)?|Jon(?:ah)?|Mic(?:ah)?|Nah(?:um)?|Hab(?:akkuk)?|Zep(?:h(?:aniah)?)?|Hag(?:gai)?|Zec(?:h(?:ariah)?)?|Mal(?:achi)?|Matt?|Mar(?:k)?|Lk|Luk(?:e)?|Jn|Joh(?:n)?|Act(?:s)?|Rom(?:ans?)?|Gal(?:atians?)?|Eph(?:esians?)?|Phi(?:l(?:ipians?)?)?|Col(?:ossians?)?|Th(?:es(?:s(?:alonians?)?)?)?|Tim(?:othy)?|Tit(?:us)?|Phm|Phil(?:m|emon)?|Heb(?:rews?)?|Jas|Jam(?:es)?|Pet(?:er)?|Jud(?:e)?|Rev(?:elation)?))\s+(\d+)(?::(\d+(?:[a-z]?)?(?:\s*[-–]\s*\d+(?:[a-z]?)?)*))?",
    re.IGNORECASE,
)

# Pattern: Book + chapter[:verse[-verse]]
# Matches "John 3:16", "1 Cor 1:13-17", "Romans 8:28, 30", "Psalm 23"
# Book: 1-3 letter abbreviation, capitalized, NOT followed by digits and colon
SCRIPTURE_PATTERN = re.compile(
    r"\b((?:[1-3])?[A-Z][a-z]{1,3})\s+(\d+)(?::(\d+(?:[a-z]?)?(?:\s*[-–]\s*\d+(?:[a-z]?)?)*))?",
)

# Whitelist: only these abbreviations count as Bible books (avoid false positives)
VALID_BOOKS = {
    "Gen", "Exo", "Exod", "Lev", "Num", "Deu", "Deut", "Jos", "Josh", "Jdg", "Judg", "Rut", "Ruth",
    "1Sa", "1Sam", "2Sa", "2Sam", "1Ki", "1Kgs", "2Ki", "2Kgs",
    "1Ch", "1Chr", "2Ch", "2Chr", "Ezr", "Ezra", "Neh", "Est", "Esth", "Job", "Psa", "Ps", "Psm", "Pro", "Prov", "Ecc", "Eccl", "Sng", "Song", "SS", "SOS",
    "Isa", "Is", "Jer", "Lam", "Eze", "Ezek", "Dan", "Hos", "Joe", "Joel", "Amo", "Amos", "Oba", "Obad", "Jon", "Jnh", "Mic", "Nah", "Nam", "Hab", "Zep", "Zeph",
    "Hag", "Zec", "Zech", "Mal", "Matt", "Mat", "Mr", "Mar", "Mark", "Lk", "Luk", "Luke", "Jn", "Joh", "John", "Act", "Acts",
    "Rom", "1Co", "1Cor", "2Co", "2Cor", "Gal", "Eph", "Phi", "Phil", "Col", "1Th", "1Thes", "2Th", "2Thes",
    "1Ti", "1Tim", "2Ti", "2Tim", "Tit", "Phm", "Philm", "Heb", "Jas", "Jam", "1Pe", "1Pet", "2Pe", "2Pet",
    "1Jo", "1Jn", "1John", "2Jo", "2Jn", "2John", "3Jo", "3Jn", "3John", "Jud", "Jude", "Rev",
}

BOOK_NORMALIZE = {
    "Gen": "Genesis", "Exo": "Exodus", "Exod": "Exodus", "Lev": "Leviticus", "Num": "Numbers",
    "Deu": "Deuteronomy", "Deut": "Deuteronomy", "Jos": "Joshua", "Josh": "Joshua", "Jdg": "Judges", "Judg": "Judges", "Rut": "Ruth",
    "1Sa": "1 Samuel", "1Sam": "1 Samuel", "2Sa": "2 Samuel", "2Sam": "2 Samuel",
    "1Ki": "1 Kings", "1Kgs": "1 Kings", "2Ki": "2 Kings", "2Kgs": "2 Kings",
    "1Ch": "1 Chronicles", "1Chr": "1 Chronicles", "2Ch": "2 Chronicles", "2Chr": "2 Chronicles",
    "Ezr": "Ezra", "Ezra": "Ezra", "Neh": "Nehemiah", "Est": "Esther", "Esth": "Esther",
    "Job": "Job", "Psa": "Psalms", "Ps": "Psalms", "Psm": "Psalms",
    "Pro": "Proverbs", "Prov": "Proverbs", "Ecc": "Ecclesiastes", "Eccl": "Ecclesiastes",
    "Sng": "Song of Solomon", "Song": "Song of Solomon", "SS": "Song of Solomon", "SOS": "Song of Solomon",
    "Isa": "Isaiah", "Is": "Isaiah", "Jer": "Jeremiah", "Lam": "Lamentations",
    "Eze": "Ezekiel", "Ezek": "Ezekiel", "Dan": "Daniel", "Hos": "Hosea",
    "Joe": "Joel", "Joel": "Joel", "Amo": "Amos", "Amos": "Amos",
    "Oba": "Obadiah", "Obad": "Obadiah", "Jon": "Jonah", "Jnh": "Jonah",
    "Mic": "Micah", "Nah": "Nahum", "Nam": "Nahum", "Hab": "Habakkuk",
    "Zep": "Zephaniah", "Zeph": "Zephaniah", "Hag": "Haggai", "Zec": "Zechariah", "Zech": "Zechariah", "Mal": "Malachi",
    "Mat": "Matthew", "Matt": "Matthew", "Mr": "Mark", "Mar": "Mark", "Mark": "Mark",
    "Lk": "Luke", "Luk": "Luke", "Luke": "Luke", "Jn": "John", "Joh": "John", "John": "John",
    "Act": "Acts", "Acts": "Acts", "Rom": "Romans",
    "1Co": "1 Corinthians", "1Cor": "1 Corinthians", "2Co": "2 Corinthians", "2Cor": "2 Corinthians",
    "Gal": "Galatians", "Eph": "Ephesians", "Phi": "Philippians", "Phil": "Philippians",
    "Col": "Colossians", "1Th": "1 Thessalonians", "1Thes": "1 Thessalonians", "2Th": "2 Thessalonians", "2Thes": "2 Thessalonians",
    "1Ti": "1 Timothy", "1Tim": "1 Timothy", "2Ti": "2 Timothy", "2Tim": "2 Timothy",
    "Tit": "Titus", "Phm": "Philemon", "Philm": "Philemon", "Heb": "Hebrews",
    "Jas": "James", "Jam": "James", "1Pe": "1 Peter", "1Pet": "1 Peter",
    "2Pe": "2 Peter", "2Pet": "2 Peter",
    "1Jo": "1 John", "1Jn": "1 John", "1John": "1 John", "2Jo": "2 John", "2Jn": "2 John", "2John": "2 John",
    "3Jo": "3 John", "3Jn": "3 John", "3John": "3 John", "Jud": "Jude", "Jude": "Jude", "Rev": "Revelation",
}

# Also handle spelled-out book names
BOOK_BY_FULL = {v.lower(): v for v in set(BOOK_NORMALIZE.values())}
BOOK_BY_FULL.update({
    "psalm": "Psalms", "psalms": "Psalms",
    "song of solomon": "Song of Solomon", "song of songs": "Song of Solomon", "song": "Song of Solomon",
    "1 corinthian": "1 Corinthians", "2 corinthian": "2 Corinthians",
    "1 thessalonian": "1 Thessalonians", "2 thessalonian": "2 Thessalonians",
    "1 timothy": "1 Timothy", "2 timothy": "2 Timothy",
    "1 peter": "1 Peter", "2 peter": "2 Peter",
    "1 john": "1 John", "2 john": "2 John", "3 john": "3 John",
    "1 samuel": "1 Samuel", "2 samuel": "2 Samuel",
    "1 kings": "1 Kings", "2 kings": "2 Kings",
    "1 chronicles": "1 Chronicles", "2 chronicles": "2 Chronicles",
    "1 corinthians": "1 Corinthians", "2 corinthians": "2 Corinthians",
    "1 thessalonians": "1 Thessalonians", "2 thessalonians": "2 Thessalonians",
    "1 timothys": "1 Timothy", "2 timothys": "2 Timothy",
    "1 peters": "1 Peter", "2 peters": "2 Peter",
    "1 johns": "1 John", "2 johns": "2 John", "3 johns": "3 John",
    "1 samuels": "1 Samuel", "2 samuels": "2 Samuel",
    "1 kingss": "1 Kings", "2 kingss": "2 Kings",
    "1 chronicless": "1 Chronicles", "2 chronicless": "2 Chronicles",
})

# Map singular pattern capture to canonical book name
SINGULAR_BOOK_MAP = {
    "psalm": "Psalms", "psa": "Psalms",
    "song": "Song of Solomon", "songs": "Song of Solomon", "songofsolomon": "Song of Solomon",
    "gen": "Genesis", "genesis": "Genesis",
    "exo": "Exodus", "exodus": "Exodus",
    "lev": "Leviticus", "leviticus": "Leviticus",
    "num": "Numbers", "numbers": "Numbers",
    "deu": "Deuteronomy", "deut": "Deuteronomy", "deuteronomy": "Deuteronomy",
    "jos": "Joshua", "joshua": "Joshua",
    "jdg": "Judges", "judges": "Judges",
    "rut": "Ruth", "ruth": "Ruth",
    "sam": "1 Samuel", "sa": "1 Samuel", "samuel": "1 Samuel",
    "ki": "1 Kings", "kings": "1 Kings", "king": "1 Kings",
    "ch": "1 Chronicles", "chr": "1 Chronicles", "chronicles": "1 Chronicles",
    "ezr": "Ezra", "ezra": "Ezra",
    "neh": "Nehemiah", "nehemiah": "Nehemiah",
    "est": "Esther", "esther": "Esther",
    "job": "Job",
    "pro": "Proverbs", "prov": "Proverbs", "proverbs": "Proverbs", "proverb": "Proverbs",
    "ecc": "Ecclesiastes", "eccl": "Ecclesiastes", "ecclesiastes": "Ecclesiastes",
    "sng": "Song of Solomon",
    "isa": "Isaiah", "isaiah": "Isaiah",
    "jer": "Jeremiah", "jeremiah": "Jeremiah",
    "lam": "Lamentations", "lamentation": "Lamentations", "lamentations": "Lamentations",
    "eze": "Ezekiel", "ezek": "Ezekiel", "ezekiel": "Ezekiel",
    "dan": "Daniel", "daniel": "Daniel",
    "hos": "Hosea", "hosea": "Hosea",
    "joel": "Joel", "joe": "Joel",
    "amos": "Amos", "amo": "Amos",
    "oba": "Obadiah", "obad": "Obadiah", "obadiah": "Obadiah",
    "jon": "Jonah", "jonah": "Jonah",
    "mic": "Micah", "micah": "Micah",
    "nah": "Nahum", "nahum": "Nahum",
    "hab": "Habakkuk", "habakkuk": "Habakkuk",
    "zep": "Zephaniah", "zeph": "Zephaniah", "zephaniah": "Zephaniah",
    "hag": "Haggai", "haggai": "Haggai",
    "zec": "Zechariah", "zech": "Zechariah", "zechariah": "Zechariah",
    "mal": "Malachi", "malachi": "Malachi",
    "matt": "Matthew", "mat": "Matthew",
    "mar": "Mark", "mark": "Mark",
    "lk": "Luke", "luk": "Luke", "luke": "Luke",
    "jn": "John", "joh": "John", "john": "John",
    "act": "Acts", "acts": "Acts",
    "rom": "Romans", "romans": "Romans", "roman": "Romans",
    "gal": "Galatians", "galatians": "Galatians", "galatian": "Galatians",
    "eph": "Ephesians", "ephesians": "Ephesians", "ephesian": "Ephesians",
    "phi": "Philippians", "phil": "Philippians", "philippians": "Philippians", "philippian": "Philippians",
    "col": "Colossians", "colossians": "Colossians", "colossian": "Colossians",
    "th": "1 Thessalonians", "thes": "1 Thessalonians", "thess": "1 Thessalonians",
    "thessalonians": "1 Thessalonians", "thessalonian": "1 Thessalonians",
    "tim": "1 Timothy", "ti": "1 Timothy", "timothy": "1 Timothy",
    "tit": "Titus", "titus": "Titus",
    "phm": "Philemon", "phil": "Philemon", "philemon": "Philemon",
    "heb": "Hebrews", "hebrews": "Hebrews", "hebrew": "Hebrews",
    "jas": "James", "jam": "James", "james": "James",
    "pet": "1 Peter", "peter": "1 Peter", "peters": "1 Peter",
    "jud": "Jude", "jude": "Jude",
    "rev": "Revelation", "revelation": "Revelation",
}


SCRIPTURE_RE_LONG = re.compile(
    r"\b(" + "|".join(re.escape(b) for b in sorted(set(BOOK_NORMALIZE.values()), key=len, reverse=True)) + r")s?\s+(\d+)(?::(\d+(?:[a-z]?)?(?:\s*[-–]\s*\d+(?:[a-z]?)?)*))?",
    re.IGNORECASE,
)


def extract_scriptures(text: str) -> list[dict]:
    """Extract scripture references from a passage. Returns list with
    {book, chapter, verse_start, verse_end, raw, jw_url}."""
    found = []
    seen = set()  # dedup by (start, end, raw_normalized)
    seen_spans = []  # to dedup overlapping matches

    def _add(book_key, chapter, verse, raw, span):
        # Normalize: strip leading/trailing whitespace from raw
        norm_raw = raw.strip().lower()
        # Skip if same span already covered
        for s_start, s_end in seen_spans:
            if not (span[1] <= s_start or span[0] >= s_end):
                return  # overlaps
        if norm_raw in seen:
            return
        seen.add(norm_raw)
        seen_spans.append(span)
        found.append((book_key, chapter, verse, raw))

    # Pass 1: short book abbreviations (case-sensitive — avoid 'And', 'The', etc.)
    for m in SCRIPTURE_PATTERN.finditer(text):
        book_raw = m.group(1).strip()
        book_key = BOOK_NORMALIZE.get(book_raw)
        if not book_key:
            continue
        _add(book_key, int(m.group(2)), m.group(3), m.group(0), m.span())

    # Pass 1b: multi-word abbreviations
    for m in MULTIWORD_PATTERN.finditer(text):
        book_raw = m.group(1).strip()
        prefix_m = re.match(r"^([1-3])\s+(.+)$", book_raw)
        if prefix_m:
            num = prefix_m.group(1)
            short = prefix_m.group(2).lower()
            full_suffix = {
                "cor": "Corinthians", "co": "Corinthians", "corinthians": "Corinthians",
                "sam": "Samuel", "sa": "Samuel", "samuel": "Samuel",
                "ki": "Kings", "kgs": "Kings", "kings": "Kings", "king": "Kings",
                "ch": "Chronicles", "chr": "Chronicles", "chronicles": "Chronicles",
                "thes": "Thessalonians", "thess": "Thessalonians",
                "thessalonians": "Thessalonians", "thessalonian": "Thessalonians",
                "tim": "Timothy", "ti": "Timothy", "timothy": "Timothy",
                "pe": "Peter", "pet": "Peter", "peter": "Peter",
                "jn": "John", "joh": "John", "jo": "John", "john": "John", "joe": "John",
            }.get(short)
            if full_suffix:
                book_key = f"{num} {full_suffix}"
            else:
                continue
        else:
            book_key = BOOK_NORMALIZE.get(book_raw)
            if not book_key:
                continue
        _add(book_key, int(m.group(2)), m.group(3), m.group(0), m.span())

    # Pass 1c: singular forms
    for m in SINGULAR_PATTERN.finditer(text):
        book_raw = m.group(1).strip()
        prefix_m = re.match(r"^([1-3])\s+(.+)$", book_raw)
        if prefix_m:
            num = prefix_m.group(1)
            short = prefix_m.group(2).lower()
            full_suffix = SINGULAR_BOOK_MAP.get(short)
            if full_suffix:
                book_key = f"{num} {full_suffix}"
            else:
                continue
        else:
            book_key = SINGULAR_BOOK_MAP.get(book_raw.lower())
            if not book_key:
                continue
        _add(book_key, int(m.group(2)), m.group(3), m.group(0), m.span())

    # Pass 2: full book names
    for m in SCRIPTURE_RE_LONG.finditer(text):
        book_raw = m.group(1).strip()
        book_key = BOOK_BY_FULL.get(book_raw.lower(), book_raw)
        if book_key not in BOOK_BY_FULL.values():
            continue
        _add(book_key, int(m.group(2)), m.group(3), m.group(0), m.span())

    # Build output
    results = []
    for book_key, chapter, verse, raw in found:
        verse_start, verse_end = None, None
        if verse:
            first_verse = re.split(r"[–—,]", verse)[0].strip()
            v_match = re.match(r"(\d+)", first_verse)
            verse_start = int(v_match.group(1)) if v_match else None
            if "-" in verse or "–" in verse:
                parts = re.split(r"\s*[-–]\s*", verse)
                if len(parts) == 2:
                    e_match = re.match(r"(\d+)", parts[1])
                    verse_end = int(e_match.group(1)) if e_match else None

        slug = book_key.lower().replace(" ", "-")
        jw_url = f"https://www.jw.org/en/library/bible/study-bible/books/{slug}/{chapter}/"
        if verse_start:
            jw_url += f"#v{book_key[:3].lower()}{chapter}:{verse_start}"

        results.append({
            "book": book_key,
            "chapter": chapter,
            "verseStart": verse_start,
            "verseEnd": verse_end,
            "raw": raw.strip(),
            "jwUrl": jw_url,
        })
    return results


@app.get("/api/scriptures")
def get_scriptures(text: str = Query(..., min_length=1)):
    """Extract scripture references from arbitrary text."""
    refs = extract_scriptures(text)
    return {"count": len(refs), "scriptures": refs}


# ── Build scripture → chunk reverse index (at module load) ──
print("Building scripture cross-reference index...")
SCRIPTURE_INDEX: dict[str, list[dict]] = {}  # "Book Ch:V" → [{id, text, publication}]

def build_scripture_index(batch_size: int = 5000):
    """Walk all chunks, extract scripture refs, build reverse lookup."""
    global SCRIPTURE_INDEX
    SCRIPTURE_INDEX = {}
    offset = 0
    total = collection.count()
    while offset < total:
        batch = collection.get(limit=batch_size, offset=offset, include=["documents", "metadatas"])
        if not batch.get("ids"):
            break
        for i, chunk_id in enumerate(batch["ids"]):
            text = batch["documents"][i] if batch.get("documents") else ""
            meta = batch["metadatas"][i] if batch.get("metadatas") else {}
            refs = extract_scriptures(text)
            for ref in refs:
                key = f"{ref['book']} {ref['chapter']}:{ref.get('verseStart', '')}"
                if key not in SCRIPTURE_INDEX:
                    SCRIPTURE_INDEX[key] = []
                if len(SCRIPTURE_INDEX[key]) < 50:
                    SCRIPTURE_INDEX[key].append({
                        "id": chunk_id,
                        "text": text[:300],
                        "publication": meta.get("publication", "Unknown"),
                        "chunk": meta.get("chunk_index", 0),
                    })
        offset += len(batch["ids"])
    print(f"Scripture index: {len(SCRIPTURE_INDEX)} unique verse references mapped")

build_scripture_index()


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
    """Collection statistics + per-publication breakdown."""
    # Per-pub counts — sample enough to get a representative mix
    sample = collection.get(limit=10000, include=["metadatas"])
    pub_counts: Counter = Counter(m.get("publication", "Unknown") for m in sample.get("metadatas") or [])
    total = collection.count()
    # Estimate full counts by extrapolation
    sample_size = len(sample.get("metadatas") or [])
    if 0 < sample_size < total:
        ratio = total / sample_size
        pub_counts = Counter({p: int(c * ratio) for p, c in pub_counts.items()})
    return {
        "name": collection.name,
        "vectors": total,
        "model": MODEL_NAME,
        "publications": [
            {"name": p, "chunks": c}
            for p, c in pub_counts.most_common(20)
        ],
    }


@app.get("/api/related")
def related(id: str = Query(...), k: int = Query(10, ge=1, le=30)):
    """Find passages semantically related to a given chunk id."""
    # Fetch the source chunk
    src = collection.get(ids=[id], include=["documents", "embeddings"])
    if not src.get("documents") or not src["documents"][0]:
        raise HTTPException(404, "Chunk not found")
    embedding = src.get("embeddings", [[]])[0]
    if not embedding:
        # Fallback: re-encode the source
        embedding = model.encode([src["documents"][0]]).tolist()[0]
    # Query
    results = collection.query(query_embeddings=[embedding], n_results=k + 1)
    hits = []
    for i, chunk_id in enumerate(results["ids"][0]):
        if chunk_id == id:
            continue
        meta = results["metadatas"][0][i]
        hits.append({
            "id": chunk_id,
            "score": round(1 - results["distances"][0][i], 4),
            "publication": meta.get("publication", "Unknown"),
            "text": results["documents"][0][i][:600],
        })
        if len(hits) >= k:
            break
    return {"sourceId": id, "results": hits}


@app.get("/api/passage/{chunk_id}")
def get_passage(chunk_id: str):
    """Get a specific passage by id, with extracted scripture references."""
    src = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
    if not src.get("documents") or not src["documents"][0]:
        raise HTTPException(404, "Chunk not found")
    text = src["documents"][0]
    meta = src["metadatas"][0]
    return {
        "id": chunk_id,
        "publication": meta.get("publication", "Unknown"),
        "chunk": meta.get("chunk_index", 0),
        "text": text,
        "scriptures": extract_scriptures(text),
    }


@app.get("/api/xref")
def crossref(q: str = Query(..., min_length=3, description="Scripture reference, e.g. 'James 2:17' or 'John 3:16'")):
    """Find all passages that reference a given scripture.

    Uses the in-memory reverse index built at startup.
    Returns matching chunks with their source publication and text snippet.
    """
    refs = extract_scriptures(q)
    if not refs:
        return {"query": q, "count": 0, "results": [], "hint": "Could not parse scripture reference. Try 'John 3:16' or 'James 2:17'."}

    all_results = []
    seen_ids = set()
    for ref in refs:
        key = f"{ref['book']} {ref['chapter']}:{ref.get('verseStart', '')}"
        # Also try partial key (without verse)
        partial_key = f"{ref['book']} {ref['chapter']}:"
        # Try exact match first
        matches = SCRIPTURE_INDEX.get(key, [])
        # Also try partial matches (same chapter, any verse)
        if not matches:
            for k, v in SCRIPTURE_INDEX.items():
                if k.startswith(partial_key) and len(matches) < 20:
                    for item in v:
                        if item["id"] not in seen_ids:
                            matches.append(item)
                            seen_ids.add(item["id"])
        for item in matches:
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                all_results.append(item)
        if len(all_results) >= 30:
            break

    return {
        "query": q,
        "parsedRefs": refs,
        "count": len(all_results),
        "results": all_results[:30],
    }


@app.get("/api/xref/graph")
def xref_graph(q: str = Query(..., min_length=3, description="Scripture reference, e.g. 'James 2:17'")):
    """Return a scripture graph: the verse + what it cites + what cites it.

    Returns nodes (verses/passages) and edges (citation relationships)
    suitable for a force-directed graph visualization.
    """
    refs = extract_scriptures(q)
    if not refs:
        return {"error": "Could not parse scripture reference"}

    # Get the source verse info
    source_ref = refs[0]
    source_id = f"{source_ref['book']} {source_ref['chapter']}:{source_ref.get('verseStart', '')}"

    # Get passages that cite this verse
    citing = SCRIPTURE_INDEX.get(source_id, [])[:15]

    # Get the passages themselves
    nodes = []
    edges = []
    node_ids = set()

    # Add source node
    nodes.append({
        "id": source_id,
        "label": source_ref["raw"],
        "type": "source",
        "group": 0,
    })
    node_ids.add(source_id)

    # For each citing passage, extract what other verses it cites
    for item in citing[:10]:
        text = item.get("text", "")
        cited_in_passage = extract_scriptures(text)

        # Add passage node
        passage_id = item["id"]
        if passage_id not in node_ids:
            nodes.append({
                "id": passage_id,
                "label": item.get("publication", "Pub"),
                "type": "passage",
                "group": 1,
                "text": text[:100],
            })
            node_ids.add(passage_id)

        # Edge: passage → source (the passage cites the source verse)
        edges.append({
            "source": passage_id,
            "target": source_id,
            "label": "cites",
        })

        # Add verse nodes + edges for other verses in the passage
        for ref in cited_in_passage[:5]:
            ref_id = f"{ref['book']} {ref['chapter']}:{ref.get('verseStart', '')}"
            if ref_id == source_id:
                continue
            if ref_id not in node_ids:
                nodes.append({
                    "id": ref_id,
                    "label": ref.get("raw", ref_id),
                    "type": "verse",
                    "group": 2,
                })
                node_ids.add(ref_id)
            edges.append({
                "source": passage_id,
                "target": ref_id,
                "label": "also cites",
            })

    return {
        "query": q,
        "sourceId": source_id,
        "nodes": nodes,
        "edges": edges,
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
