from __future__ import annotations

"""
Local RAG index (TF-IDF + cosine similarity).

Indexed corpora (curated only):
  - reports/knowledge_base/**  (.md, .txt)
  - reports/criteres_score_et_crispdm.md

Excluded: generated reports, demo notes, dev scratch files.
"""

from dataclasses import dataclass
from pathlib import Path
import re
import time
import joblib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import MODELS_DIR, REPORTS_DIR


RAG_INDEX_FILE = MODELS_DIR / "rag_index.joblib"

# French stop words (compact set — enough for TF-IDF noise reduction)
_FRENCH_STOP = {
    "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux", "et", "ou", "en",
    "dans", "pour", "par", "sur", "avec", "sans", "ce", "cet", "cette", "ces", "son",
    "sa", "ses", "leur", "leurs", "qui", "que", "dont", "où", "est", "sont", "a", "as",
    "ont", "été", "etre", "être", "avoir", "mais", "donc", "or", "ni", "car", "si",
    "comme", "plus", "moins", "très", "tres", "tout", "tous", "toute", "toutes",
    "the", "and", "or", "of", "to", "in", "for", "on", "at", "by", "from",
}

# Files under reports/ root allowed in addition to knowledge_base/
_EXTRA_REPORT_FILES = ("criteres_score_et_crispdm.md",)

_DEFAULT_MIN_SCORE = 0.06


@dataclass
class RagIndex:
    vectorizer: TfidfVectorizer
    matrix: np.ndarray
    chunks: list[dict]  # {source, chunk_id, text}
    built_at: float


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore")


def _normalize_for_index(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_by_paragraphs(text: str, chunk_size: int = 850, overlap: int = 100) -> list[str]:
    """Prefer paragraph boundaries, then fixed-size windows."""
    text = _normalize_for_index(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    merged: list[str] = []
    buf = ""
    for p in paragraphs:
        candidate = f"{buf}\n\n{p}".strip() if buf else p
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf:
                merged.append(buf)
            if len(p) <= chunk_size:
                buf = p
            else:
                # Very long paragraph — fall back to sliding window
                i = 0
                while i < len(p):
                    j = min(len(p), i + chunk_size)
                    merged.append(p[i:j])
                    if j == len(p):
                        break
                    i = max(0, j - overlap)
                buf = ""
    if buf:
        merged.append(buf)

    # Secondary split for oversized merged blocks
    out: list[str] = []
    for block in merged:
        if len(block) <= chunk_size:
            out.append(block)
            continue
        i = 0
        while i < len(block):
            j = min(len(block), i + chunk_size)
            out.append(block[i:j])
            if j == len(block):
                break
            i = max(0, j - overlap)
    return out


def _collect_source_files(knowledge_dir: Path | None = None) -> list[Path]:
    knowledge_dir = knowledge_dir or (REPORTS_DIR / "knowledge_base")
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    files_kb = sorted(
        p for p in knowledge_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt"}
    )
    files_extra = [
        REPORTS_DIR / name
        for name in _EXTRA_REPORT_FILES
        if (REPORTS_DIR / name).is_file()
    ]
    return sorted(set(files_kb + files_extra))


def _newest_source_mtime(files: list[Path]) -> float:
    if not files:
        return 0.0
    return max(p.stat().st_mtime for p in files)


def _index_is_stale(files: list[Path]) -> bool:
    if not RAG_INDEX_FILE.exists():
        return True
    try:
        payload = joblib.load(RAG_INDEX_FILE)
        built_at = float(payload.get("built_at", 0)) if isinstance(payload, dict) else 0.0
        index_mtime = RAG_INDEX_FILE.stat().st_mtime
        source_mtime = _newest_source_mtime(files)
        return source_mtime > max(built_at, index_mtime)
    except Exception:
        return True


def build_rag_index(knowledge_dir: Path | None = None, *, force: bool = False) -> Path:
    files = _collect_source_files(knowledge_dir)
    if not files:
        raise FileNotFoundError(
            f"No RAG source files in {knowledge_dir or REPORTS_DIR / 'knowledge_base'}"
        )

    if not force and not _index_is_stale(files):
        return RAG_INDEX_FILE

    chunks: list[dict] = []
    texts: list[str] = []

    for p in files:
        raw = _read_text_file(p)
        try:
            rel = p.relative_to(REPORTS_DIR)
        except ValueError:
            rel = Path(p.name)
        for k, ch in enumerate(_chunk_by_paragraphs(raw)):
            chunks.append({"source": str(rel).replace("\\", "/"), "chunk_id": k, "text": ch})
            texts.append(ch)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        max_features=30_000,
        min_df=1,
        stop_words=list(_FRENCH_STOP),
        sublinear_tf=True,
    )
    mat = vectorizer.fit_transform(texts)

    payload = {
        "vectorizer": vectorizer,
        "matrix": mat,
        "chunks": chunks,
        "built_at": time.time(),
        "source_files": [str(f) for f in files],
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, RAG_INDEX_FILE)
    return RAG_INDEX_FILE


def load_rag_index(*, auto_rebuild: bool = True) -> RagIndex:
    files = _collect_source_files()
    if auto_rebuild and _index_is_stale(files):
        build_rag_index(force=True)
    elif not RAG_INDEX_FILE.exists():
        build_rag_index(force=True)

    obj = joblib.load(RAG_INDEX_FILE)
    if isinstance(obj, dict):
        return RagIndex(
            vectorizer=obj["vectorizer"],
            matrix=obj["matrix"],
            chunks=obj["chunks"],
            built_at=float(obj.get("built_at", 0)),
        )
    if hasattr(obj, "vectorizer") and hasattr(obj, "matrix") and hasattr(obj, "chunks"):
        return RagIndex(
            vectorizer=obj.vectorizer,
            matrix=obj.matrix,
            chunks=obj.chunks,
            built_at=float(getattr(obj, "built_at", 0)),
        )
    raise TypeError(f"Invalid RAG index artifact type: {type(obj)}")


def retrieve(
    query: str,
    k: int = 4,
    *,
    min_score: float = _DEFAULT_MIN_SCORE,
) -> list[dict]:
    """Retrieve top-k chunks for a single query string."""
    if not (query or "").strip():
        return []

    idx = load_rag_index()
    qv = idx.vectorizer.transform([query.strip()])
    sims = cosine_similarity(qv, idx.matrix).ravel()

    order = np.argsort(-sims)
    out: list[dict] = []
    for i in order:
        score = float(sims[int(i)])
        if score < min_score:
            break
        item = dict(idx.chunks[int(i)])
        item["score"] = score
        out.append(item)
        if len(out) >= k:
            break

    # If everything is below threshold, return best effort (top 1) for REF-08 grounding
    if not out and len(order) > 0:
        i = int(order[0])
        item = dict(idx.chunks[i])
        item["score"] = float(sims[i])
        out.append(item)

    return out


def retrieve_multi(
    queries: list[str],
    k: int = 5,
    *,
    min_score: float = _DEFAULT_MIN_SCORE,
) -> list[dict]:
    """Merge results from several queries (keeps best score per chunk)."""
    merged: dict[tuple[str, int], dict] = {}
    for q in queries:
        for item in retrieve(q, k=k, min_score=min_score):
            key = (str(item.get("source")), int(item.get("chunk_id", 0)))
            prev = merged.get(key)
            if prev is None or float(item["score"]) > float(prev["score"]):
                merged[key] = item
    ranked = sorted(merged.values(), key=lambda x: float(x.get("score", 0)), reverse=True)
    return ranked[:k]


if __name__ == "__main__":
    path = build_rag_index(force=True)
    print(f"RAG index saved at {path}")
    samples = retrieve_multi(
        [
            "score KYC probabilité défaut",
            "LSTM GRU transactions remboursements",
            "GraphSAGE réseau relationnel",
        ],
        k=5,
    )
    for s in samples:
        print(f"- {s['source']}#{s['chunk_id']} score={s['score']:.3f} :: {s['text'][:70]}...")
