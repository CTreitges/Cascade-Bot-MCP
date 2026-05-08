"""Cross-task memory: best-effort persistence of decisions / findings / facts
that should outlive a single cascade run.

Two backends, used in this priority order:
  1. HTTP — if `RLM_HTTP_ENDPOINT` is set, POST {category, content, tags,
     importance, project} to that URL. Used when an RLM-server is running
     side-by-side and exposes /remember.
  2. Local JSONL — always-on fallback. Append one JSON object per line to
     `<CASCADE_HOME>/store/memory.jsonl`. This is the durable record even
     when nothing else is reachable, and can be replayed into a real RLM
     later.

Reads (`recall_context`) walk the local JSONL backwards looking for tagged
entries whose tag-set intersects the query keywords.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Literal

from .config import settings

log = logging.getLogger("cascade.memory")

PROJECT = "cascade-bot-mcp"


def _memory_path() -> Path:
    s = settings()
    p = s.cascade_home / "store" / "memory.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


async def _http_post(url: str, body: dict, *, timeout_s: float = 10) -> bool:
    """POST a memory entry to an external RLM endpoint. Returns True on 2xx."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, json=body)
        return 200 <= r.status_code < 300
    except Exception as e:
        log.debug("rlm http post failed: %s", e)
        return False


def _append_jsonl(entry: dict) -> bool:
    """Synchronous, append-only — runs in to_thread."""
    path = _memory_path()
    line = json.dumps(entry, ensure_ascii=False, default=str)
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return True
    except Exception as e:
        log.warning("memory jsonl append failed: %s", e)
        return False


async def remember_finding(
    content: str,
    *,
    category: Literal["finding", "decision", "preference", "fact"] = "finding",
    importance: Literal["low", "medium", "high", "critical"] = "medium",
    tags: str = "cascade-bot-mcp",
    extra: dict[str, Any] | None = None,
) -> bool:
    """Record an insight. Best-effort: never raises, returns True iff at least
    one backend succeeded."""
    entry = {
        "ts": time.time(),
        "project": PROJECT,
        "category": category,
        "importance": importance,
        "tags": tags,
        "content": content,
        **(extra or {}),
    }

    ok = False

    # 1) external RLM via HTTP (if configured)
    url = os.getenv("RLM_HTTP_ENDPOINT")
    if url:
        if await _http_post(url, entry):
            ok = True

    # 2) local JSONL (always)
    if await asyncio.to_thread(_append_jsonl, entry):
        ok = True

    # 3) Plan v5 R4 — RAG-Index (best-effort, no-op wenn deps fehlen).
    #    Embeds new insights damit semantic-recall sie findet auch wenn keine
    #    BM25-Term-Overlap besteht.
    try:
        await asyncio.to_thread(_rag_upsert_entry, entry)
    except Exception as e:
        log.debug("rag upsert failed: %s", e)

    if ok:
        log.info("memory[%s/%s] %s", category, importance, content[:120])
    return ok


async def remember_decision(content: str, **kw: Any) -> bool:
    return await remember_finding(content, category="decision", **kw)


async def remember_fact(content: str, **kw: Any) -> bool:
    return await remember_finding(content, category="fact", **kw)


async def cleanup_old_entries(*, retention_days: int = 90) -> int:
    """Trim the memory.jsonl: drop entries older than retention_days.
    Returns count removed. No-op if file doesn't exist."""
    path = _memory_path()
    if not path.exists():
        return 0

    def _do() -> int:
        cutoff = time.time() - retention_days * 86400
        kept: list[str] = []
        removed = 0
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except Exception:
                        kept.append(line.rstrip("\n"))
                        continue
                    if (e.get("ts") or 0) >= cutoff:
                        kept.append(line.rstrip("\n"))
                    else:
                        removed += 1
            if removed:
                path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        except Exception as e:
            log.warning("memory cleanup failed: %s", e)
        return removed

    return await asyncio.to_thread(_do)


# Minimal stopword list (DE + EN). Kept small on purpose — the BM25 IDF
# already discounts common terms, but trimming the obvious ones first
# keeps the tokenized query focused and avoids matching "for"/"the"/etc.
_STOPWORDS: frozenset[str] = frozenset({
    # English
    "the", "and", "for", "with", "from", "this", "that", "have", "has",
    "are", "was", "were", "you", "your", "our", "out", "into", "but",
    "not", "all", "any", "can", "will", "would", "could", "should", "may",
    "might", "what", "which", "who", "how", "why", "when", "where",
    "there", "here", "than", "then", "them", "they", "their", "ours",
    # German
    "und", "oder", "aber", "der", "die", "das", "den", "dem", "des",
    "ein", "eine", "einen", "einem", "einer", "ist", "war", "sind", "waren",
    "ich", "du", "er", "sie", "es", "wir", "ihr", "mit", "von", "zu",
    "bei", "auf", "für", "über", "unter", "nach", "vor", "ohne", "gegen",
    "wie", "wann", "warum", "wo", "was", "wer", "ja", "nein", "nicht",
    "doch", "mal", "schon", "noch", "auch", "nur", "sehr", "dann",
})


def _tokenize(text: str, *, min_len: int = 3) -> list[str]:
    """Lowercase, split on non-alnum, filter stopwords + min length.
    Returns a list of tokens (preserving multiplicity for term-frequency)."""
    if not text:
        return []
    out: list[str] = []
    cur: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                tok = "".join(cur)
                cur = []
                if len(tok) >= min_len and tok not in _STOPWORDS:
                    out.append(tok)
    if cur:
        tok = "".join(cur)
        if len(tok) >= min_len and tok not in _STOPWORDS:
            out.append(tok)
    return out


_IMPORTANCE_BOOST = {
    "critical": 1.30,
    "high":     1.15,
    "medium":   1.00,
    "low":      0.85,
}


def _bm25_score(
    query_terms: list[str],
    doc_terms: list[str],
    df: dict[str, int],
    n_docs: int,
    avgdl: float,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Vanilla BM25 score for one (query, doc). df = document-frequency map
    over the whole collection. Returns 0.0 if no query term hits."""
    if not query_terms or not doc_terms:
        return 0.0
    dl = len(doc_terms)
    # Term-frequency map for this doc
    tf: dict[str, int] = {}
    for t in doc_terms:
        tf[t] = tf.get(t, 0) + 1
    score = 0.0
    import math
    for q in set(query_terms):
        f = tf.get(q, 0)
        if f == 0:
            continue
        n_q = df.get(q, 0)
        idf = math.log(1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5))
        norm = f * (k1 + 1.0) / (f + k1 * (1.0 - b + b * (dl / max(avgdl, 1.0))))
        score += idf * norm
    return score


# Plan v5 R4 — Lazy-Singleton für RagStore. Persist-dir wird beim ersten
# Zugriff erstellt. Wenn chromadb / sentence-transformers fehlen oder der
# Store-Init scheitert: dauerhaft None → recall_context fällt sauber auf
# BM25-only zurück.
_RAG_STORE: Any = None
_RAG_INIT_TRIED: bool = False


def _get_rag_store() -> Any:
    """Lazy lookup. Returns RagStore wenn deps + persist-dir ok, sonst None."""
    global _RAG_STORE, _RAG_INIT_TRIED
    if _RAG_STORE is not None:
        return _RAG_STORE
    if _RAG_INIT_TRIED:
        return None
    _RAG_INIT_TRIED = True
    try:
        from .rag import RagStore, is_available
        if not is_available():
            return None
        s = settings()
        _RAG_STORE = RagStore(persist_dir=s.cascade_home / "store" / "rag")
        # warm-up: ensure client is loaded so ersten search nicht blockiert
        try:
            _RAG_STORE._ensure_client()
        except Exception:
            pass
        return _RAG_STORE
    except Exception as e:
        log.debug("rag store lazy-init failed: %s", e)
        _RAG_STORE = None
        return None


def _rlm_entry_id(entry: dict) -> str:
    """Stabile ID für ein RLM-Entry: hash über (ts, content[:200])."""
    import hashlib
    h = hashlib.sha1(
        f"{entry.get('ts', 0):.0f}|{(entry.get('content') or '')[:200]}".encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:16]
    return f"rlm:{h}"


def _rag_upsert_entry(entry: dict) -> None:
    """Synchronous helper (run via to_thread). Embedded ein RLM-Entry in den
    RAG-Index. Skipt sauber wenn deps fehlen oder Store nicht init."""
    store = _get_rag_store()
    if store is None:
        return
    try:
        from .rag import RagDoc
        text = (entry.get("content") or "").strip()
        if not text:
            return
        doc = RagDoc(
            id=_rlm_entry_id(entry),
            text=text,
            source="rlm",
            metadata={
                "category": str(entry.get("category", "")),
                "importance": str(entry.get("importance", "")),
                "tags": str(entry.get("tags", "")),
            },
        )
        store.upsert([doc])
    except Exception as e:
        log.debug("rag upsert (entry) failed: %s", e)


async def recall_context(task: str, *, limit: int = 3) -> str | None:
    """BM25-ranked recall over the local memory.jsonl. Searches across both
    `content` and `file_content`-like fields plus tags. Importance metadata
    nudges the ranking (`high`/`critical` rank slightly higher).

    Plan v5 R4: zusätzlich semantic-recall via RAG (chromadb+sentence-
    transformers) wenn deps installiert. RLM (BM25, primary, weight=1.5)
    und RAG (vector, secondary, weight=1.0) werden via reciprocal-rank-
    fusion kombiniert. Wenn RAG-Layer no-op ist, fällt der Pfad sauber
    auf reines BM25 zurück.

    Returns a bullet-list of the top `limit` matches, or None if nothing
    scores above zero.
    """
    path = _memory_path()
    if not path.exists():
        return None

    def _scan_and_rank() -> list[tuple[float, dict]]:
        q_terms = _tokenize(task)
        if not q_terms:
            return []

        # First pass — load entries + tokenize. This is bounded by the
        # JSONL size; we cap at the latest 5000 entries to keep recall
        # snappy even when the file has grown over months of use.
        entries: list[dict] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            return []
        if len(entries) > 5000:
            entries = entries[-5000:]
        if not entries:
            return []

        # Build per-doc token lists + document-frequency map
        docs: list[list[str]] = []
        df: dict[str, int] = {}
        for e in entries:
            haystack = " ".join(
                str(e.get(k, ""))
                for k in ("content", "tags", "category")
            )
            tokens = _tokenize(haystack)
            docs.append(tokens)
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        n_docs = len(docs)
        avgdl = sum(len(d) for d in docs) / max(n_docs, 1)

        # Score every doc; keep only those with score>0
        scored: list[tuple[float, dict]] = []
        for e, dt in zip(entries, docs):
            s = _bm25_score(q_terms, dt, df, n_docs, avgdl)
            if s <= 0.0:
                continue
            boost = _IMPORTANCE_BOOST.get(e.get("importance", "medium"), 1.0)
            scored.append((s * boost, e))

        scored.sort(key=lambda p: p[0], reverse=True)
        return scored[:limit]

    ranked = await asyncio.to_thread(_scan_and_rank)

    # Plan v5 R4 — RAG-Augmentation. Liefert leere Liste wenn deps fehlen
    # oder Index leer. Best-effort, jeder Fehler → bm25-only fallback.
    # Min-Similarity-Threshold (0.30) damit cosine-similarity nicht
    # konstant-low irrelevante Hits durchlässt — RAG soll nur ergänzen
    # wenn semantisch wirklich nahe, nicht "irgendwas" zurückwerfen.
    def _rag_search() -> list:
        store = _get_rag_store()
        if store is None:
            return []
        try:
            hits = store.search(task, n=max(limit, 5))
            return [h for h in hits if h.score >= 0.30]
        except Exception as e:
            log.debug("rag search failed: %s", e)
            return []

    rag_hits = await asyncio.to_thread(_rag_search)

    if not ranked and not rag_hits:
        return None

    rlm_dicts: list[dict] = []
    for score, e in ranked or []:
        rlm_dicts.append({
            "id": _rlm_entry_id(e),
            "content": (e.get("content") or "")[:240],
            "category": e.get("category", "?"),
            "importance": e.get("importance", "?"),
            "bm25_score": score,
        })

    if rag_hits:
        try:
            from .rag import reciprocal_rank_fusion
            fused = reciprocal_rank_fusion(rlm_dicts, rag_hits)[:limit]
        except Exception as e:
            log.debug("rrf failed, falling back to bm25-only: %s", e)
            fused = rlm_dicts[:limit]
    else:
        fused = rlm_dicts[:limit]

    if not fused:
        return None

    lines = []
    for h in fused:
        meta = h.get("metadata") or {}
        cat = meta.get("category") or h.get("category", "?")
        imp = meta.get("importance") or h.get("importance", "?")
        src = h.get("source", "rlm")
        score = h.get("rrf_score") or h.get("bm25_score") or h.get("score", 0.0)
        content = (h.get("content") or "")[:240]
        src_marker = f" via {src}" if src and src != "rlm" else ""
        lines.append(f"  [{cat}/{imp} score={score:.3f}{src_marker}] {content}")
    return "\n".join(lines)
