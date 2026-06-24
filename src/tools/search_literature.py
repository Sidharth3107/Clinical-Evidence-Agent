"""
search_literature -- the agent's evidence-retrieval tool.

Embeds a clinician query with the same model used to build the index, runs an
exact cosine search over the chunked PubMed corpus, collapses chunks to the best
one per paper, and returns the top-k DISTINCT sources with their PMID for
citation. Pure local retrieval -- no API calls, zero cost per query.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT / "data" / "faiss_index.bin"
META_PATH = ROOT / "data" / "chunks_meta.json"
# Single source of truth for the embedder: the query must be embedded with the SAME
# model that built the index, so import it from build_index rather than duplicating it.
sys.path.insert(0, str(ROOT / "src" / "rag"))
from build_index import MODEL_NAME  # noqa: E402


class LiteratureRetriever:
    """Loads the model + index + metadata once; reuse across many queries."""

    def __init__(self) -> None:
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(str(INDEX_PATH))
        self.meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    def search(self, query: str, k: int = 5, min_score: float = 0.0,
               allowed_pmids: set | None = None) -> list[dict]:
        k = max(1, min(int(k), 20))
        # when filtering to a sub-corpus, scan the whole (tiny) index so we don't miss
        # matches; otherwise over-fetch and collapse to unique papers (best chunk wins)
        n_fetch = self.index.ntotal if allowed_pmids else min(max(k * 6, 30), self.index.ntotal)
        qv = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        scores, idxs = self.index.search(qv, n_fetch)

        best: dict[str, dict] = {}
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:                 # faiss pads with -1 when fewer hits than asked
                continue
            m = self.meta[i]
            if allowed_pmids is not None and m["pmid"] not in allowed_pmids:
                continue
            score = float(score)
            if score < min_score:
                continue
            pmid = m["pmid"]
            if pmid not in best:      # results are score-sorted: first seen = best chunk
                best[pmid] = {
                    "pmid": pmid,
                    "title": m["title"],
                    "journal": m["journal"],
                    "year": m["year"],
                    "score": round(score, 4),
                    "text": m["text"],
                }
            if len(best) >= k:
                break
        return list(best.values())


_retriever: LiteratureRetriever | None = None


def get_retriever() -> LiteratureRetriever:
    """Lazy-load the shared retriever once (reused by lookup_guideline too)."""
    global _retriever
    if _retriever is None:
        _retriever = LiteratureRetriever()
    return _retriever


def search_literature(query: str, k: int = 5) -> list[dict]:
    """Top-k distinct cited passages for a query. Lazy-loads the retriever once."""
    return get_retriever().search(query, k)


if __name__ == "__main__":
    demo = [
        "Do SGLT2 inhibitors reduce mortality in heart failure?",
        "Does spironolactone help reduced ejection fraction?",
        "What is the capital of France?",   # off-topic: scores should be low
    ]
    for q in demo:
        print("Q:", q)
        for r in search_literature(q, k=3):
            print(f"  {r['score']:.3f}  PMID {r['pmid']} ({r['year']})  {r['title'][:70]}")
        print()
