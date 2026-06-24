"""
lookup_guideline -- retrieve passages from authoritative guideline / GDMT documents only.

Reuses the shared FAISS retriever but restricts results to the subset of the corpus
that are practice guidelines, consensus/position statements, or guideline-directed
medical-therapy (GDMT) documents. Local, zero cost per query.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "tools"))
from search_literature import get_retriever  # noqa: E402

CORPUS = ROOT / "data" / "hf_corpus.jsonl"
_GUIDELINE_TYPES = {"Guideline", "Practice Guideline",
                    "Consensus Development Conference", "Consensus Statement"}
_GUIDELINE_WORDS = ("guideline", "recommendation", "consensus", "position statement")
_guideline_pmids: set | None = None


def _is_guideline(rec: dict) -> bool:
    if set(rec.get("pub_types", [])) & _GUIDELINE_TYPES:
        return True
    title = rec["title"].lower()
    return any(w in title for w in _GUIDELINE_WORDS)


def _pmids() -> set:
    global _guideline_pmids
    if _guideline_pmids is None:
        ids = set()
        for line in CORPUS.open(encoding="utf-8"):
            rec = json.loads(line)
            if _is_guideline(rec):
                ids.add(rec["pmid"])
        _guideline_pmids = ids
    return _guideline_pmids


def lookup_guideline(topic: str, k: int = 3) -> list[dict]:
    """Top-k passages drawn only from guideline / GDMT documents, with PMIDs to cite."""
    return get_retriever().search(topic, k, allowed_pmids=_pmids())


if __name__ == "__main__":
    print(f"guideline documents in corpus: {len(_pmids())}\n")
    for topic in [
        "first-line pharmacologic therapy for heart failure with reduced ejection fraction",
        "device therapy recommendations for heart failure",
    ]:
        print("TOPIC:", topic)
        for r in lookup_guideline(topic, 3):
            print(f"  {r['score']:.3f}  PMID {r['pmid']} ({r['year']})  {r['title'][:62]}")
        print()
