"""
A/B retrieval embedders (and an optional cross-encoder reranker) on the gold set.

Picks the retrieval stack by MEASURED recall@k / MRR -- not by reputation. For each
candidate embedder it builds an in-memory FAISS index, scores it against
eval/rag_gold.json, then optionally reranks the top candidates with a cross-encoder.
Reuses the exact chunking from build_index.py so the comparison is apples-to-apples,
and never touches the production index. Local, CPU, $0 (downloads each model once).

Run:  python eval/compare_embedders.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "rag"))
from build_index import OVERLAP_WORDS, WORDS_PER_CHUNK, chunk_words  # noqa: E402

CORPUS = ROOT / "data" / "hf_corpus.jsonl"
GOLD = ROOT / "eval" / "rag_gold.json"
KS = [1, 3, 5, 10]
N_RERANK = 30  # candidate chunks fed to the cross-encoder per query

EMBEDDERS = [
    "all-MiniLM-L6-v2",                  # current baseline (general-domain)
    "pritamdeka/S-PubMedBert-MS-MARCO",  # biomedical, retrieval-tuned
]
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_gold(corpus_pmids: set) -> list[dict]:
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    out = []
    for g in gold:
        rel = {p for p in g["relevant_pmids"] if p in corpus_pmids}
        if rel:
            out.append({"q": g["question"], "rel": rel})
    return out


def chunk_corpus(records: list[dict]):
    pmids, texts = [], []
    for r in records:
        for piece in chunk_words(r["abstract"], WORDS_PER_CHUNK, OVERLAP_WORDS):
            pmids.append(r["pmid"])
            texts.append(piece)
    return pmids, texts


def ranked_pmids(query, index, pmids, texts, embedder, reranker=None) -> list[str]:
    qv = embedder.encode([query], normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False).astype("float32")
    _, idxs = index.search(qv, min(N_RERANK, index.ntotal))
    cand = [(pmids[i], texts[i]) for i in idxs[0] if i >= 0]
    if reranker is not None:
        scores = reranker.predict([(query, t) for _, t in cand])
        cand = [c for _, c in sorted(zip(scores, cand), key=lambda x: -float(x[0]))]
    order, seen = [], set()
    for pmid, _ in cand:               # collapse chunks -> unique papers, best first
        if pmid not in seen:
            seen.add(pmid)
            order.append(pmid)
    return order


def score(gold, index, pmids, texts, embedder, reranker=None) -> dict:
    recall = {k: 0 for k in KS}
    rr = 0.0
    for g in gold:
        order = ranked_pmids(g["q"], index, pmids, texts, embedder, reranker)
        first = next((i + 1 for i, p in enumerate(order) if p in g["rel"]), None)
        rr += (1.0 / first) if first else 0.0
        for k in KS:
            if any(p in g["rel"] for p in order[:k]):
                recall[k] += 1
    n = len(gold)
    return {**{f"R@{k}": recall[k] / n for k in KS}, "MRR": rr / n}


def main() -> None:
    records = [json.loads(line) for line in CORPUS.open(encoding="utf-8")]
    pmids, texts = chunk_corpus(records)
    gold = load_gold({r["pmid"] for r in records})
    print(f"{len(records)} abstracts -> {len(texts)} chunks; {len(gold)} gold questions\n")

    reranker = CrossEncoder(RERANKER)
    header = f"{'config':54s}" + "".join(f"  R@{k}" for k in KS) + "    MRR"
    print(header)
    print("-" * len(header))

    rows = []
    for name in EMBEDDERS:
        emb = SentenceTransformer(name)
        vecs = emb.encode(texts, batch_size=64, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False).astype("float32")
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        for label, rr in [("", None), ("  + reranker", reranker)]:
            m = score(gold, index, pmids, texts, emb, rr)
            short = name.split("/")[-1] + label
            print(f"{short:54s}" + "".join(f"  {m[f'R@{k}']:.2f}" for k in KS) + f"   {m['MRR']:.3f}")
            rows.append((short, m))

    best = max(rows, key=lambda r: (r[1]["R@1"], r[1]["MRR"]))
    print(f"\nbest by (R@1, MRR): {best[0]}")


if __name__ == "__main__":
    main()
