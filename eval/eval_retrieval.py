"""
Retrieval evaluation for the literature RAG layer.

For each gold question (a clinically-phrased query mapped to the relevant PMID(s)
in our corpus, built independently of the retriever), check whether the retriever
surfaces a relevant paper. Reports recall@k and MRR. Validates that every gold
PMID actually exists in the corpus so the eval can't be silently wrong. Local, $0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "tools"))
from search_literature import search_literature  # noqa: E402

GOLD = ROOT / "eval" / "rag_gold.json"
CORPUS = ROOT / "data" / "hf_corpus.jsonl"
KS = [1, 3, 5, 10]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    corpus_pmids = {json.loads(line)["pmid"] for line in CORPUS.open(encoding="utf-8")}
    gold = json.loads(GOLD.read_text(encoding="utf-8"))

    # validate the gold set against the corpus -- drop a gold PMID the corpus lacks
    valid = []
    for g in gold:
        present = [p for p in g["relevant_pmids"] if p in corpus_pmids]
        missing = [p for p in g["relevant_pmids"] if p not in corpus_pmids]
        if missing:
            print(f"NOTE: gold PMIDs not in corpus {missing} for: {g['question'][:55]}")
        if not present:
            print(f"WARNING: no usable gold PMID -> skipping: {g['question'][:55]}")
            continue
        valid.append({"question": g["question"], "relevant": set(present)})
    print(f"\n{len(valid)} / {len(gold)} gold questions usable\n")

    recall = {k: 0 for k in KS}
    rr_sum = 0.0
    misses = []
    for g in valid:
        ranked = [r["pmid"] for r in search_literature(g["question"], k=max(KS))]
        first = next((i + 1 for i, p in enumerate(ranked) if p in g["relevant"]), None)
        rr_sum += (1.0 / first) if first else 0.0
        for k in KS:
            if any(p in g["relevant"] for p in ranked[:k]):
                recall[k] += 1
        if not first or first > 5:
            misses.append((g["question"], first))

    n = len(valid)
    print("Retrieval quality:")
    for k in KS:
        print(f"  recall@{k:<2d}: {recall[k] / n:.2f}  ({recall[k]}/{n})")
    print(f"  MRR@{max(KS)} : {rr_sum / n:.3f}")
    if misses:
        print(f"\nQuestions with no relevant hit in top-5 ({len(misses)}):")
        for q, rank in misses:
            print(f"  first_hit_rank={rank}  {q[:72]}")


if __name__ == "__main__":
    main()
