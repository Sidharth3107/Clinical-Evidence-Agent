"""
Build the retrieval index for the HF literature corpus.

Pipeline:  load data/hf_corpus.jsonl -> chunk each abstract (carrying the PMID) ->
embed chunks with all-MiniLM-L6-v2 (local, CPU) -> L2-normalize -> FAISS
IndexFlatIP (cosine, exact) -> persist index + metadata aligned to index rows.

Everything runs locally. No API calls, zero cost.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "hf_corpus.jsonl"
INDEX_PATH = ROOT / "data" / "faiss_index.bin"
META_PATH = ROOT / "data" / "chunks_meta.json"

MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"  # biomedical, retrieval-tuned (chosen by eval/compare_embedders.py)
WORDS_PER_CHUNK = 150   # ~210 tokens -> safely under the embedder's max sequence length
OVERLAP_WORDS = 30      # carry context across chunk boundaries


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Sliding word window so long abstracts never get silently truncated."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def main() -> None:
    records = [json.loads(line) for line in CORPUS.open(encoding="utf-8")]
    print(f"loaded {len(records)} abstracts")

    # 1) chunk -- one row per chunk, PMID + citation metadata carried on each
    meta: list[dict] = []
    texts: list[str] = []
    for r in records:
        for i, piece in enumerate(chunk_words(r["abstract"], WORDS_PER_CHUNK, OVERLAP_WORDS)):
            meta.append({
                "pmid": r["pmid"],
                "title": r["title"],
                "journal": r["journal"],
                "year": r["year"],
                "chunk_id": f'{r["pmid"]}-{i}',
                "text": piece,
            })
            texts.append(piece)
    print(f"produced {len(texts)} chunks (avg {len(texts) / len(records):.1f} per abstract)")

    # 2) embed locally on CPU; normalize so inner product == cosine similarity
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    print(f"embeddings: {emb.shape}  (dim={emb.shape[1]})")

    # 3) exact cosine index -- a flat index is instant at this scale (~hundreds of vectors)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    print(f"index size: {index.ntotal} vectors")

    # 4) persist index + metadata aligned to index row order
    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {INDEX_PATH.name} and {META_PATH.name}")


if __name__ == "__main__":
    main()
