"""
Ingest heart-failure literature from PubMed (NCBI E-utilities) into a local corpus.

Pipeline:  esearch (topic -> PMIDs) -> dedupe -> efetch (PMIDs -> records) ->
parse XML -> write data/hf_corpus.jsonl (one JSON record per line).

Each record carries its PMID -- the citation anchor that flows through the whole
RAG pipeline. Core pipeline is standard-library only; it optionally loads a .env
via python-dotenv (if installed) to read NCBI credentials.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# Optionally load project-root .env so NCBI_EMAIL / NCBI_API_KEY are available when
# run standalone; degrades to stdlib-only if python-dotenv isn't installed.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = "cardio-evidence-agent"
EMAIL = os.environ.get("NCBI_EMAIL", "")   # NCBI etiquette: identify the caller (set in .env)
API_KEY = os.environ.get("NCBI_API_KEY")   # optional NCBI key -> 10 req/s instead of 3
PER_TOPIC = 40                        # PMIDs requested per topic
REQUEST_PAUSE = 0.34                  # ~3 req/s without an API key
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "hf_corpus.jsonl"

# topic label -> PubMed search phrase. The last three deliberately mirror the
# risk model's top features (ejection fraction, creatinine, sodium) so the agent
# can cite literature about the exact factors the model flags.
TOPICS = {
    "HFrEF GDMT": "heart failure reduced ejection fraction guideline directed medical therapy",
    "HFpEF": "heart failure preserved ejection fraction treatment",
    "Beta-blockers": "beta blockers heart failure mortality",
    "ACEi / ARNI": "sacubitril valsartan ARNI heart failure",
    "SGLT2 inhibitors": "SGLT2 inhibitors heart failure outcomes",
    "MRA": "mineralocorticoid receptor antagonist spironolactone heart failure",
    "Device therapy": "implantable cardioverter defibrillator cardiac resynchronization heart failure",
    "Ejection fraction prognosis": "ejection fraction prognosis heart failure mortality",
    "Renal function": "serum creatinine renal function heart failure prognosis",
    "Serum sodium": "hyponatremia serum sodium heart failure prognosis",
    "Mortality risk prediction": "heart failure mortality risk prediction model",
}
FILTER = "hasabstract[text] AND English[lang]"


def _fetch(url: str, retries: int = 2) -> bytes:
    """GET with a couple of retries for transient NCBI hiccups."""
    req = urllib.request.Request(url, headers={"User-Agent": TOOL})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(1.5 * (attempt + 1))


def _params(**kw) -> str:
    kw.update(tool=TOOL, email=EMAIL)
    if API_KEY:
        kw["api_key"] = API_KEY
    return urllib.parse.urlencode(kw)


def esearch(term: str, retmax: int) -> list[str]:
    url = f"{EUTILS}/esearch.fcgi?" + _params(
        db="pubmed",
        term=f"({term}) AND {FILTER}",
        retmax=retmax,
        retmode="json",
        sort="relevance",
    )
    data = json.loads(_fetch(url))
    return data.get("esearchresult", {}).get("idlist", [])


def efetch(pmids: list[str]) -> bytes:
    url = f"{EUTILS}/efetch.fcgi?" + _params(
        db="pubmed",
        id=",".join(pmids),
        retmode="xml",
        rettype="abstract",
    )
    return _fetch(url)


def _text(el) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def parse_articles(xml_bytes: bytes) -> dict[str, dict]:
    """Extract pmid/title/journal/year/abstract/pub_types from an efetch XML."""
    root = ET.fromstring(xml_bytes)
    out: dict[str, dict] = {}
    for art in root.findall(".//PubmedArticle"):
        cit = art.find(".//MedlineCitation")
        if cit is None:
            continue
        pmid = _text(cit.find("PMID"))
        article = cit.find("Article")
        if not pmid or article is None:
            continue

        title = _text(article.find("ArticleTitle"))
        journal = _text(article.find(".//Journal/Title"))
        year = _text(article.find(".//JournalIssue/PubDate/Year")) or \
            _text(article.find(".//JournalIssue/PubDate/MedlineDate"))[:4]

        # abstracts can be split into multiple labelled sections
        parts = []
        for ab in article.findall(".//Abstract/AbstractText"):
            label = ab.get("Label")
            txt = _text(ab)
            parts.append(f"{label}: {txt}" if label else txt)
        abstract = "\n".join(p for p in parts if p)

        pub_types = [_text(pt) for pt in article.findall(".//PublicationType")]

        if abstract:  # skip anything that slipped through without abstract text
            out[pmid] = {
                "pmid": pmid,
                "title": title,
                "journal": journal,
                "year": year,
                "abstract": abstract,
                "pub_types": pub_types,
            }
    return out


def main() -> None:
    # 1) esearch each topic; collect unique PMIDs, remembering which topics hit
    pmid_topics: dict[str, list[str]] = {}
    print("esearch:")
    for label, term in TOPICS.items():
        ids = esearch(term, PER_TOPIC)
        for pid in ids:
            pmid_topics.setdefault(pid, []).append(label)
        print(f"  {label:30s} -> {len(ids):3d} PMIDs")
        time.sleep(REQUEST_PAUSE)

    pmids = list(pmid_topics)
    print(f"\n{len(pmids)} unique PMIDs across {len(TOPICS)} topics\n")

    # 2) efetch in batches of 200; parse; merge
    records: dict[str, dict] = {}
    print("efetch:")
    for i in range(0, len(pmids), 200):
        batch = pmids[i:i + 200]
        parsed = parse_articles(efetch(batch))
        records.update(parsed)
        print(f"  batch {i // 200 + 1}: +{len(parsed)} parsed (total {len(records)})")
        time.sleep(REQUEST_PAUSE)

    # 3) attach topic tags and write JSONL
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for pmid, rec in records.items():
            rec["topics"] = pmid_topics.get(pmid, [])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} abstracts -> {OUT_PATH}")
    print(f"Every record has a PMID + abstract text.")


if __name__ == "__main__":
    main()
