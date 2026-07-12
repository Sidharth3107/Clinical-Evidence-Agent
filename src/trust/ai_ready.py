"""
Step 4 of the Evidence Engine: emit the governed, training-ready record.

This is the bridge to a data-management business: a social post is worthless to
an LLM team until it carries provenance (who, are they licensed), consent (may we
train on it), jurisdiction (which data law applies), a trust signal (is it
correct), and de-identification (is it safe). We attach all five and gate the
record so only clean, consented, verified content is marked training-eligible.

Pure transformation -- no API calls, $0. The expensive thinking already happened
in Steps 1-3; here we just package and govern the result.

De-identification note: the scrub below is a transparent, best-effort regex pass
for obvious identifiers. A production deployment would route text through a
dedicated medical de-id service (e.g. AWS Comprehend Medical or Microsoft Presidio)
-- this function is the seam where that would plug in.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import AIReadyRecord, Citation, PostTrustReport, Verdict  # noqa: E402
from score import verdict_summary                                      # noqa: E402

# A post must clear all of these to be marked training-eligible.
MIN_TRUST_FOR_TRAINING = 50

# Best-effort PHI patterns (obvious identifiers only; see module docstring).
_PHI_PATTERNS = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    (re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), "[DATE]"),
    (re.compile(r"\b\d{1,3}[\s-]?(?:year|yr|y/?o)s?[\s-]?old\b", re.I), "[AGE]"),
    (re.compile(r"\b\d{7,}\b"), "[ID]"),     # MRN-like long digit runs (years are 4 digits -> safe)
]


def deidentify(text: str) -> tuple[str, int]:
    """Redact obvious identifiers. Returns (scrubbed_text, redaction_count)."""
    count = 0
    for pat, repl in _PHI_PATTERNS:
        text, n = pat.subn(repl, text)
        count += n
    return text, count


def _eligibility(report: PostTrustReport, meta: dict) -> tuple[bool, list[str]]:
    """Transparent training-eligibility gate. Every failure carries a reason."""
    reasons: list[str] = []
    if not meta.get("consent_to_train", False):
        reasons.append("no_consent")
    if not meta.get("author_licensed", False):
        reasons.append("author_not_licensed")
    if "prompt_injection_detected" in report.flags:
        reasons.append("prompt_injection")
    if "contradicted_claim" in report.flags:
        reasons.append("contains_contradicted_claim")
    if report.trust_score is None:
        reasons.append("no_verifiable_claims")
    elif report.trust_score < MIN_TRUST_FOR_TRAINING:
        reasons.append("trust_below_threshold")
    return (not reasons), reasons


def to_ai_ready_record(report: PostTrustReport, meta: dict,
                       *, deidentify_text: bool = True) -> AIReadyRecord:
    """Package a scored post into a governed, training-ready record."""
    raw = meta.get("text", "")
    text = deidentify(raw)[0] if deidentify_text else raw

    eligible, reasons = _eligibility(report, meta)

    # distil: keep only the SUPPORTED claims and their (deduped) citations
    supported = [cv for cv in report.claim_verdicts if cv.verdict is Verdict.SUPPORTED]
    verified_claims = [cv.claim.text for cv in supported]
    seen: set[str] = set()
    citations: list[Citation] = []
    for cv in supported:
        for c in cv.citations:
            if c.pmid not in seen:
                seen.add(c.pmid)
                citations.append(c)

    return AIReadyRecord(
        record_id=AIReadyRecord.make_id(raw),
        text=text,
        trust_score=report.trust_score,
        verdict_summary=verdict_summary(report.claim_verdicts),
        author_specialty=meta.get("author_specialty", ""),
        author_licensed=bool(meta.get("author_licensed", False)),
        consent_to_train=bool(meta.get("consent_to_train", False)),
        jurisdiction=meta.get("jurisdiction", ""),
        deidentified=deidentify_text,
        training_eligible=eligible,
        eligibility_reasons=reasons,
        verified_claims=verified_claims,
        citations=citations,
    )


if __name__ == "__main__":
    import json

    from schema import Claim, ClaimType, ClaimVerdict
    from score import score_post

    # de-id demo
    demo = "Reach me at jane.doe@clinic.com or 555-123-4567 re: my 72-year-old patient, MRN 12345678, seen 03/14/2025."
    print("de-id:", deidentify(demo)[0], "\n")

    # $0 verification: synthetic verdicts (mirroring real Step-2 output) + real metadata.
    posts = {p["post_id"]: p for p in
             json.loads((ROOT / "data" / "sample_posts.json").read_text(encoding="utf-8"))}
    V = Verdict

    def cvd(text, verdict, conf, *cites):
        cites = [Citation(p, "title", "2021", 0.9) for p in cites]
        return ClaimVerdict(Claim(0, text, ClaimType.EMPIRICAL), verdict, conf, "", cites)

    scenarios = {
        "post_001": (False, [cvd("SGLT2i reduce CV death in HFrEF", V.SUPPORTED, .95, "32877652"),
                             cvd("Beta-blockers titrated to target in HFrEF", V.SUPPORTED, .95, "40834370"),
                             cvd("Beta-blockers reduce mortality in HFrEF", V.INSUFFICIENT, .95)]),
        "post_002": (False, [cvd("Beta-blockers avoided in all HF", V.CONTRADICTED, .99, "22582541")]),
        "post_003": (True,  [cvd("SGLT2i cure HF in one dose", V.CONTRADICTED, .99, "40968746")]),
        "post_004": (False, [cvd("breathing exercises feel calmer", V.NOT_VERIFIABLE, 1.0),
                             cvd("continue prescribed medications", V.SUPPORTED, .95, "36740210")]),
    }

    print(f"{'post':9} {'score':>5} {'eligible':>9}  reasons / payload")
    for pid, (inj, vs) in scenarios.items():
        report = score_post(pid, vs, contains_injection=inj)
        rec = to_ai_ready_record(report, posts[pid])
        tag = "YES" if rec.training_eligible else "no"
        detail = (f"{len(rec.verified_claims)} verified claim(s), {len(rec.citations)} cite(s)"
                  if rec.training_eligible else ", ".join(rec.eligibility_reasons))
        print(f"{pid:9} {str(rec.trust_score):>5} {tag:>9}  {detail}")

    # rules must hold
    r1 = to_ai_ready_record(score_post("p1", scenarios["post_001"][1]), posts["post_001"])
    assert r1.training_eligible and r1.verified_claims and r1.citations, r1.to_dict()
    r3 = to_ai_ready_record(score_post("p3", scenarios["post_003"][1], contains_injection=True), posts["post_003"])
    assert not r3.training_eligible and "prompt_injection" in r3.eligibility_reasons, r3.to_dict()
    json.dumps(r1.to_dict())   # must serialise
    print("\nall AI-ready gate assertions passed.")
