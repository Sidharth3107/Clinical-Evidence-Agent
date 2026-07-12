"""
Step 3 of the Evidence Engine: roll per-claim verdicts into a post-level TrustScore.

Pure aggregation -- no API calls, $0. The scoring rule is deliberately transparent
so the number is defensible to a clinician or a CEO:

    TrustScore = 100 * (sum of  verdict_weight * confidence  over scored claims)
                       / (sum of confidence over scored claims)

  * weights come from the visible VERDICT_WEIGHT table (supported=1, insufficient=0.5,
    contradicted=0); NOT_VERIFIABLE (opinion) claims are excluded entirely.
  * confidence-weighting means a high-certainty contradiction hurts far more than a
    hesitant one.
  * a single high-confidence contradiction caps the label at "Mixed" -- a medical
    post can't get a clean bill of health while it contains a refuted claim.

This module also exposes analyze_post(): the one-call extract -> grade -> score
pipeline used by the demo UI.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import (  # noqa: E402
    VERDICT_WEIGHT, ClaimVerdict, PostTrustReport, Verdict,
)

# label thresholds (transparent, single source of truth)
WELL_SUPPORTED = 80
MIXED = 50


def verdict_summary(verdicts: list[ClaimVerdict]) -> dict[str, int]:
    """Count claims by verdict -- feeds the UI badges and the AI-ready record."""
    out = {v.value: 0 for v in Verdict}
    for cv in verdicts:
        out[cv.verdict.value] += 1
    return out


def _label(score: int | None, has_contradiction: bool) -> str:
    if score is None:
        return "Unscored"
    if has_contradiction and score >= WELL_SUPPORTED:
        return "Mixed"                      # a contradiction blocks a clean bill of health
    if score >= WELL_SUPPORTED:
        return "Well-supported"
    if score >= MIXED:
        return "Mixed"
    return "Unsupported"


def score_post(post_id: str, verdicts: list[ClaimVerdict], *,
               contains_injection: bool = False, model: str = "",
               cost_usd: float = 0.0) -> PostTrustReport:
    """Aggregate verdicts into a PostTrustReport. No API calls."""
    scored = [(VERDICT_WEIGHT[cv.verdict], cv.confidence)
              for cv in verdicts if VERDICT_WEIGHT[cv.verdict] is not None]

    if scored:
        denom = sum(c for _, c in scored) or 1.0
        score: int | None = round(100 * sum(w * c for w, c in scored) / denom)
    else:
        score = None                        # nothing checkable -> honestly unscored

    has_contradiction = any(cv.verdict is Verdict.CONTRADICTED for cv in verdicts)

    flags: list[str] = []
    if contains_injection:
        flags.append("prompt_injection_detected")
    if has_contradiction:
        flags.append("contradicted_claim")
    if any(cv.verdict is Verdict.INSUFFICIENT for cv in verdicts):
        flags.append("unverified_claims")
    if score is None:
        flags.append("no_checkable_claims")

    return PostTrustReport(
        post_id=post_id, trust_score=score, label=_label(score, has_contradiction),
        claim_verdicts=verdicts, flags=flags, model=model, cost_usd=cost_usd)


def analyze_post(post_text: str, *, post_id: str = "", k: int = 5,
                 extract_model: str | None = None,
                 grade_model: str | None = None) -> PostTrustReport:
    """End-to-end: extract -> grade -> score. The single entry point for the UI."""
    from extract_claims import MODEL as EXTRACT_MODEL, extract_claims
    from grade_claim import MODEL as GRADE_MODEL, grade_claims
    from schema import AIReadyRecord

    ext = extract_claims(post_text, model=extract_model or EXTRACT_MODEL)
    verdicts, gcost = grade_claims(ext.claims, k=k, model=grade_model or GRADE_MODEL)
    return score_post(
        post_id or AIReadyRecord.make_id(post_text),
        verdicts, contains_injection=ext.contains_injection,
        model=grade_model or GRADE_MODEL, cost_usd=ext.cost_usd + gcost)


if __name__ == "__main__":
    # $0 verification: build synthetic verdicts mirroring the real Step-2 outputs and
    # check the scoring/label/flag logic deterministically -- no API calls.
    from schema import Claim, ClaimType

    def cv(verdict: Verdict, conf: float, ct: ClaimType = ClaimType.EMPIRICAL) -> ClaimVerdict:
        return ClaimVerdict(Claim(0, "x", ct), verdict, conf, "")

    V = Verdict
    cases = {
        "post_001 (supported)":   (False, [cv(V.SUPPORTED, .95), cv(V.SUPPORTED, .95),
                                           cv(V.SUPPORTED, .99), cv(V.INSUFFICIENT, .95),
                                           cv(V.SUPPORTED, .95)]),
        "post_002 (misinfo)":     (False, [cv(V.CONTRADICTED, .99), cv(V.CONTRADICTED, .95),
                                           cv(V.CONTRADICTED, .95)]),
        "post_003 (injection)":   (True,  [cv(V.CONTRADICTED, .99)]),
        "post_004 (opinion+1)":   (False, [cv(V.NOT_VERIFIABLE, 1.0, ClaimType.OPINION),
                                           cv(V.SUPPORTED, .95)]),
        "all-opinion":            (False, [cv(V.NOT_VERIFIABLE, 1.0, ClaimType.OPINION),
                                           cv(V.NOT_VERIFIABLE, 1.0, ClaimType.OPINION)]),
        "9 supported + 1 contra": (False, [cv(V.SUPPORTED, .95) for _ in range(9)] + [cv(V.CONTRADICTED, .95)]),
    }

    print(f"{'case':26} {'score':>6}  {'label':16} flags")
    for name, (inj, vs) in cases.items():
        r = score_post(name, vs, contains_injection=inj)
        print(f"{name:26} {str(r.trust_score):>6}  {r.label:16} {r.flags}")

    # assertions: the rules must hold
    r2 = score_post("a", cases["post_002 (misinfo)"][1])
    assert r2.trust_score == 0 and r2.label == "Unsupported", r2
    r5 = score_post("a", cases["all-opinion"][1])
    assert r5.trust_score is None and r5.label == "Unscored", r5
    r6 = score_post("a", cases["9 supported + 1 contra"][1])
    assert r6.trust_score >= 80 and r6.label == "Mixed", r6   # contradiction caps the label
    print("\nall scoring assertions passed.")
