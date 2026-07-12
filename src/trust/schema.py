"""
Typed data contracts for the TALMedora Evidence Engine.

Stdlib dataclasses only -- no new dependency. Every object is JSON-serialisable
via `to_dict` so reports and AI-ready records can be persisted, shipped to a UI,
or written to a governed store without bespoke encoders.

The pipeline produces these objects in order:
    Post  -> [Claim]  -> [ClaimVerdict]  -> PostTrustReport  -> AIReadyRecord
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum


class Verdict(str, Enum):
    """How a single claim fared against the retrieved evidence."""

    SUPPORTED = "supported"          # evidence in the corpus backs the claim
    CONTRADICTED = "contradicted"    # evidence in the corpus opposes the claim  <- misinformation signal
    INSUFFICIENT = "insufficient"    # corpus has nothing that settles it either way
    NOT_VERIFIABLE = "not_verifiable"  # opinion / experience / non-empirical -> not penalised as false


# Severity weights drive the TrustScore: a contradicted claim hurts far more than
# one we simply couldn't verify. Kept here as the single source of truth so the
# scoring rubric stays transparent and explainable (no hidden magic numbers).
VERDICT_WEIGHT = {
    Verdict.SUPPORTED: 1.0,
    Verdict.CONTRADICTED: 0.0,
    Verdict.INSUFFICIENT: 0.5,
    Verdict.NOT_VERIFIABLE: None,   # excluded from the score (neither rewarded nor punished)
}


class ClaimType(str, Enum):
    """What kind of assertion a claim is -- decides whether we spend a grading call on it."""

    EMPIRICAL = "empirical"      # checkable efficacy/safety/risk/guideline claim -> grade vs literature
    OPINION = "opinion"          # experience / value judgement / self-care -> NOT_VERIFIABLE, not graded as false
    NON_MEDICAL = "non_medical"  # no clinical content (chit-chat, spam, injected commands) -> excluded


@dataclass(frozen=True)
class Citation:
    """A single supporting/contradicting source, traceable back to PubMed by PMID."""

    pmid: str
    title: str
    year: str
    score: float            # retriever similarity (relative rank within the corpus)

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"


@dataclass(frozen=True)
class Claim:
    """One discrete, atomic assertion lifted out of a post."""

    id: int
    text: str
    claim_type: ClaimType = ClaimType.EMPIRICAL

    @property
    def is_checkable(self) -> bool:
        """Only empirical claims earn a (paid) evidence-grading call."""
        return self.claim_type is ClaimType.EMPIRICAL


@dataclass
class ClaimVerdict:
    """The graded result for one claim, with the evidence behind it."""

    claim: Claim
    verdict: Verdict
    confidence: float                       # 0..1, the grader's certainty
    rationale: str                          # short, must reference the cited passages
    citations: list[Citation] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        d["claim"]["claim_type"] = self.claim.claim_type.value
        return d


@dataclass
class PostTrustReport:
    """The post-level verdict surfaced in the UI: a score, a label, and the why."""

    post_id: str
    trust_score: int | None                 # 0..100, or None when there are no checkable claims
    label: str                              # "Well-supported" | "Mixed" | "Unsupported" | "Unscored"
    claim_verdicts: list[ClaimVerdict]
    flags: list[str] = field(default_factory=list)   # e.g. "contradicted_claim", "prompt_injection_detected"
    model: str = ""
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "post_id": self.post_id,
            "trust_score": self.trust_score,
            "label": self.label,
            "flags": self.flags,
            "model": self.model,
            "cost_usd": round(self.cost_usd, 6),
            "claim_verdicts": [cv.to_dict() for cv in self.claim_verdicts],
        }


@dataclass
class AIReadyRecord:
    """
    The governance payload -- a verified post repackaged for safe LLM training.

    This is the bridge to Solix's data-management business: content is useless to
    an LLM unless it carries provenance (who said it, are they licensed), consent
    (did they agree to training use), jurisdiction (which data law applies), and a
    trust signal (is it actually correct). We attach all four.
    """

    record_id: str
    text: str
    trust_score: int | None
    verdict_summary: dict                   # {"supported": 2, "contradicted": 0, ...}
    # --- provenance ---
    author_specialty: str
    author_licensed: bool
    # --- consent / governance ---
    consent_to_train: bool
    jurisdiction: str                       # e.g. "US", "IN", "EU"
    deidentified: bool
    # --- training gate + distilled payload ---
    training_eligible: bool = False
    eligibility_reasons: list[str] = field(default_factory=list)
    verified_claims: list[str] = field(default_factory=list)   # the SUPPORTED claim texts -- the clean nuggets
    citations: list[Citation] = field(default_factory=list)

    @staticmethod
    def make_id(text: str) -> str:
        """Stable, content-derived id (no PII) for dedup + traceability."""
        return "rec_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["citations"] = [asdict(c) for c in self.citations]
        return d
