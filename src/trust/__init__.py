"""
TALMedora Evidence Engine ("TrustScore").

A content-trust layer for a licensed-professionals-only medical platform: it takes
a free-text clinical post, splits it into discrete checkable claims, grounds each
claim in the existing PubMed/guideline retriever, grades it (supported /
contradicted / insufficient), rolls the verdicts into a transparent post-level
TrustScore, and emits a governed, provenance- and consent-tagged "AI-ready"
record for downstream LLM training.

Design principles (inherited from the Clinical Evidence Agent):
  - Reuse the local, $0 FAISS retriever for evidence; no new infrastructure.
  - The post is UNTRUSTED input and is treated as a prompt-injection boundary.
  - Synthetic data only; never real PHI.
  - Cheap models + caching + capped tokens; cost is metered, not assumed.
"""
