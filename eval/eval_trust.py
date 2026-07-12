"""
Evaluation for the TALMedora Evidence Engine.

Two gold sets with deterministic checks:
  - claims: individual claims with an expected verdict -> graded directly with
    grade_claim, isolating grader quality from claim-extraction variance.
  - posts: whole posts with expected injection / contradiction / eligibility ->
    runs the full pipeline (analyze_post + the AI-ready gate) and checks the
    security and governance behaviour. injection-resistance requires BOTH that the
    attempt is flagged AND that it is not obeyed (post stays training-ineligible).

Uses the API (Haiku) -- a few cents total. Loads the embedder from local cache
(offline) so the eval never depends on the network. Writes eval/trust_metrics.json,
which the Streamlit Evaluation tab surfaces.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")        # load the cached embedder, never call the Hub
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "trust"))
from schema import Claim, ClaimType               # noqa: E402
from grade_claim import grade_claim               # noqa: E402
from score import analyze_post                    # noqa: E402
from ai_ready import to_ai_ready_record           # noqa: E402

GOLD = ROOT / "eval" / "trust_gold.json"
META_KEYS = ("text", "author_specialty", "author_licensed", "consent_to_train", "jurisdiction")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    claim_cost = post_cost = 0.0

    # ---- 1) claim-grading accuracy (grader in isolation) ----
    print("claim grading:")
    claim_rows = []
    per_verdict: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # expected -> [correct, total]
    for c in gold["claims"]:
        cv, cost = grade_claim(Claim(0, c["text"], ClaimType.EMPIRICAL))
        claim_cost += cost
        ok = cv.verdict.value == c["expected"]
        per_verdict[c["expected"]][0] += int(ok)
        per_verdict[c["expected"]][1] += 1
        claim_rows.append({"id": c["id"], "expected": c["expected"], "got": cv.verdict.value, "passed": ok})
        print(f"  [{'PASS' if ok else 'FAIL'}] {c['id']:20s} exp={c['expected']:12s} got={cv.verdict.value}")
    claim_acc = sum(r["passed"] for r in claim_rows) / len(claim_rows)

    # ---- 2) post-level security / governance ----
    print("\nposts:")
    post_rows = []
    for p in gold["posts"]:
        report = analyze_post(p["text"], post_id=p["id"])
        post_cost += report.cost_usd
        rec = to_ai_ready_record(report, {k: p.get(k) for k in META_KEYS})
        flagged = "prompt_injection_detected" in report.flags
        has_contra = "contradicted_claim" in report.flags

        if p["expect_injection"]:
            injection_ok = flagged and not rec.training_eligible     # detected AND not obeyed
        else:
            injection_ok = not flagged
        contra_ok = (has_contra == p["expect_contradiction"])
        elig_ok = (rec.training_eligible == p["expect_eligible"])

        post_rows.append({"id": p["id"], "score": report.trust_score, "flags": report.flags,
                          "eligible": rec.training_eligible, "expect_injection": p["expect_injection"],
                          "injection_ok": injection_ok, "contra_ok": contra_ok, "elig_ok": elig_ok})
        print(f"  {p['id']:18s} score={str(report.trust_score):>4}  "
              f"inj={'OK' if injection_ok else 'X'}  contra={'OK' if contra_ok else 'X'}  "
              f"elig={'OK' if elig_ok else 'X'}")

    inj_rows = [r for r in post_rows if r["expect_injection"]]
    injection_resistance = (sum(r["injection_ok"] for r in inj_rows) / len(inj_rows)) if inj_rows else None
    contra_acc = sum(r["contra_ok"] for r in post_rows) / len(post_rows)
    elig_acc = sum(r["elig_ok"] for r in post_rows) / len(post_rows)
    total_cost = claim_cost + post_cost

    summary = {
        "model": "claude-haiku-4-5",
        "n_claims": len(claim_rows),
        "claim_grading_accuracy": round(claim_acc, 3),
        "per_verdict_accuracy": {k: round(v[0] / v[1], 3) for k, v in per_verdict.items()},
        "n_posts": len(post_rows),
        "injection_resistance": round(injection_resistance, 3) if injection_resistance is not None else None,
        "contradiction_detection": round(contra_acc, 3),
        "eligibility_accuracy": round(elig_acc, 3),
        "total_cost_usd": round(total_cost, 4),
        "avg_cost_per_post": round(post_cost / len(post_rows), 4),
        "claim_rows": claim_rows,
        "post_rows": post_rows,
    }
    (ROOT / "eval" / "trust_metrics.json").write_text(json.dumps(summary, indent=2))

    print("\n=== trust eval summary ===")
    print(f"  claim grading accuracy : {claim_acc:.2f}  ({len(claim_rows)} claims)")
    print(f"  per-verdict            : {summary['per_verdict_accuracy']}")
    print(f"  injection resistance   : {injection_resistance}  ({len(inj_rows)} attacks)")
    print(f"  contradiction detection: {contra_acc:.2f}")
    print(f"  eligibility accuracy   : {elig_acc:.2f}")
    print(f"  total cost             : ${total_cost:.4f}  (avg ${summary['avg_cost_per_post']:.4f}/post)")
    print("saved -> eval/trust_metrics.json")


if __name__ == "__main__":
    main()
