"""
Agent evaluation: tool-selection accuracy, deferral, citation accuracy, cost + latency.

Runs each scenario through the real (instrumented) agent loop and scores it:
- tool-selection: did it call the expected tool(s)? (deferral scenarios expect none)
- citation accuracy: were the PMIDs it cited actually in what it retrieved?
- cost + latency per query.
Uses the API (Haiku) -- a few cents total. Loads the key from .env via agent.py.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "agent"))
import agent  # noqa: E402

SCEN = ROOT / "eval" / "agent_scenarios.json"
PMID_RE = re.compile(r"PMID:?\s*(\d+)")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    scenarios = json.loads(SCEN.read_text(encoding="utf-8"))
    print(f"running {len(scenarios)} scenarios on {agent.MODEL}\n")

    rows = []
    for s in scenarios:
        try:
            t0 = time.perf_counter()
            tr = agent.run_agent(s["query"], verbose=False, return_trace=True)
            dt = time.perf_counter() - t0
        except Exception as e:
            print(f"  ERROR {s['id']}: {e}")
            continue

        called = sorted(set(tr["tools_called"]))
        expected = set(s["expected_tools"])
        sel_ok = expected.issubset(set(called)) if expected else (len(called) == 0)
        cited = set(PMID_RE.findall(tr["answer"]))
        cite_acc = (len(cited & tr["retrieved_pmids"]) / len(cited)) if cited else None

        rows.append({"id": s["id"], "category": s["category"], "sel_ok": sel_ok,
                     "called": called, "expected": sorted(expected),
                     "n_cited": len(cited), "cite_acc": cite_acc,
                     "cost": round(tr["cost"], 4), "latency": round(dt, 1)})

        mark = "OK" if sel_ok else "XX"
        ca = "-" if cite_acc is None else f"{cite_acc:.2f}"
        print(f"  [{mark}] {s['id']:15s} {s['category']:13s} called={called} "
              f"cite_acc={ca} {dt:4.1f}s ${tr['cost']:.4f}")

    n = len(rows)
    sel_acc = sum(r["sel_ok"] for r in rows) / n
    defer = [r for r in rows if not r["expected"]]
    defer_rate = (sum(r["sel_ok"] for r in defer) / len(defer)) if defer else None
    cites = [r for r in rows if r["cite_acc"] is not None]
    mean_cite = (sum(r["cite_acc"] for r in cites) / len(cites)) if cites else None
    n_ungrounded = sum(1 for r in cites if r["cite_acc"] < 1.0)
    total_cost = sum(r["cost"] for r in rows)
    mean_lat = sum(r["latency"] for r in rows) / n

    print("\n=== agent eval summary ===")
    print(f"  tool-selection accuracy : {sel_acc:.2f}  ({sum(r['sel_ok'] for r in rows)}/{n})")
    if defer_rate is not None:
        print(f"  appropriate-deferral    : {defer_rate:.2f}  ({sum(r['sel_ok'] for r in defer)}/{len(defer)})")
    if mean_cite is not None:
        print(f"  citation accuracy       : {mean_cite:.2f}  over {len(cites)} cited answers "
              f"({n_ungrounded} with an ungrounded citation)")
    print(f"  total cost              : ${total_cost:.4f}")
    print(f"  mean latency            : {mean_lat:.1f}s")

    summary = {
        "model": agent.MODEL, "n": n,
        "tool_selection_accuracy": round(sel_acc, 3),
        "deferral_rate": round(defer_rate, 3) if defer_rate is not None else None,
        "citation_accuracy": round(mean_cite, 3) if mean_cite is not None else None,
        "ungrounded_citation_answers": n_ungrounded,
        "total_cost_usd": round(total_cost, 4),
        "mean_latency_s": round(mean_lat, 1),
        "per_scenario": rows,
    }
    (ROOT / "eval" / "agent_metrics.json").write_text(json.dumps(summary, indent=2))
    print("\nsaved -> eval/agent_metrics.json")


if __name__ == "__main__":
    main()
