"""
Streamlit demo for the Clinical Evidence Agent (heart failure).

Three tabs:
  - Ask        : run the agent on a clinical question (optionally with synthetic
                 patient data); shows the cited answer, the tools it called, and
                 tokens / cost / latency, plus the sources behind each citation.
  - Evidence   : query the literature retriever directly and inspect ranked,
                 cited sources. Local, $0.
  - Evaluation : the committed eval results (ML, agent, safety) + figures.

Cost / security by design:
  - Heavy resources (embedding model, FAISS index, risk model) load ONCE via
    st.cache_resource -- not on every Streamlit rerun.
  - The agent runs only on an explicit button click, defaults to the cheapest
    model (Haiku), and the per-query token cost is shown.
  - The Evidence and Evaluation tabs need no API key and cost nothing.
  - Synthetic data only -- the UI must never receive real PHI.
  - The question box is length-capped (bounds token cost); the API key is read
    from .env and never displayed.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC / "agent"))
sys.path.insert(0, str(SRC / "tools"))

import agent                                       # noqa: E402  (loads .env, wires tools)
from search_literature import search_literature    # noqa: E402

EVAL = ROOT / "eval"
CORPUS = ROOT / "data" / "hf_corpus.jsonl"
PMID_RE = re.compile(r"PMID:?\s*(\d+)")
MAX_QUERY_CHARS = 1000
MODELS = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8"]

# A synthetic example patient (NOT a real person) -- used to pre-fill the inputs.
EXAMPLE_PATIENT = {
    "age": 72, "anaemia": 0, "creatinine_phosphokinase": 180, "diabetes": 1,
    "ejection_fraction": 25, "high_blood_pressure": 1, "platelets": 210000,
    "serum_creatinine": 2.1, "serum_sodium": 131, "sex": 1, "smoking": 0,
}
CONT = [  # (key, label, min, max, step)
    ("age", "Age (yrs)", 18.0, 110.0, 1.0),
    ("ejection_fraction", "Ejection fraction (%)", 5.0, 80.0, 1.0),
    ("serum_creatinine", "Serum creatinine (mg/dL)", 0.1, 15.0, 0.1),
    ("serum_sodium", "Serum sodium (mEq/L)", 110.0, 160.0, 1.0),
    ("creatinine_phosphokinase", "CPK (mcg/L)", 10.0, 8000.0, 10.0),
    ("platelets", "Platelets (/mL)", 25000.0, 500000.0, 1000.0),
]
BIN = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]

st.set_page_config(page_title="Clinical Evidence Agent — Heart Failure",
                   page_icon="🫀", layout="wide")


# ---- cached resources (load once per session) ------------------------------
@st.cache_resource(show_spinner="Loading literature corpus…")
def corpus_by_pmid() -> dict:
    out: dict[str, dict] = {}
    if CORPUS.exists():
        for line in CORPUS.open(encoding="utf-8"):
            r = json.loads(line)
            out[r["pmid"]] = r
    return out


@st.cache_resource(show_spinner="Warming up the retriever…")
def warm_retriever():
    from search_literature import get_retriever
    return get_retriever()


@st.cache_data
def load_metrics(name: str):
    p = EVAL / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def source_card(pmid: str) -> None:
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    rec = corpus_by_pmid().get(pmid)
    if not rec:
        st.markdown(f"- **PMID [{pmid}]({url})** — not in local corpus")
        return
    st.markdown(f"**[{rec['title']}]({url})**  \n"
                f"*{rec.get('journal', '')}* ({rec.get('year', '')}) · PMID {pmid}")
    abstract = rec.get("abstract", "")
    if abstract:
        st.caption(abstract[:380] + ("…" if len(abstract) > 380 else ""))


# ---- header / disclaimer ---------------------------------------------------
st.title("🫀 Clinical Evidence Agent — Heart Failure")
st.caption("Evidence-grounded **decision support** for clinicians & researchers — "
           "not a patient-facing or diagnostic tool.")
st.warning("Demo runs on **synthetic data only**. Do not enter real patient "
           "information (PHI).", icon="⚠️")

ask_tab, evidence_tab, eval_tab = st.tabs(["💬 Ask", "📚 Evidence", "📊 Evaluation"])

# ===========================================================================
# Ask
# ===========================================================================
with ask_tab:
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not has_key:
        st.info("No `ANTHROPIC_API_KEY` in `.env`, so the agent is disabled here. "
                "The **Evidence** and **Evaluation** tabs work without a key (and cost nothing).")

    model = st.selectbox("Model", MODELS, index=0,
                         help="Haiku is cheapest; the actual cost per query is shown after each run.")

    examples = {
        "Evidence": "What does the evidence say about SGLT2 inhibitors and mortality in heart failure?",
        "Guideline": "What do the guidelines recommend as first-line therapy for HFrEF?",
        "Out-of-scope": "What's a good recipe for dinner tonight?",
    }
    if "query" not in st.session_state:
        st.session_state["query"] = examples["Evidence"]
    st.write("**Try an example:**")
    for col, (label, text) in zip(st.columns(len(examples)), examples.items()):
        if col.button(label, use_container_width=True):
            st.session_state["query"] = text

    query = st.text_area("Clinical question", key="query", height=100,
                         max_chars=MAX_QUERY_CHARS)

    with st.expander("➕ Add synthetic patient data (triggers the calibrated risk model)"):
        st.caption("Pre-filled with a synthetic example — never enter real patient data.")
        include = st.checkbox("Include this patient's data in the question")
        pv: dict[str, float] = {}
        cols = st.columns(3)
        for i, (k, label, lo, hi, step) in enumerate(CONT):
            pv[k] = cols[i % 3].number_input(label, min_value=lo, max_value=hi,
                                             value=float(EXAMPLE_PATIENT[k]), step=step)
        bcols = st.columns(len(BIN))
        for col, k in zip(bcols, BIN):
            pv[k] = col.selectbox(k, [0, 1], index=int(EXAMPLE_PATIENT[k]))

    run = st.button("Run agent", type="primary", disabled=not has_key)
    if run and query.strip():
        full_query = query.strip()
        if include:
            feats = ", ".join(f"{k}={pv[k]:g}" for k in EXAMPLE_PATIENT)
            full_query += f"\n\nPatient (synthetic): {feats}"
        try:
            with st.spinner(f"Running {model}…"):
                t0 = time.perf_counter()
                tr = agent.run_agent(full_query, model=model, verbose=False, return_trace=True)
                dt = time.perf_counter() - t0
        except Exception as e:  # never surface a raw traceback/secret to the UI
            st.error(f"Agent error: {type(e).__name__}: {e}")
        else:
            st.markdown("### Answer")
            st.markdown(tr["answer"])
            c = st.columns(4)
            c[0].metric("Tools called", ", ".join(tr["tools_called"]) or "none")
            c[1].metric("Tokens (in/out)", f'{tr["tokens_in"]}/{tr["tokens_out"]}')
            c[2].metric("Cost", f'${tr["cost"]:.4f}')
            c[3].metric("Latency", f"{dt:.1f}s")
            cited = sorted(set(PMID_RE.findall(tr["answer"])), key=int)
            if cited:
                st.markdown("### Sources cited")
                for pmid in cited:
                    source_card(pmid)

# ===========================================================================
# Evidence
# ===========================================================================
with evidence_tab:
    st.write("Query the literature retriever directly — local FAISS search, **$0**, no API key.")
    eq = st.text_input("Search the heart-failure literature",
                       "spironolactone in heart failure with reduced ejection fraction")
    k = st.slider("Results", 1, 10, 5)
    if eq.strip():
        warm_retriever()
        for h in search_literature(eq, k=k):
            st.markdown(
                f"**{h['score']:.3f}** · [{h['title']}]"
                f"(https://pubmed.ncbi.nlm.nih.gov/{h['pmid']}/) — "
                f"*{h['journal']}* ({h['year']}) · PMID {h['pmid']}")
            st.caption(h["text"][:300] + ("…" if len(h["text"]) > 300 else ""))
            st.divider()

# ===========================================================================
# Evaluation
# ===========================================================================
with eval_tab:
    st.write("Committed evaluation results — the reliability dashboard. (Regenerate "
             "with the scripts in `eval/`.)")

    ml = load_metrics("ml_metrics.json")
    if ml:
        st.subheader("Risk model (held-out test)")
        c = st.columns(4)
        c[0].metric("ROC-AUC", ml["test_roc_auc"])
        c[1].metric("PR-AUC", ml["test_pr_auc"],
                    help=f"no-skill baseline = prevalence {ml['pr_auc_baseline_prevalence']}")
        c[2].metric("Brier", ml["test_brier"],
                    help=f"base-rate baseline = {ml['brier_baseline']} (lower is better)")
        c[3].metric("CV ROC-AUC",
                    f'{ml["repeated_cv_roc_auc_mean"]}±{ml["repeated_cv_roc_auc_std"]}')

    ag = load_metrics("agent_metrics.json")
    if ag:
        st.subheader(f"Agent  ·  {ag.get('n', '?')} scenarios on {ag.get('model', '')}")
        c = st.columns(4)
        c[0].metric("Tool-selection", ag["tool_selection_accuracy"])
        c[1].metric("Appropriate deferral", ag["deferral_rate"])
        c[2].metric("Citation accuracy", ag["citation_accuracy"])
        c[3].metric("Eval cost", f'${ag["total_cost_usd"]}')

    sf = load_metrics("safety_metrics.json")
    if sf:
        st.subheader(f"Safety / adversarial  ·  {sf.get('n', '?')} scenarios")
        c = st.columns(3)
        c[0].metric("Overall resistance", sf["overall_resistance"])
        c[1].metric("Injection/leak resistance", sf["injection_resistance"])
        c[2].metric("Scope/diagnosis boundary", sf["boundary_rate"])

    figs = sorted((EVAL / "figures").glob("*.png"))
    if figs:
        st.subheader("Figures")
        fcols = st.columns(2)
        for i, f in enumerate(figs):
            fcols[i % 2].image(str(f), caption=f.stem.replace("_", " "))
