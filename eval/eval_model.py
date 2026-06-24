"""
ML model evaluation: held-out test performance + calibration + interpretability.

Reconstructs the exact Day-1 train/test split (drop `time`, stratified 80/20,
random_state=42), evaluates the saved calibrated model on the untouched test set,
and commits the metrics + figures (ROC, PR, reliability, SHAP) for the README.
Local, $0. Reproducing test ROC-AUC ~0.748 confirms the split matches Day 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import joblib
from sklearn.base import clone
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    RocCurveDisplay, PrecisionRecallDisplay,
)
from sklearn.calibration import calibration_curve
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
FIGS = ROOT / "eval" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    model = joblib.load(MODELS / "heart_failure_risk_model.joblib")
    schema = joblib.load(MODELS / "feature_schema.joblib")

    ds = fetch_ucirepo(id=519)
    X = ds.data.features[schema].copy()          # drops `time`, fixes column order
    y = ds.data.targets.iloc[:, 0].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ---- numeric metrics --------------------------------------------------
    proba = model.predict_proba(X_te)[:, 1]
    auroc = float(roc_auc_score(y_te, proba))
    prauc = float(average_precision_score(y_te, proba))
    brier = float(brier_score_loss(y_te, proba))
    prevalence = float(y.mean())
    brier_base = prevalence * (1 - prevalence)

    base = clone(getattr(model, "estimator", None) or getattr(model, "base_estimator"))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    cv_auroc = cross_val_score(base, X_tr, y_tr, scoring="roc_auc", cv=cv)

    print(f"Test set: {len(y_te)} patients, {y_te.mean():.1%} deaths")
    print(f"  ROC-AUC : {auroc:.3f}   (reproduces Day 1 if ~0.748)")
    print(f"  PR-AUC  : {prauc:.3f}   (no-skill baseline = prevalence {prevalence:.2f})")
    print(f"  Brier   : {brier:.3f}   (base-rate baseline = {brier_base:.3f}; lower is better)")
    print(f"  repeated-CV ROC-AUC: {cv_auroc.mean():.3f} +/- {cv_auroc.std():.3f}")

    metrics = {
        "test_n": int(len(y_te)),
        "test_roc_auc": round(auroc, 4),
        "test_pr_auc": round(prauc, 4),
        "test_brier": round(brier, 4),
        "brier_baseline": round(brier_base, 4),
        "pr_auc_baseline_prevalence": round(prevalence, 4),
        "repeated_cv_roc_auc_mean": round(float(cv_auroc.mean()), 4),
        "repeated_cv_roc_auc_std": round(float(cv_auroc.std()), 4),
    }
    (ROOT / "eval" / "ml_metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- figures ----------------------------------------------------------
    RocCurveDisplay.from_predictions(y_te, proba)
    plt.title("ROC curve (held-out test)"); plt.tight_layout()
    plt.savefig(FIGS / "roc_curve.png", dpi=120); plt.close()

    PrecisionRecallDisplay.from_predictions(y_te, proba)
    plt.title("Precision-Recall curve (held-out test)"); plt.tight_layout()
    plt.savefig(FIGS / "pr_curve.png", dpi=120); plt.close()

    frac_pos, mean_pred = calibration_curve(y_te, proba, n_bins=5, strategy="uniform")
    plt.figure()
    plt.plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
    plt.plot(mean_pred, frac_pos, "o-", label="model")
    plt.xlabel("Predicted risk"); plt.ylabel("Observed death frequency")
    plt.title(f"Reliability curve (Brier {brier:.3f})"); plt.legend(); plt.tight_layout()
    plt.savefig(FIGS / "reliability_curve.png", dpi=120); plt.close()

    # SHAP global summary over the test set (guarded so a plot hiccup can't lose metrics)
    try:
        import shap
        f = lambda Z: model.predict_proba(Z)[:, 1]
        bg = joblib.load(MODELS / "shap_background.joblib")
        sv = shap.Explainer(f, bg, algorithm="exact")(X_te)
        shap.plots.beeswarm(sv, show=False)
        plt.tight_layout()
        plt.savefig(FIGS / "shap_summary.png", dpi=120, bbox_inches="tight"); plt.close()
        print("  SHAP summary plot saved")
    except Exception as e:
        print(f"  (SHAP plot skipped: {e})")

    print(f"\nsaved figures -> {FIGS}")
    print("saved metrics -> eval/ml_metrics.json")


if __name__ == "__main__":
    main()
