"""
estimate_patient_risk -- the agent's calibrated risk-model tool.

For a patient's baseline features, returns a calibrated mortality-risk
probability, an input-domain CONFIDENCE flag (does this patient resemble the
training data, or is the model extrapolating?), and the top SHAP contributing
factors. Local inference only -- no API calls, zero cost.

The interface is deployment-agnostic: estimate(features) is a pure function over
a dict, so its body can later be swapped to call a SageMaker endpoint without
changing anything the agent sees.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"

_BOUNDARY = (
    "Calibrated model estimate of in-study mortality risk, not a diagnosis. "
    "Decision support only -- verify with a licensed clinician."
)


class RiskModel:
    """Loads model + schema + SHAP background once; reuse across patients."""

    def __init__(self) -> None:
        self.model = joblib.load(MODELS / "heart_failure_risk_model.joblib")
        self.schema = joblib.load(MODELS / "feature_schema.joblib")
        self.background = joblib.load(MODELS / "shap_background.joblib")

        # per-feature stats from the training sample -> drive the OOD confidence flag
        self.stats = {}
        for c in self.schema:
            col = self.background[c]
            self.stats[c] = {
                "binary": set(col.unique()) <= {0, 1},
                "mean": float(col.mean()),
                "std": float(col.std()),
            }

        # deterministic, exactly-additive SHAP over the calibrated predict_proba
        f = lambda X: self.model.predict_proba(X)[:, 1]
        self.explainer = shap.Explainer(f, self.background, algorithm="exact")

    def estimate(self, features: dict) -> dict:
        # 1) validate presence
        missing = [c for c in self.schema if c not in features]
        if missing:
            return {"error": f"missing required features: {missing}",
                    "required_features": self.schema}
        # 2) validate numeric + binary domain
        try:
            vals = {c: float(features[c]) for c in self.schema}
        except (TypeError, ValueError):
            return {"error": "all feature values must be numeric"}
        bad_binary = [c for c in self.schema
                      if self.stats[c]["binary"] and vals[c] not in (0.0, 1.0)]
        if bad_binary:
            return {"error": f"these features must be 0 or 1: {bad_binary}"}

        # 3) one-row frame in the exact training column order
        row = pd.DataFrame([[vals[c] for c in self.schema]], columns=self.schema)

        # 4) calibrated probability of death
        risk = float(self.model.predict_proba(row)[0, 1])

        # 5) input-domain confidence: flag continuous features >3 SD from training
        flags = []
        for c in self.schema:
            s = self.stats[c]
            if s["binary"] or s["std"] == 0:
                continue
            z = (vals[c] - s["mean"]) / s["std"]
            if abs(z) > 3:
                where = "high" if z > 0 else "low"
                flags.append(
                    f"{c} = {vals[c]:g} is unusually {where} versus training "
                    f"(mean ~{s['mean']:.1f}); model is extrapolating."
                )
        confidence = "high" if not flags else "moderate" if len(flags) == 1 else "low"

        # 6) top SHAP contributors for THIS patient
        sv = self.explainer(row)
        shap_arr = np.array(sv.values)
        shap_vals = shap_arr[0] if shap_arr.ndim == 2 else shap_arr
        base = float(np.array(sv.base_values).reshape(-1)[0])
        order = sorted(range(len(self.schema)),
                       key=lambda i: abs(shap_vals[i]), reverse=True)[:3]
        top_factors = [{
            "feature": self.schema[i],
            "value": vals[self.schema[i]],
            "effect": "increases risk" if shap_vals[i] > 0 else "decreases risk",
            "shap": round(float(shap_vals[i]), 4),
        } for i in order]

        return {
            "risk_probability": round(risk, 4),
            "risk_percent": f"{round(risk * 100)}%",
            "baseline_risk_percent": f"{round(base * 100)}%",
            "confidence": confidence,
            "uncertainty_flags": flags,
            "top_factors": top_factors,
            "note": _BOUNDARY,
        }


_risk_model: RiskModel | None = None


def estimate_patient_risk(features: dict) -> dict:
    """Calibrated mortality risk + confidence + top factors for one patient."""
    global _risk_model
    if _risk_model is None:
        _risk_model = RiskModel()
    return _risk_model.estimate(features)


if __name__ == "__main__":
    import json

    high = {"age": 75, "anaemia": 1, "creatinine_phosphokinase": 250, "diabetes": 1,
            "ejection_fraction": 20, "high_blood_pressure": 1, "platelets": 240000,
            "serum_creatinine": 2.5, "serum_sodium": 130, "sex": 1, "smoking": 0}
    low = {"age": 50, "anaemia": 0, "creatinine_phosphokinase": 120, "diabetes": 0,
           "ejection_fraction": 45, "high_blood_pressure": 0, "platelets": 280000,
           "serum_creatinine": 0.9, "serum_sodium": 140, "sex": 0, "smoking": 0}
    ood = {**high, "serum_creatinine": 12.0}
    incomplete = {k: v for k, v in low.items() if k != "serum_sodium"}

    for name, p in [("HIGH-RISK", high), ("LOW-RISK", low),
                    ("OOD creatinine", ood), ("MISSING feature", incomplete)]:
        print(f"=== {name} ===")
        print(json.dumps(estimate_patient_risk(p), indent=2))
        print()
