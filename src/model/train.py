"""
Train the calibrated heart-failure mortality-risk model -- the artifact behind the
estimate_patient_risk tool.

This makes the Day-1 modeling decisions reproducible from a single script (they
previously lived only in notebooks/eda.ipynb):

  1. Load UCI Heart Failure Clinical Records (299 patients); DROP `time`
     (follow-up length -> target leakage) and the DEATH_EVENT target.
  2. Stratified 80/20 split (random_state=42) -- the SAME split eval/eval_model.py
     re-creates, so the held-out test set stays untouched and the reported
     test ROC-AUC (~0.748) reproduces.
  3. Compare candidates by 5-fold CV ROC-AUC on the TRAIN set only:
     Logistic Regression (scaled), Random Forest, XGBoost.
  4. Deploy a CALIBRATED Logistic Regression: competitive discrimination on this
     small dataset, but better-calibrated probabilities and fully interpretable --
     the right trade-off for a clinical risk estimate. Calibrated with
     CalibratedClassifierCV(method='sigmoid', cv=5).
  5. Save the three artifacts the tools load: the calibrated model, the feature
     schema (names + order), and a fixed SHAP background sample.

Everything is local and deterministic. No API calls, zero cost.

Run:  python src/model/train.py
Then: python eval/eval_model.py   # refreshes the committed metrics + figures
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
LEAKY = "time"          # follow-up length -> entangled with the outcome; excluded by design
RANDOM_STATE = 42


def load_xy():
    """UCI Heart Failure (id=519) -> (X without `time`, y, schema in dataset order)."""
    ds = fetch_ucirepo(id=519)
    features = ds.data.features
    schema = [c for c in features.columns if c != LEAKY]   # 11 features, dataset order
    X = features[schema].copy()
    y = ds.data.targets.iloc[:, 0].astype(int)
    return X, y, schema


def lr_pipeline() -> Pipeline:
    """Scale -> logistic regression: the deployed base estimator."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])


def compare_models(X_train, y_train) -> None:
    """5-fold CV ROC-AUC for each candidate -- the model-selection evidence."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        "LogisticRegression (scaled)": lr_pipeline(),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
    }
    try:
        from xgboost import XGBClassifier
        candidates["XGBoost"] = XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1
        )
    except ImportError:
        print("  (xgboost not installed -- skipping that comparator)")

    print("5-fold CV ROC-AUC on the training set (model selection):")
    for name, est in candidates.items():
        s = cross_val_score(est, X_train, y_train, cv=cv, scoring="roc_auc")
        print(f"  {name:30s} {s.mean():.3f} +/- {s.std():.3f}")
    print("  -> deploying CALIBRATED Logistic Regression: interpretable, well-calibrated,")
    print("     and competitive on this small dataset.\n")


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    X, y, schema = load_xy()
    print(f"loaded {len(X)} patients, {len(schema)} features (dropped '{LEAKY}'); "
          f"death rate {y.mean():.1%}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"train {X_train.shape[0]} / test {X_test.shape[0]} (stratified, seed {RANDOM_STATE})\n")

    compare_models(X_train, y_train)

    # deployed model: calibrated logistic regression, fit on the TRAIN set only
    model = CalibratedClassifierCV(lr_pipeline(), method="sigmoid", cv=5)
    model.fit(X_train, y_train)

    # fixed SHAP background: a 100-row reference sample from the training set
    background = X_train.sample(n=100, random_state=RANDOM_STATE)

    joblib.dump(model, MODELS / "heart_failure_risk_model.joblib")
    joblib.dump(schema, MODELS / "feature_schema.joblib")
    joblib.dump(background, MODELS / "shap_background.joblib")

    # honest held-out sanity check (full metrics + figures come from eval/eval_model.py)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"held-out test ({len(y_test)} patients): "
          f"ROC-AUC {roc_auc_score(y_test, proba):.3f}  "
          f"Brier {brier_score_loss(y_test, proba):.3f}")
    print("saved -> models/{heart_failure_risk_model,feature_schema,shap_background}.joblib")
    print("next: python eval/eval_model.py   # refresh committed metrics + figures")


if __name__ == "__main__":
    main()
