"""
Phase 3 — Train Logistic Regression Confidence Fusion Model

Loads per-frame (X, y) training data produced by build_training_data.py
and trains a LogisticRegression that captures co-occurrence patterns
(e.g. geometry AND velocity simultaneously) better than a weighted average.

Usage:
    python RapidAid-Accident-Detection-System/scripts/train_fusion_lr.py
"""
import sys
import os
import pickle
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAPIDAID_ROOT = os.path.dirname(SCRIPT_DIR)
WEIGHTS_DIR = os.path.join(RAPIDAID_ROOT, "weights")

sys.path.insert(0, os.path.dirname(RAPIDAID_ROOT))  # Mini_Project/
sys.path.insert(0, RAPIDAID_ROOT)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

FEATURE_NAMES = ["detector", "tracking", "velocity", "optical_flow", "disappearance", "geometry"]


def main():
    # ── 3a. Load training data ──────────────────────────────────────────
    x_path = os.path.join(WEIGHTS_DIR, "fusion_X_train.npy")
    y_path = os.path.join(WEIGHTS_DIR, "fusion_y_train.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"[ERROR] Training data not found. Run build_training_data.py first.")
        sys.exit(1)

    X = np.load(x_path)
    y = np.load(y_path)
    print(f"Loaded: X shape={X.shape}, y shape={y.shape}")
    print(f"  Positive (y=1): {int(y.sum())} ({100*y.mean():.1f}%)")
    print(f"  Negative (y=0): {int((1-y).sum())} ({100*(1-y).mean():.1f}%)")
    print()

    # ── 3b. Scale and train ─────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        C=0.1,                    # strong regularization — important for small dataset
        class_weight="balanced",  # handles severe class imbalance automatically
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    model.fit(X_scaled, y)
    print("Model trained.")

    # ── 3c. Cross-validation AUC ────────────────────────────────────────
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="roc_auc")
    print(f"CV AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    if cv_scores.mean() < 0.70:
        print("[WARN] CV AUC < 0.70 — investigate data quality or expand GT windows")
    else:
        print("CV AUC PASS (>= 0.70)")
    print()

    # ── 3d. Coefficients ────────────────────────────────────────────────
    print("Learned coefficients (positive = more collision-like):")
    for name, coef in zip(FEATURE_NAMES, model.coef_[0]):
        bar = "+" * max(0, int(abs(coef) * 10))
        direction = "+" if coef >= 0 else "-"
        print(f"  {name:15s}: {coef:+.4f}  {direction}{bar}")

    intercept = model.intercept_[0]
    print(f"  {'intercept':15s}: {intercept:+.4f}")
    print()

    # Sanity check
    all_positive = all(c >= 0 for c in model.coef_[0])
    if not all_positive:
        neg_features = [f for f, c in zip(FEATURE_NAMES, model.coef_[0]) if c < 0]
        print(f"[WARN] Negative coefficients for: {neg_features}")
        print("       This may indicate label inversion — check GT_IMPACT_WINDOWS")
    else:
        print("All coefficients positive — expected direction confirmed.")

    # ── 3e. Save model and scaler ───────────────────────────────────────
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    model_path = os.path.join(WEIGHTS_DIR, "fusion_lr_model.pkl")
    bundle = {"model": model, "scaler": scaler, "feature_names": FEATURE_NAMES}
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved: {model_path}")

    # Quick smoke test — 6-dim: [det, trk, vel, flow, dis, geo]
    x_test = np.array([[0.85, 0.90, 0.95, 0.70, 0.50, 0.80]])
    x_test_scaled = scaler.transform(x_test)
    prob = model.predict_proba(x_test_scaled)[0][1]
    print(f"Smoke test (all-strong signals): {prob:.3f}  (expected > 0.5)")


if __name__ == "__main__":
    main()
