"""
Phase 3 — Build LR Training Data from Tracked Reports

Reads per-frame signal data from tracked_Acc video N.json reports
and uses confidence-threshold labeling to produce labeled (X, y) arrays.

LABELING STRATEGY: confidence-threshold (not GT_IMPACT_WINDOWS)
  y=1 if final_confidence > CONF_POSITIVE_THRESHOLD  (pipeline detected crash)
  y=0 if final_confidence < CONF_NEGATIVE_THRESHOLD  (clearly non-crash)
  skip: ambiguous zone between thresholds

Rationale: GT_IMPACT_WINDOWS define PHYSICAL collision time, but the pipeline's
signals peak AFTER the collision due to accumulation logic. Using GT windows
produces inverted coefficients (high vel/geo → lower probability) because
aftermath frames have higher signals than impact-window frames.
The confidence-threshold approach trains LR to reproduce the weighted model's
correct behavior, but with learned interaction weights that can capture
co-occurrence patterns.

Does NOT call main_pipeline.py — reads only existing JSON files.

Usage:
    python RapidAid-Accident-Detection-System/scripts/build_training_data.py
"""
import sys
import os
import json
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Mini_Project/
RAPIDAID_ROOT = os.path.dirname(SCRIPT_DIR)                  # RapidAid-Accident-Detection-System/

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, RAPIDAID_ROOT)

REPORTS_DIR = os.path.join(RAPIDAID_ROOT, "outputs", "reports")
WEIGHTS_DIR = os.path.join(RAPIDAID_ROOT, "weights")

# ── Labeling thresholds ─────────────────────────────────────────────────
# Frames above POSITIVE_THRESH → crash confirmed (y=1)
# Frames below NEGATIVE_THRESH → clearly no crash (y=0)
# Between thresholds → skipped (ambiguous)
CONF_POSITIVE_THRESHOLD = 0.65   # strong crash detection
CONF_NEGATIVE_THRESHOLD = 0.25   # clearly pre-event or no event

# ── Feature order ───────────────────────────────────────────────────────
# All 6 signals: positive coefficients expected for all (high signal = crash)
FEATURE_NAMES = ["detector", "tracking", "velocity", "optical_flow", "disappearance", "geometry"]

# Key name aliases in the JSON
SIGNAL_ALIASES = {
    "detector":      ["detector_score", "detector"],
    "tracking":      ["tracking_score", "tracking"],
    "velocity":      ["velocity_score", "velocity"],
    "optical_flow":  ["optical_flow_score", "optical_flow", "flow"],
    "disappearance": ["disappearance_score", "disappearance"],
    "geometry":      ["geometry_score", "geometry"],
}


def get_signal(frame_dict: dict, key: str) -> float:
    """Extract signal value from frame dict, trying all known aliases."""
    for alias in SIGNAL_ALIASES[key]:
        val = frame_dict.get(alias)
        if val is not None:
            return float(val)
    return 0.0


def load_video_frames(report_path: str):
    """Load per-frame signal data from a tracked report JSON."""
    with open(report_path, "r") as f:
        data = json.load(f)
    frames = data.get("per_frame_summary", [])
    return frames


def build_training_data():
    """
    Build (X, y) training arrays from tracked report JSON files using
    confidence-threshold labeling.

    Returns:
        X: np.ndarray of shape (N, 6)
        y: np.ndarray of shape (N,) with 0/1 labels
    """
    X_rows = []
    y_rows = []

    video_names = [
        "Acc video 1", "Acc video 2", "Acc video 3", "Acc video 4",
        "Acc video 5", "Acc video 6", "Acc video 7", "Acc video 8",
        "Acc video 9",
    ]

    print("Building training data from tracked reports (confidence-threshold labeling)...")
    print(f"  y=1 threshold: conf > {CONF_POSITIVE_THRESHOLD}")
    print(f"  y=0 threshold: conf < {CONF_NEGATIVE_THRESHOLD}")
    print(f"  skip: {CONF_NEGATIVE_THRESHOLD} <= conf <= {CONF_POSITIVE_THRESHOLD}\n")

    for vid_name in video_names:
        report_name = f"tracked_{vid_name}.json"
        report_path = os.path.join(REPORTS_DIR, report_name)

        if not os.path.exists(report_path):
            print(f"  [WARN] Missing report: {report_path}")
            continue

        frames = load_video_frames(report_path)
        if not frames:
            print(f"  [WARN] No per-frame data in {report_name}")
            continue

        pos = neg = skip = 0
        for frame in frames:
            conf = frame.get("final_confidence", 0.5)
            x = [get_signal(frame, feat) for feat in FEATURE_NAMES]

            if conf > CONF_POSITIVE_THRESHOLD:
                X_rows.append(x)
                y_rows.append(1)
                pos += 1
            elif conf < CONF_NEGATIVE_THRESHOLD:
                X_rows.append(x)
                y_rows.append(0)
                neg += 1
            else:
                skip += 1

        print("  %-12s: %3d frames - y=1: %d, y=0: %d, skip: %d" % (vid_name, len(frames), pos, neg, skip))

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    return X, y


def main():
    X, y = build_training_data()

    n_total = len(y)
    n_pos = int(y.sum())
    n_neg = n_total - n_pos

    print()
    print("=" * 55)
    print(f"Total labeled frames:    {n_total}")
    print(f"Crash frames (y=1):      {n_pos} ({100*n_pos/n_total:.1f}%)")
    print(f"Non-crash frames (y=0):  {n_neg} ({100*n_neg/n_total:.1f}%)")
    print("=" * 55)

    if n_pos < 10:
        print("\n[ERROR] Too few positive samples. Lower CONF_POSITIVE_THRESHOLD.")
        sys.exit(1)
    if n_neg < 10:
        print("\n[ERROR] Too few negative samples. Raise CONF_NEGATIVE_THRESHOLD.")
        sys.exit(1)

    # Save
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    x_path = os.path.join(WEIGHTS_DIR, "fusion_X_train.npy")
    y_path = os.path.join(WEIGHTS_DIR, "fusion_y_train.npy")
    np.save(x_path, X)
    np.save(y_path, y)
    print(f"\nSaved: X shape={X.shape}, y shape={y.shape}")
    print(f"  {x_path}")
    print(f"  {y_path}")


if __name__ == "__main__":
    main()
