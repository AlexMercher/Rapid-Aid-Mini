"""
Phase 3 -- Build LR Training Data: Three-Zone Labeling

THREE-ZONE strategy (no ambiguity, no aftermath contamination):

  NEGATIVE (y=0): frames where timestamp < first_suspicious_time
                  Definitively pre-event clean traffic.
                  first_suspicious_time = timestamp of first CLEAR->SUSPICIOUS
                  transition (from state_transitions in tracked JSON).

  POSITIVE (y=1): frames where gt_start <= timestamp <= gt_end
                  Verified collision windows from GT_IMPACT_WINDOWS.
                  GT label wins if overlap with CLEAR zone (e.g. V7).

  SKIP:           all other frames (SUSPICIOUS buildup, CONFIRMED,
                  AFTERMATH) -- excludes bus-pass false positive, etc.

Rationale:
  - CLEAR frames have definitively LOW signals (no crash building up yet)
  - GT impact frames have HIGH signals at the physical collision moment
  - Aftermath frames (high vel but not crash) are excluded, preventing
    the inverted-coefficient failure from the first attempt
  - Bus-pass false positive (V1, 9-14s, CONFIRMED) is in SKIP zone

Does NOT call main_pipeline.py -- reads existing JSON files only.

Usage:
    python RapidAid-Accident-Detection-System/scripts/build_training_data.py
"""
import sys
import os
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAPIDAID_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(RAPIDAID_ROOT)

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, RAPIDAID_ROOT)

from shared.constants import GT_IMPACT_WINDOWS

REPORTS_DIR = os.path.join(RAPIDAID_ROOT, "outputs", "reports")
WEIGHTS_DIR = os.path.join(RAPIDAID_ROOT, "weights")

# Feature names -- 6 signals
FEATURE_NAMES = [
    "detector_score",
    "tracking_score",
    "velocity_score",
    "optical_flow_score",
    "disappearance_score",
    "geometry_score",
]

# Friendly names for display
FEATURE_DISPLAY = ["detector", "tracking", "velocity", "optical_flow", "disappearance", "geometry"]


def get_first_suspicious_time(tracked_data: dict) -> float:
    """
    Extract the timestamp of the first CLEAR->SUSPICIOUS transition.
    Falls back to first_confirmed_time if no SUSPICIOUS transition found.
    """
    transitions = tracked_data.get("state_transitions", [])
    for t in transitions:
        if t.get("from") == "CLEAR" and t.get("to") == "SUSPICIOUS":
            return float(t["timestamp"])
    # No SUSPICIOUS phase -- use confirmed time as fallback
    fc = tracked_data.get("first_confirmed_time")
    if fc is not None:
        return float(fc)
    return 0.0


def normalize_video_name(vid_name: str) -> str:
    """Resolve vid_name to GT_IMPACT_WINDOWS key (case-insensitive)."""
    for k in GT_IMPACT_WINDOWS:
        if k.lower() == vid_name.lower():
            return k
    return None


def build_training_data():
    """
    Build (X, y) using three-zone labeling from tracked JSON reports.

    Returns:
        X: np.ndarray of shape (N, 6)
        y: np.ndarray of shape (N,) with 0/1 labels
    """
    X_rows, y_rows = [], []

    video_names = [
        "Acc video 1", "Acc video 2", "Acc video 3", "Acc video 4",
        "Acc video 5", "Acc video 6", "Acc video 7", "Acc video 8",
        "Acc video 9",
    ]

    print("Building training data (three-zone labeling)...")
    print("  y=0: frames where t < first_suspicious_time (CLEAR zone)")
    print("  y=1: frames in GT_IMPACT_WINDOW (GT label wins on overlap)")
    print("  SKIP: all other frames (aftermath, SUSPICIOUS buildup)\n")

    for vid_name in video_names:
        report_path = os.path.join(REPORTS_DIR, "tracked_%s.json" % vid_name)
        if not os.path.exists(report_path):
            print("  [WARN] Missing: %s" % report_path)
            continue

        with open(report_path) as f:
            tracked = json.load(f)

        frames = tracked.get("per_frame_summary", [])
        if not frames:
            print("  [WARN] No per_frame_summary in %s" % report_path)
            continue

        # Check signals are present
        if "detector_score" not in frames[0]:
            print("  [WARN] No signal scores in %s -- re-run pipeline first" % report_path)
            continue

        # Resolve GT window
        gt_key = normalize_video_name(vid_name)
        if gt_key is None:
            # Try title-case
            gt_key = normalize_video_name(vid_name.title())
        if gt_key is None:
            print("  [WARN] No GT window for '%s' -- skipping" % vid_name)
            continue

        gt_start, gt_end = GT_IMPACT_WINDOWS[gt_key]

        # Get first suspicious time
        first_susp = get_first_suspicious_time(tracked)

        pos = neg = skip = 0
        for frame in frames:
            t = float(frame.get("timestamp_sec", -1))
            x = [float(frame.get(fn, 0.0)) for fn in FEATURE_NAMES]

            # Priority 1: GT impact window -> y=1
            if gt_start <= t <= gt_end:
                X_rows.append(x)
                y_rows.append(1)
                pos += 1
            # Priority 2: before first suspicious -> y=0 (CLEAR zone)
            elif t < first_susp:
                X_rows.append(x)
                y_rows.append(0)
                neg += 1
            # Priority 3: skip (aftermath / SUSPICIOUS buildup)
            else:
                skip += 1

        print("  %-12s first_susp=%.2fs gt=[%.1f,%.1f]  y=1: %d, y=0: %d, skip: %d" % (
            vid_name, first_susp, gt_start, gt_end, pos, neg, skip))

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
    print("Total labeled frames:    %d" % n_total)
    print("Impact frames  (y=1):   %d (%.1f%%)" % (n_pos, 100.0 * n_pos / n_total if n_total else 0))
    print("Pre-event frames (y=0): %d (%.1f%%)" % (n_neg, 100.0 * n_neg / n_total if n_total else 0))
    print("=" * 55)

    if n_pos < 5:
        print("\n[ERROR] Too few positive samples (%d). Check GT_IMPACT_WINDOWS." % n_pos)
        sys.exit(1)
    if n_neg < 5:
        print("\n[ERROR] Too few negative samples (%d). Check first_suspicious_time." % n_neg)
        sys.exit(1)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    x_path = os.path.join(WEIGHTS_DIR, "fusion_X_train.npy")
    y_path = os.path.join(WEIGHTS_DIR, "fusion_y_train.npy")
    np.save(x_path, X)
    np.save(y_path, y)
    print("\nSaved: X=%s, y=%s" % (str(X.shape), str(y.shape)))
    print("  %s" % x_path)
    print("  %s" % y_path)


if __name__ == "__main__":
    main()
