"""
Phase 3 Unit Tests -- ConfidenceFusion learned mode

Tests:
  T1: All-zero inputs -> final_confidence < 0.10 in learned mode
  T2: All-strong signals -> final_confidence > 0.70 in learned mode
  T3: LR output is in [0.0, 1.0]
  T4: Fallback to weighted when model file not found
  T5: Result dict has all required keys
  T6: Weighted fallback is deterministic
  T7: LR mode scores high-signal > low-signal (monotonicity check)

Run from Mini_Project root:
    python RapidAid-Accident-Detection-System/tests/test_confidence_fusion_phase3.py
"""
import sys
import os

# Ensure correct paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAPIDAID_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(RAPIDAID_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, RAPIDAID_ROOT)

# Required result dict keys
REQUIRED_KEYS = {"final_confidence", "weighted_components", "raw_scores", "dominant_signal"}


def check(condition, test_name, detail=""):
    status = "PASS" if condition else "FAIL"
    msg = "  [%s] %s" % (status, test_name)
    if detail:
        msg += ": %s" % detail
    print(msg)
    return condition


def make_fusion_learned(model_path=None):
    """Create ConfidenceFusion in learned mode with given model path."""
    from config import settings as s
    s.FUSION_MODE = "learned"
    if model_path is not None:
        s.FUSION_MODEL_PATH = model_path
    # Force reimport
    import importlib
    import models.confidence_fusion as cf
    importlib.reload(cf)
    return cf.ConfidenceFusion()


def make_fusion_weighted():
    """Create ConfidenceFusion in weighted mode."""
    from config import settings as s
    s.FUSION_MODE = "weighted"
    import importlib
    import models.confidence_fusion as cf
    importlib.reload(cf)
    return cf.ConfidenceFusion()


def run_tests():
    results = []
    print("=" * 60)
    print("  Phase 3 ConfidenceFusion Unit Tests")
    print("=" * 60)

    MODEL_PATH = "RapidAid-Accident-Detection-System/weights/fusion_lr_model.pkl"

    # ---------------------------------------------------------------
    print("\n--- Testing LEARNED mode ---")
    if not os.path.exists(MODEL_PATH):
        print("  [SKIP] Model not found. Run train_fusion_lr.py first.")
        fusion_lr = None
    else:
        fusion_lr = make_fusion_learned(MODEL_PATH)

    if fusion_lr is not None:
        # T1: All zeros -> very low confidence
        t1_r = fusion_lr.compute(
            detector_score=0.0, tracking_score=0.0, velocity_score=0.0,
            optical_flow_score=0.0, disappearance_score=0.0, geometry_score=0.0
        )
        t1 = check(t1_r["final_confidence"] < 0.10, "T1: zeros -> conf < 0.10",
                   "got %.3f" % t1_r["final_confidence"])
        results.append(t1)

        # T2: All strong signals -> high confidence
        t2_r = fusion_lr.compute(
            detector_score=0.90, tracking_score=0.95, velocity_score=0.90,
            optical_flow_score=0.85, disappearance_score=0.70, geometry_score=0.95
        )
        t2 = check(t2_r["final_confidence"] > 0.70, "T2: strong -> conf > 0.70",
                   "got %.3f" % t2_r["final_confidence"])
        results.append(t2)

        # T3: Output bounded in [0, 1]
        t3 = check(0.0 <= t2_r["final_confidence"] <= 1.0, "T3: output in [0.0, 1.0]",
                   "got %.3f" % t2_r["final_confidence"])
        results.append(t3)

        # T7: Monotonicity -- high signals > low signals
        low_r = fusion_lr.compute(
            detector_score=0.10, tracking_score=0.05, velocity_score=0.00,
            optical_flow_score=0.05, disappearance_score=0.05, geometry_score=0.10
        )
        t7 = check(
            t2_r["final_confidence"] > low_r["final_confidence"],
            "T7: high-signal conf > low-signal conf",
            "high=%.3f, low=%.3f" % (t2_r["final_confidence"], low_r["final_confidence"])
        )
        results.append(t7)

    # ---------------------------------------------------------------
    print("\n--- Testing result dict structure ---")
    fusion_w = make_fusion_weighted()
    w_r = fusion_w.compute(
        detector_score=0.8, tracking_score=0.7, velocity_score=0.6,
        optical_flow_score=0.5, disappearance_score=0.3, geometry_score=0.9
    )
    t5 = check(REQUIRED_KEYS.issubset(set(w_r.keys())),
               "T5: result dict has required keys",
               "keys=%s" % str(set(w_r.keys())))
    results.append(t5)

    # ---------------------------------------------------------------
    print("\n--- Testing fallback behavior ---")
    fusion_fb = make_fusion_learned("/nonexistent/path/model.pkl")
    fb_r = fusion_fb.compute(
        detector_score=0.8, tracking_score=0.7, velocity_score=0.6,
        optical_flow_score=0.5, disappearance_score=0.3, geometry_score=0.9
    )
    t4 = check(0.0 <= fb_r["final_confidence"] <= 1.0,
               "T4: fallback when model not found returns valid conf",
               "conf=%.3f" % fb_r["final_confidence"])
    results.append(t4)

    # T6: Weighted mode is deterministic
    fusion_w2 = make_fusion_weighted()
    ra = fusion_w2.compute(
        detector_score=0.8, tracking_score=0.7, velocity_score=0.6,
        optical_flow_score=0.5, disappearance_score=0.3, geometry_score=0.9
    )
    rb = fusion_w2.compute(
        detector_score=0.8, tracking_score=0.7, velocity_score=0.6,
        optical_flow_score=0.5, disappearance_score=0.3, geometry_score=0.9
    )
    t6 = check(ra["final_confidence"] == rb["final_confidence"],
               "T6: weighted mode is deterministic",
               "a=%.3f, b=%.3f" % (ra["final_confidence"], rb["final_confidence"]))
    results.append(t6)

    # Restore correct settings
    from config import settings as s
    s.FUSION_MODE = "learned"
    s.FUSION_MODEL_PATH = MODEL_PATH

    # ---------------------------------------------------------------
    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    print("  RESULT: %d/%d tests passed" % (passed, total))
    print("  STATUS: %s" % ("ALL PASS" if passed == total else "FAILURES DETECTED"))
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
