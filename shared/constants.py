"""
Shared constants and event tier definitions for the Multi-Stage
Event Validation Platform.
"""
import os

# ─── Project Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAPIDAID_ROOT = os.path.join(PROJECT_ROOT, "RapidAid-Accident-Detection-System")
REPORTS_ROOT = os.path.join(PROJECT_ROOT, "reports")
EVENTS_DIR = os.path.join(REPORTS_ROOT, "events")

# Report tier directories
TIER_DIRS = {
    "VERIFIED_MAJOR":  os.path.join(REPORTS_ROOT, "verified"),
    "VERIFIED_MINOR":  os.path.join(REPORTS_ROOT, "verified_minor"),
    "AMBIGUOUS":       os.path.join(REPORTS_ROOT, "ambiguous"),
    "LOW_CONFIDENCE":  os.path.join(REPORTS_ROOT, "low_confidence"),
    "HARD_NEGATIVE":   os.path.join(REPORTS_ROOT, "hard_negatives"),
}

# Ensure directories exist
for d in [EVENTS_DIR, *TIER_DIRS.values()]:
    os.makedirs(d, exist_ok=True)


# ─── Event Tiers ─────────────────────────────────────────────────────────────
class EventTier:
    VERIFIED_MAJOR = "VERIFIED_MAJOR"
    VERIFIED_MINOR = "VERIFIED_MINOR"
    AMBIGUOUS = "AMBIGUOUS"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"

    DISPATCHABLE = {VERIFIED_MAJOR, VERIFIED_MINOR}


# ─── Event-State Frame Config ───────────────────────────────────────────────
EVENT_STATE_FRAMES = 5         # pre_anomaly, convergence, impact, disruption, aftermath
SEMANTIC_FRAME_WIDTH = 768     # Resolution for bakllava input
SEMANTIC_FRAME_HEIGHT = 432

# ─── Ground Truth Impact Windows (Supervised Calibration) ──────────────────
GT_IMPACT_WINDOWS = {
    "Acc Video 1": (24.0, 26.0),
    "Acc Video 2": (13.0, 14.0),
    "Acc Video 3": (3.0, 4.0),
    "Acc Video 4": (1.5, 2.5),
    "Acc Video 5": (4.0, 5.0),
    "Acc Video 6": (3.5, 5.0),
    "Acc Video 7": (2.5, 4.0),
    "Acc Video 8": (10.0, 12.0),
    "Acc Video 9": (3.5, 4.5),
}

# ─── Event Clip Config ──────────────────────────────────────────────────────
EVENT_PRE_SECONDS = 8.0        # Seconds before event to include
EVENT_POST_SECONDS = 12.0      # Seconds after event to include
FRAME_BUFFER_SECONDS = 25.0    # Rolling buffer size

# ─── Consensus Thresholds ───────────────────────────────────────────────────
CONSENSUS_MAJOR_MIN_RAPIDAID = 0.60
CONSENSUS_MINOR_MIN_RAPIDAID = 0.35
CONSENSUS_AMBIGUOUS_MAX = 0.55
CONSENSUS_VETO_KEYWORDS = [
    "no collision", "no accident", "normal traffic",
    "no damage", "no impact", "parked vehicles",
]
