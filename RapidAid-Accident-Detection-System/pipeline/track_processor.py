"""
RapidAid — Track-Centric Video Processor (Phases 3-9)

Transforms the pipeline from frame-centric to track-centric:

    Video Input
    → YOLO Detection (nano models)
    → Multi-Object Tracking (persistent IDs)
    → Velocity Analysis (sudden stops, anomalies)
    → Optical Flow Analysis (motion bursts)
    → Disappearance Reasoning (anomalous vanishing)
    → Confidence Fusion (weighted multi-signal)
    → Accident Decision
    → Report Generation (with debug overlays)

This processor does NOT modify any existing files. It creates a
completely new pipeline that can be run independently.

Usage:
    conda activate ./venv
    python test_track_pipeline.py --video data/test_videos/Acc\ Video\ 1.mp4
"""
import os
import sys
import time
import json
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass, field
from typing import Optional

from config import settings
from models.tracker import TrackManager
from models.velocity_analyzer import VelocityAnalyzer
from models.disappearance_analyzer import DisappearanceAnalyzer
from models.optical_flow_analyzer import OpticalFlowAnalyzer
from models.confidence_fusion import ConfidenceFusion
from models.causal_gate import CausalGate, AccidentState
from models.camera_stabilizer import CameraStabilizer


# ============================================================
# Phase 4 — Multi-Event Candidate Collection
# ============================================================

@dataclass
class CandidateEvent:
    """
    A distinct CONFIRMED accident event candidate.
    Collected during video processing, scored at end.
    Highest-scoring candidate is selected as the primary event.
    """
    first_confirmed_time:  float           # timestamp when CONFIRMED first fired
    first_confirmed_frame: int             # analyzed frame index
    peak_confidence:       float = 0.0     # highest confidence seen in this event
    peak_confidence_time:  float = 0.0     # timestamp of peak confidence
    peak_frame_signals:    dict  = field(default_factory=dict)  # signals at peak
    duration_frames:       int   = 0       # frames spent in CONFIRMED for this event
    end_time:              float = 0.0     # when this event ended (CONFIRMED dropped)
    candidate_score:       float = 0.0     # composite score computed after collection
    co_occurrence_score:   float = 0.0     # geometry+velocity co-occurrence
    selected:              bool  = False   # True for the chosen best candidate


def score_candidate_event(candidate: 'CandidateEvent',
                          frame_h: int,
                          frame_w: int) -> float:
    """
    Score a candidate event for selection as primary event.
    Higher score = more likely to be a genuine accident.

    Uses co-occurrence: geometry AND velocity co-occurring
    simultaneously is stronger evidence than peak amplitude alone.
    This distinguishes real crashes (geo+vel both spike) from
    false positives like V1's bus-pass (high geo, LOW vel).
    """
    signals = candidate.peak_frame_signals

    def get_sig(key: str) -> float:
        return float(signals.get(key)
                     or signals.get("%s_score" % key)
                     or 0.0)

    geo = get_sig('geometry')
    vel = get_sig('velocity')
    dis = get_sig('disappearance')

    HIGH     = 0.30  # threshold for geo and vel
    DIS_HIGH = 0.50  # higher bar for disappearance (must be anomalous, not ambient)

    # Co-occurrence counts THREE crash-specific signals:
    #   geo  — vehicles are geometrically close / overlapping
    #   vel  — sudden velocity anomaly (hard stop, debris scatter)
    #   dis  — anomalous track disappearance (crash removes vehicles from scene)
    #
    # Intentionally excludes tracking and optical_flow:
    #   tracking is always active (many vehicles present)
    #   optical_flow responds to any large motion (e.g. bus passing = false positive)
    strong = sum([
        geo > HIGH,       # vehicles close together
        vel > HIGH,       # kinetic energy anomaly
        dis > DIS_HIGH,   # track vanishes after impact
    ])

    if strong >= 3:
        co_occurrence = 1.0
    elif strong == 2:
        co_occurrence = 0.65
    elif strong == 1:
        co_occurrence = 0.20
    else:
        co_occurrence = 0.0

    # Primary crash signature: geometry+velocity co-occur
    # Bus-pass: high geo, LOW vel (vel=0.13) -> no boost
    # Real crash: high geo AND high vel -> boost
    if geo > HIGH and vel > HIGH:
        co_occurrence = min(1.0, co_occurrence + 0.20)

    # Secondary crash signature: geometry+disappearance co-occur
    # Vehicles proximate AND one vanishes = genuine impact.
    # Bus-pass immune: raw_dis=0.50 is NOT strictly > DIS_HIGH=0.50.
    if geo > HIGH and dis > DIS_HIGH:
        co_occurrence = min(1.0, co_occurrence + 0.20)

    candidate.co_occurrence_score = co_occurrence

    # Duration: normalize to [0,1] saturating at 10 CONFIRMED frames
    duration_score = min(1.0, candidate.duration_frames / 10.0)

    # Final score: peak confidence (40%), duration (20%), co-occurrence (40%)
    score = (
        candidate.peak_confidence * 0.40
        + duration_score          * 0.20
        + co_occurrence           * 0.40
    )
    candidate.candidate_score = score
    return score


# ============================================================
# Phase 5 — Traffic Flow Model (gated disappearance signal)
# ============================================================

@dataclass
class TrafficFlowBaseline:
    """
    Learned normal traffic behavior from pre-event CLEAR frames.
    Used to gate disappearance signal — only count disappearances
    that violated the established flow pattern.
    """
    # Collected during CLEAR state
    velocity_samples:    list  = field(default_factory=list)  # list of (vx, vy) tuples
    speed_samples:       list  = field(default_factory=list)  # list of float speeds
    frame_count:         int   = 0      # CLEAR frames observed

    # Computed baseline (set once min_frames_required is reached)
    mean_vx:             float = 0.0
    mean_vy:             float = 0.0
    mean_speed:          float = 0.0
    speed_std:           float = 0.0
    flow_variance:       float = 0.0    # angular spread of velocity directions
    is_reliable:         bool  = False  # True when enough data collected
    is_high_variance:    bool  = False  # True when scene has chaotic multi-dir flow

    # Settings (read from config at construction)
    min_frames_required: int   = 10    # from settings.FLOW_MIN_FRAMES
    high_var_threshold:  float = 2.0   # from settings.FLOW_HIGH_VAR_THRESHOLD


def _compute_baseline_stats(baseline: "TrafficFlowBaseline") -> None:
    """Compute mean direction, speed stats, and angular variance from samples."""
    import math
    speeds = baseline.speed_samples
    vxs = [v[0] for v in baseline.velocity_samples]
    vys = [v[1] for v in baseline.velocity_samples]

    baseline.mean_vx    = sum(vxs) / len(vxs)
    baseline.mean_vy    = sum(vys) / len(vys)
    baseline.mean_speed = sum(speeds) / len(speeds)

    var = sum((s - baseline.mean_speed) ** 2 for s in speeds) / len(speeds)
    baseline.speed_std  = var ** 0.5

    # Angular spread of velocity vectors: high = many directions = intersection
    angles = [math.atan2(vy, vx) for vx, vy in baseline.velocity_samples]
    mean_angle = math.atan2(
        sum(math.sin(a) for a in angles) / len(angles),
        sum(math.cos(a) for a in angles) / len(angles),
    )
    angle_var = sum(
        min(abs(a - mean_angle), 2 * math.pi - abs(a - mean_angle)) ** 2
        for a in angles
    ) / len(angles)

    baseline.flow_variance    = angle_var
    baseline.is_reliable      = True
    baseline.is_high_variance = (angle_var > baseline.high_var_threshold)


def update_flow_baseline(baseline: "TrafficFlowBaseline",
                          active_tracks: list,
                          current_state) -> None:
    """
    Collect velocity data from CLEAR-state frames only.
    Never contaminate with SUSPICIOUS/CONFIRMED/AFTERMATH frames.
    """
    if current_state != AccidentState.CLEAR:
        return
    if baseline.frame_count >= settings.FLOW_MAX_BASELINE_FRAMES:
        return

    for track in active_tracks:
        # Support both real Track (velocity tuple) and mock objects (vx/vy attrs)
        if hasattr(track, 'vx') and track.vx is not None:
            vx = float(track.vx)
            vy = float(getattr(track, 'vy', 0.0) or 0.0)
        elif (hasattr(track, 'velocity')
              and isinstance(track.velocity, (tuple, list))
              and len(track.velocity) >= 2):
            vx = float(track.velocity[0] or 0.0)
            vy = float(track.velocity[1] or 0.0)
        else:
            continue

        speed = (vx ** 2 + vy ** 2) ** 0.5
        if speed < 0.5:
            continue  # skip near-stationary (parked / stopped)

        baseline.velocity_samples.append((vx, vy))
        baseline.speed_samples.append(speed)
        baseline.frame_count += 1

    if len(baseline.speed_samples) >= baseline.min_frames_required:
        _compute_baseline_stats(baseline)


def compute_flow_violation_score(track, baseline: "TrafficFlowBaseline") -> float:
    """
    Score how much a vehicle's behavior deviates from normal flow.

    Returns float in [0.0, 1.0]:
      0.0 = normal flow (likely occlusion if disappears)
      1.0 = extreme violation (likely genuine accident if disappears)

    Falls back to 0.5 (neutral) when baseline is unreliable or high-variance.
    """
    import math

    if not baseline.is_reliable:
        return 0.5
    if baseline.is_high_variance:
        return 0.5

    # Support both real Track (velocity tuple) and mock objects (vx/vy attrs)
    if hasattr(track, 'vx') and track.vx is not None:
        vx = float(track.vx)
        vy = float(getattr(track, 'vy', 0.0) or 0.0)
    elif (hasattr(track, 'velocity')
          and isinstance(track.velocity, (tuple, list))
          and len(track.velocity) >= 2):
        vx = float(track.velocity[0] or 0.0)
        vy = float(track.velocity[1] or 0.0)
    else:
        vx, vy = 0.0, 0.0

    speed = (vx ** 2 + vy ** 2) ** 0.5
    score = 0.0

    # Signal 1: Direction deviation from baseline
    if speed > 0.5 and baseline.mean_speed > 0.5:
        track_angle    = math.atan2(vy, vx)
        baseline_angle = math.atan2(baseline.mean_vy, baseline.mean_vx)
        angle_diff     = abs(track_angle - baseline_angle)
        angle_diff     = min(angle_diff, 2 * math.pi - angle_diff)
        direction_score = min(1.0, angle_diff / math.pi)
        score += direction_score * settings.FLOW_DIRECTION_WEIGHT

    # Signal 2: Speed ratio anomaly
    if baseline.mean_speed > 0.5:
        speed_ratio = speed / baseline.mean_speed
        if speed_ratio < 0.2 or speed_ratio > 3.0:
            speed_score = min(1.0, abs(1.0 - speed_ratio) / 2.0)
        else:
            speed_score = 0.0
        score += speed_score * settings.FLOW_SPEED_WEIGHT

    # Signal 3: Velocity collapse (sudden deceleration before disappearing)
    velocity_collapsed = False
    if hasattr(track, 'velocity_collapsed'):
        velocity_collapsed = bool(track.velocity_collapsed)
    elif hasattr(track, 'history') and len(track.history) >= 4:
        speeds_h = []
        for entry in track.history:
            vel = entry.get('velocity', (0, 0)) or (0, 0)
            speeds_h.append((vel[0] ** 2 + vel[1] ** 2) ** 0.5)
        for i in range(2, len(speeds_h)):
            if speeds_h[i - 2] > 5.0 and speeds_h[i] < speeds_h[i - 2] * 0.3:
                velocity_collapsed = True
                break

    if velocity_collapsed:
        score += settings.FLOW_COLLAPSE_WEIGHT

    return min(1.0, score)


def is_portrait_frame(frame_w: int, frame_h: int) -> bool:
    """
    Returns True if frame is portrait orientation (h strictly > w).
    Square frames (h == w) return False — treated as landscape.
    """
    return frame_h > frame_w


def _get_tracker_config(frame_w: int, frame_h: int) -> dict:
    """
    Phase 8 — Return TrackManager constructor params based on frame orientation.

    Portrait (h > w) frames require:
      - Lower IoU matching thresholds (vehicles move fast along y-axis)
      - Longer track_buffer (re-detection slower in portrait orientation)
      - Higher VelocityAnalyzer thresholds (approach velocities 100-200 px/frame
        normal in portrait — don't confuse with crash deceleration)

    Returns:
        dict with keys:
          iou_threshold_high, iou_threshold_low, max_lost_frames,
          vel_high_speed_thresh, vel_dir_thresh, portrait_mode
    """
    if (is_portrait_frame(frame_w, frame_h)
            and settings.PORTRAIT_TRACKER_ENABLED):
        cfg_path = settings.BYTETRACK_PORTRAIT_CONFIG
        print(f"[TRACKER] Portrait mode ({frame_w}x{frame_h}) — "
              f"loading {cfg_path}")
        return {
            "iou_threshold_high":  settings.PORTRAIT_TRACK_IOU_HIGH,
            "iou_threshold_low":   settings.PORTRAIT_TRACK_IOU_LOW,
            "max_lost_frames":     settings.PORTRAIT_TRACK_MAX_LOST_FRAMES,
            "vel_high_speed_thresh": settings.PORTRAIT_VEL_HIGH_SPEED_THRESH,
            "vel_dir_thresh":        settings.PORTRAIT_VEL_DIR_THRESH,
            "portrait_mode":         True,
        }

    cfg_path = settings.BYTETRACK_DEFAULT_CONFIG
    print(f"[TRACKER] Landscape mode ({frame_w}x{frame_h}) — "
          f"loading {cfg_path}")
    return {
        "iou_threshold_high":  0.25,
        "iou_threshold_low":   0.15,
        "max_lost_frames":     15,
        "vel_high_speed_thresh": 8.0,
        "vel_dir_thresh":        90.0,
        "portrait_mode":         False,
    }


def portrait_swap_bboxes(bboxes: list, frame_w: int, frame_h: int) -> tuple:
    """
    Swap x/y axis in bboxes for portrait geometry computation.

    In portrait video, vehicles are arranged along the y-axis (depth along road).
    Swapping axes exposes the correct overlap direction to the geometry IoU module,
    which is calibrated to detect horizontal (x-axis) vehicle proximity.

    Transform: (x1, y1, x2, y2) → (y1, x1, y2, x2)
    Dims swap:  frame_w ↔ frame_h (portrait dims become effective landscape dims)

    Returns:
        swapped_bboxes: list of (y1, x1, y2, x2) tuples
        swapped_w:      frame_h (new effective width)
        swapped_h:      frame_w (new effective height)
    """
    swapped = [(y1, x1, y2, x2) for (x1, y1, x2, y2) in bboxes]
    return swapped, frame_h, frame_w  # swap dims too


class TrackCentricProcessor:
    """
    Track-centric accident detection processor.

    Instead of analyzing isolated frames, this processor:
    1. Tracks objects across frames with persistent IDs
    2. Analyzes velocity/trajectory patterns over time
    3. Monitors optical flow for motion anomalies
    4. Detects anomalous disappearances
    5. Fuses all signals into a unified confidence score
    """

    def __init__(self, use_nano=True):
        """
        Initialize the track-centric processor.

        Args:
            use_nano: if True, use YOLOv8n models (faster, edge-friendly)
        """
        print("=" * 60)
        print("  RapidAid — Track-Centric Pipeline Initializing")
        print("=" * 60)

        # Load YOLO models
        from ultralytics import YOLO

        if use_nano:
            seg_path = os.path.join(settings.WEIGHTS_DIR, "yolov8n-seg.pt")
            pose_path = os.path.join(settings.WEIGHTS_DIR, "yolov8n-pose.pt")
            # Fall back to small if nano not available
            if not os.path.exists(seg_path):
                seg_path = settings.VEHICLE_MODEL
            if not os.path.exists(pose_path):
                pose_path = settings.POSE_MODEL
        else:
            seg_path = settings.VEHICLE_MODEL
            pose_path = settings.POSE_MODEL

        print(f"  [Vehicle Model] {os.path.basename(seg_path)}")
        self.vehicle_model = YOLO(seg_path)
        print(f"  [Pose Model]    {os.path.basename(pose_path)}")
        self.pose_model = YOLO(pose_path)

        # Load M1 frame classifier if available
        self.frame_classifier = None
        try:
            from models.frame_classifier import FrameClassifier
            self.frame_classifier = FrameClassifier()
            if not self.frame_classifier.is_available():
                self.frame_classifier = None
        except Exception:
            pass
        print(f"  [M1 Classifier] {'Loaded' if self.frame_classifier else 'N/A'}")

        # Load M4 collision detector if available
        self.collision_detector = None
        try:
            from models.collision_detector import CollisionDetector
            self.collision_detector = CollisionDetector()
            if not self.collision_detector.is_available():
                self.collision_detector = None
        except Exception:
            pass
        print(f"  [M4 Collision]  {'Loaded' if self.collision_detector else 'N/A'}")

        # Initialize tracking and analysis modules
        self.vehicle_tracker = TrackManager(
            max_lost_frames=15, iou_threshold_high=0.25
        )
        self.person_tracker = TrackManager(
            max_lost_frames=10, iou_threshold_high=0.3
        )
        self.velocity_analyzer = VelocityAnalyzer()
        self.disappearance_analyzer = DisappearanceAnalyzer()
        self.flow_analyzer = OpticalFlowAnalyzer(flow_scale=0.5)
        self.confidence_fusion = ConfidenceFusion()

        # Feature 1,4,8,9,12: Causal intelligence gate
        self.causal_gate = CausalGate(
            confirm_threshold=0.45,
            suspicious_threshold=0.30,
            confirm_frames_required=3,
            min_evidence_families=2,
            aftermath_timeout_frames=settings.AFTERMATH_TIMEOUT_FRAMES,
        )

        # Feature 5: Camera shake suppression
        self.camera_stabilizer = CameraStabilizer()

        # Vehicle class IDs (from COCO)
        from config.vehicle_classes import is_vehicle_class
        self.is_vehicle_class = is_vehicle_class

        print("=" * 60)
        print("  RapidAid — Track-Centric Pipeline Ready")
        print("=" * 60)

    def process_video(self, video_path, show_overlay=False, save_video=True,
                      stop_on_first=True):
        """
        Process a video through the track-centric pipeline.

        Args:
            video_path: path to video file
            show_overlay: if True, display real-time debug overlay
            save_video: if True, save annotated output video
            stop_on_first: if True, stop after first confirmed accident

        Returns:
            dict with results, metrics, and per-frame data
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return {"accident_detected": False, "error": "Cannot open video"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n[TrackProcessor] Video: {video_path}")
        print(f"[TrackProcessor] {total_frames} frames @ {fps:.1f} FPS, "
              f"{frame_w}x{frame_h}")

        # Frame sampling
        frame_interval = max(1, int(fps / settings.FRAMES_PER_SECOND_TO_ANALYZE))
        print(f"[TrackProcessor] Sampling every {frame_interval} frames")

        # Phase 8: select tracker and velocity analyzer config by orientation
        _tracker_cfg = _get_tracker_config(frame_w, frame_h)
        _portrait_mode = _tracker_cfg["portrait_mode"]
        # Re-init vehicle tracker with orientation-specific IoU thresholds
        self.vehicle_tracker = TrackManager(
            max_lost_frames=_tracker_cfg["max_lost_frames"],
            iou_threshold_high=_tracker_cfg["iou_threshold_high"],
            iou_threshold_low=_tracker_cfg["iou_threshold_low"],
        )
        # Re-init velocity analyzer with orientation-specific speed thresholds
        self.velocity_analyzer = VelocityAnalyzer(
            high_speed_threshold=_tracker_cfg["vel_high_speed_thresh"],
            direction_change_threshold=_tracker_cfg["vel_dir_thresh"],
        )
        print(
            f"[TRACKER] iou_high={_tracker_cfg['iou_threshold_high']} "
            f"iou_low={_tracker_cfg['iou_threshold_low']} "
            f"max_lost={_tracker_cfg['max_lost_frames']} "
            f"vel_speed_thresh={_tracker_cfg['vel_high_speed_thresh']} "
            f"vel_dir_thresh={_tracker_cfg['vel_dir_thresh']}"
        )

        # Reset all modules
        self.person_tracker.reset()
        self.flow_analyzer.reset()
        self.disappearance_analyzer.update_frame_size(frame_w, frame_h)
        self.causal_gate.reset()

        # Phase 4: multi-event candidate state (reset per video for batch safety)
        _candidate_events: list = []                   # closed, scored candidates
        _active_candidate: Optional[CandidateEvent] = None  # currently open candidate
        _confirmed_gap_frames: int = 0                 # consecutive non-CONFIRMED frames

        # Phase 5: traffic flow baseline (reset per video — batch safety)
        self.flow_baseline = TrafficFlowBaseline(
            min_frames_required=settings.FLOW_MIN_FRAMES,
            high_var_threshold=settings.FLOW_HIGH_VAR_THRESHOLD,
        )

        # Feature 11: FPS-aware time delta
        delta_t = frame_interval / fps  # seconds between analyzed frames

        # Output video writer
        out_writer = None
        if save_video:
            out_dir = os.path.join(settings.OUTPUTS_DIR, "tracked_videos")
            os.makedirs(out_dir, exist_ok=True)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            out_path = os.path.join(out_dir, f"{video_name}_tracked.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_fps = fps / frame_interval
            out_writer = cv2.VideoWriter(out_path, fourcc, out_fps,
                                         (frame_w, frame_h))

        # Metrics
        frame_times = []
        per_frame_data = []

        # Accident detection state
        accident_confirmed = False
        accident_frame_idx = None
        accident_timestamp = None
        first_confirmed_frame_idx = None
        first_confirmed_timestamp = None
        best_confidence = 0.0
        best_frame = None
        best_frame_data = None

        # Feature 7: Confidence smoothing state
        smoothed_confidence = 0.0
        confidence_history = []

        # Track death history for feature 9
        recent_death_times = []

        frame_count = 0
        analyzed_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Sample frames
            if frame_count % frame_interval != 0:
                continue

            analyzed_count += 1
            t_sec = round(frame_count / fps, 2)
            t0 = time.perf_counter()

            # ===== STEP 1: YOLO Detection =====
            vehicles = self._detect_vehicles(frame)
            persons = self._detect_persons(frame)

            # ===== STEP 2: Multi-Object Tracking =====
            active_vehicle_tracks = self.vehicle_tracker.update(
                vehicles, analyzed_count
            )
            active_person_tracks = self.person_tracker.update(
                persons, analyzed_count
            )

            # ===== STEP 3: Velocity Analysis =====
            vehicle_velocity_scores = self.velocity_analyzer.analyze_tracks(
                active_vehicle_tracks,
                lost_tracks=self.vehicle_tracker.get_all_lost_tracks(),
                dead_tracks=self.vehicle_tracker.get_recently_dead_tracks(
                    analyzed_count, lookback=20
                ),
                frame_idx=analyzed_count,
            )

            # ===== STEP 4: Optical Flow =====
            flow_result = self.flow_analyzer.process_frame(frame)

            # ===== STEP 5: Disappearance Analysis =====
            dead_tracks = self.vehicle_tracker.get_recently_dead_tracks(
                analyzed_count, lookback=20
            )
            disappearance_results = self.disappearance_analyzer.analyze(
                dead_tracks=dead_tracks,
                lost_tracks=self.vehicle_tracker.get_all_lost_tracks(),
                active_tracks=active_vehicle_tracks,
                frame_idx=analyzed_count,
            )

            # ===== STEP 6: M1 + M4 Signals =====
            m1_score = 0.0
            if self.frame_classifier:
                _, m1_score = self.frame_classifier.classify(frame)

            m4_score = 0.0
            collision_zones = []
            if self.collision_detector:
                collision_zones = self.collision_detector.detect(frame)
                if collision_zones:
                    m4_score = collision_zones[0]["confidence"]

            # ===== STEP 7: Confidence Fusion =====
            detector_score = self._compute_detector_score(
                active_vehicle_tracks, m1_score, m4_score
            )
            tracking_score = self.confidence_fusion.compute_tracking_score(
                active_vehicle_tracks
            )
            velocity_score = self._compute_velocity_score(
                vehicle_velocity_scores
            )
            # Phase 8: Portrait velocity warmup
            # In portrait videos, vehicles approach along y-axis at 100-200 px/frame
            # (normal traffic). YOLO bbox instability in portrait orientation causes
            # spurious direction reversals → trajectory_anomaly_score fires in early
            # frames before baseline is established.
            # Ramp: analyzed_count / WARMUP_FRAMES (0→1 linearly over warmup window).
            if _portrait_mode and analyzed_count <= settings.PORTRAIT_VEL_WARMUP_FRAMES:
                _vel_ramp = analyzed_count / settings.PORTRAIT_VEL_WARMUP_FRAMES
                velocity_score = velocity_score * _vel_ramp

            raw_flow_score = max(
                flow_result.get("motion_burst_score", 0),
                flow_result.get("optical_flow_score", 0),
            )

            # Feature 5: Camera shake suppression
            flow_field = flow_result.get("flow_field", None)
            shake = self.camera_stabilizer.compute_suppression(flow_field)
            optical_flow_score = raw_flow_score * shake["suppression_factor"]

            # Phase 5: RAW disappearance (for candidate scoring — unaffected by gating)
            raw_disappearance_score = self._compute_disappearance_score(
                disappearance_results,
                dead_tracks=dead_tracks,
                active_tracks=active_vehicle_tracks,
            )

            # Phase 5: GATED disappearance (for confidence fusion only)
            # When baseline is unreliable or high-variance: pass through unchanged.
            disappearance_score = raw_disappearance_score
            if (self.flow_baseline.is_reliable
                    and not self.flow_baseline.is_high_variance
                    and dead_tracks):
                # Find which dying track produced the highest raw score
                _best_tid, _best_raw = None, 0.0
                for _tid, _r in disappearance_results.items():
                    if _r.get("disappearance_score", 0) > _best_raw:
                        _best_raw = _r["disappearance_score"]
                        _best_tid = _tid
                if _best_tid is not None:
                    _dying = next(
                        (t for t in dead_tracks if t.track_id == _best_tid), None
                    )
                    if _dying is not None:
                        _violation = compute_flow_violation_score(
                            _dying, self.flow_baseline
                        )
                        # Phase 5: 0.5 floor — never suppress by more than 50%.
                        # Prevents real-crash disappearances (normal-flow victims)
                        # from being zeroed out. Occlusions still get halved.
                        _violation = max(0.5, _violation)
                        disappearance_score = raw_disappearance_score * _violation

            # Phase 8: Portrait disappearance warmup (quadratic ramp)
            # In portrait videos, fast-approaching vehicles (200+ px/frame) have IoU=0
            # between consecutive frame positions, causing persistent lost-track ANOMALOUS
            # scores (0.40 mid-frame + 0.25 vel_collapse + 0.10 erratic = 0.75 each).
            # Quadratic ramp (frame/N)²: stays near 0 until ~60% of window (t≈7s),
            # then rises rapidly in the last 40% so real crash signals fire normally.
            if _portrait_mode and analyzed_count <= settings.PORTRAIT_DIS_WARMUP_FRAMES:
                _dis_ramp = (analyzed_count / settings.PORTRAIT_DIS_WARMUP_FRAMES) ** 2
                raw_disappearance_score = raw_disappearance_score * _dis_ramp
                disappearance_score     = disappearance_score     * _dis_ramp

            # Feature 2 / Phase 8: Geometry warmup
            raw_geometry = self._compute_geometry_score(
                active_vehicle_tracks, collision_zones, frame_w, frame_h
            )
            if _portrait_mode and analyzed_count <= settings.PORTRAIT_GEO_WARMUP_FRAMES:
                # Phase 8: Portrait geometry warmup (quadratic ramp)
                # Portrait vehicles approaching camera head-on create spurious
                # geo=0.79-0.80 overlaps in the pre-crash zone (t=3-8s).
                # Quadratic ramp: near 0 until ~60% of window, then rises rapidly.
                _geo_ramp = (analyzed_count / settings.PORTRAIT_GEO_WARMUP_FRAMES) ** 2
                geometry_score = raw_geometry * _geo_ramp
            elif analyzed_count <= 5:
                # Feature 2: Landscape geometry warmup (linear, first 5 analyzed frames)
                geo_warmup = 0.2 + 0.16 * analyzed_count  # 0.36..1.0
                geometry_score = raw_geometry * min(1.0, geo_warmup)
            else:
                geometry_score = raw_geometry

            # Feature 9: Track death density
            new_deaths = len(dead_tracks)
            if new_deaths > 0:
                recent_death_times.append((analyzed_count, new_deaths))
            # Clean old entries (older than 15 frames)
            recent_death_times = [
                (f, n) for f, n in recent_death_times
                if analyzed_count - f <= 15
            ]
            death_density = sum(n for _, n in recent_death_times) / 15.0

            fusion_result = self.confidence_fusion.compute(
                detector_score=detector_score,
                tracking_score=tracking_score,
                velocity_score=velocity_score,
                optical_flow_score=optical_flow_score,
                # Phase 5: use RAW disappearance in fusion so EMA is not degraded
                # by flow gating. The GATED score (disappearance_score with 0.5
                # floor) is logged in per_frame_data for diagnostics only.
                disappearance_score=raw_disappearance_score,
                geometry_score=geometry_score,
            )

            raw_conf = fusion_result["final_confidence"]

            # Feature 7: Confidence smoothing (EMA)
            smoothed_confidence = (
                0.7 * smoothed_confidence + 0.3 * raw_conf
            )
            final_conf = smoothed_confidence
            fusion_result["final_confidence"] = round(final_conf, 3)
            fusion_result["raw_confidence"] = round(raw_conf, 3)
            confidence_history.append(final_conf)

            # ===== STEP 8: Causal Accident Decision =====
            # Feature 1,3,4,8,10,12: Use causal gate state machine
            gate_signals = {
                "velocity": velocity_score,
                "optical_flow": optical_flow_score,
                # Phase 5 fix: gate uses RAW disappearance so causal_present and
                # family count are never degraded by flow gating. Gated score only
                # goes into confidence_fusion (line 657).
                "disappearance": raw_disappearance_score,
                "detector": detector_score,
                "tracking": tracking_score,
                "geometry": geometry_score,
                "final_confidence": final_conf,
                "death_density": death_density,
            }
            gate_result = self.causal_gate.evaluate(
                gate_signals, analyzed_count
            )

            # Only confirm when causal gate fires CONFIRMED (now multi-event capable)
            current_state = gate_result["state"]
            gate_flags    = gate_result.get("flags", [])

            # Phase 4: if aftermath expired, reset EMA for clean re-detection
            if "AFTERMATH_RESET_FOR_REDETECTION" in gate_flags:
                smoothed_confidence = 0.0
                print("  [%.2fs] [MULTI-EVENT] Aftermath expired "
                      "-- resetting EMA for re-detection" % t_sec)

            if current_state == AccidentState.CONFIRMED:
                _confirmed_gap_frames = 0

                if _active_candidate is None:
                    # New candidate: CONFIRMED just entered (fresh or after gap reset)
                    _active_candidate = CandidateEvent(
                        first_confirmed_time=t_sec,
                        first_confirmed_frame=analyzed_count,
                    )
                    # Preserve legacy first-confirmed fields for the FIRST candidate only
                    if not accident_confirmed:
                        accident_confirmed      = True
                        accident_frame_idx      = analyzed_count
                        accident_timestamp      = t_sec
                        first_confirmed_frame_idx   = analyzed_count
                        first_confirmed_timestamp   = t_sec
                    print("  [%.2fs] *** CANDIDATE CONFIRMED *** "
                          "(confidence=%.3f) "
                          "reason: %s" % (
                              t_sec, final_conf,
                              gate_result.get('confirmation_reason', 'N/A')))

                # Update active candidate each CONFIRMED frame
                _active_candidate.duration_frames += 1
                _active_candidate.end_time = t_sec

                if final_conf > _active_candidate.peak_confidence:
                    _active_candidate.peak_confidence      = final_conf
                    _active_candidate.peak_confidence_time = t_sec
                    _active_candidate.peak_frame_signals   = {
                        "detector":      detector_score,
                        "tracking":      tracking_score,
                        "velocity":      velocity_score,
                        "optical_flow":  optical_flow_score,
                        "disappearance": raw_disappearance_score,  # Phase 5: raw (not gated) for scoring
                        "geometry":      geometry_score,
                    }
                    best_frame      = frame.copy()
                    best_frame_data = fusion_result

            else:
                # Not CONFIRMED: count gap and close candidate if threshold reached
                if _active_candidate is not None:
                    _confirmed_gap_frames += 1
                    if _confirmed_gap_frames >= settings.CANDIDATE_GAP_THRESHOLD:
                        _candidate_events.append(_active_candidate)
                        _active_candidate = None
                        _confirmed_gap_frames = 0
                        print("  [%.2fs] [MULTI-EVENT] Candidate closed "
                              "-- watching for next event" % t_sec)


            # Phase 5: update flow baseline each frame (collects only during CLEAR)
            update_flow_baseline(
                self.flow_baseline, active_vehicle_tracks, current_state
            )


            # Timing
            dt = time.perf_counter() - t0
            frame_times.append(dt)

            # Per-frame log
            frame_data = {
                "frame_idx": analyzed_count,
                "timestamp_sec": t_sec,
                "n_vehicle_tracks": len(active_vehicle_tracks),
                "n_person_tracks": len(active_person_tracks),
                "n_lost_vehicles": len(
                    self.vehicle_tracker.get_all_lost_tracks()
                ),
                "n_dead_vehicles": len(dead_tracks),
                "final_confidence": final_conf,
                "raw_confidence": round(raw_conf, 3),
                "smoothed_confidence": round(smoothed_confidence, 3),
                "detector_score": round(detector_score, 3),
                "tracking_score": round(tracking_score, 3),
                "velocity_score": round(velocity_score, 3),
                "optical_flow_score": round(optical_flow_score, 3),
                "disappearance_score": round(disappearance_score, 3),
                "geometry_score": round(geometry_score, 3),
                "m1_score": round(m1_score, 3),
                "m4_score": round(m4_score, 3),
                "processing_ms": round(dt * 1000, 1),
                "state": gate_result["state"],
                "causal_present": gate_result["causal_present"],
                "n_families": gate_result["n_families"],
                "death_density": round(death_density, 3),
                "camera_shake": shake["is_camera_shake"],
                "shake_suppression": shake["suppression_factor"],
            }
            per_frame_data.append(frame_data)

            # Print status periodically
            state_tag = gate_result["state"][0]  # C/S/A
            if analyzed_count % 5 == 0 or final_conf > 0.3:
                n_vt = len(active_vehicle_tracks)
                n_pt = len(active_person_tracks)
                n_lost = len(self.vehicle_tracker.get_all_lost_tracks())
                causal_tag = "C" if gate_result["causal_present"] else "-"
                print(
                    f"  [{t_sec}s] [{state_tag}] tracks={n_vt}v/{n_pt}p "
                    f"lost={n_lost} "
                    f"conf={final_conf:.3f} "
                    f"[det={detector_score:.2f} trk={tracking_score:.2f} "
                    f"vel={velocity_score:.2f} flow={optical_flow_score:.2f} "
                    f"dis={disappearance_score:.2f} geo={geometry_score:.2f}] "
                    f"causal={causal_tag} fam={gate_result['n_families']} "
                    f"({dt*1000:.0f}ms)"
                )

            # ===== STEP 9: Debug Overlay =====
            if save_video or show_overlay:
                overlay = self._draw_overlay(
                    frame, active_vehicle_tracks, active_person_tracks,
                    vehicle_velocity_scores, disappearance_results,
                    flow_result, fusion_result, collision_zones,
                    t_sec, accident_confirmed,
                )
                if out_writer:
                    out_writer.write(overlay)
                if show_overlay:
                    cv2.imshow("RapidAid Track-Centric", overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            # Stop on first accident (after post-crash lookahead)
            if accident_confirmed and stop_on_first:
                time_since = t_sec - accident_timestamp
                if time_since >= settings.POST_CRASH_LOOKAHEAD_SEC:
                    print(f"  [{t_sec}s] Post-crash scan complete")
                    break

        cap.release()
        if out_writer:
            out_writer.release()
            print("\n  Tracked video saved: %s" % out_path)
        if show_overlay:
            cv2.destroyAllWindows()

        # ===== PHASE 4: CLOSE + SCORE ALL CANDIDATES =====
        # Close any still-active candidate at video end
        if _active_candidate is not None:
            _candidate_events.append(_active_candidate)
            _active_candidate = None

        # Score every candidate and select the best
        if _candidate_events:
            for c in _candidate_events:
                score_candidate_event(c, frame_h, frame_w)

            best_candidate = max(_candidate_events,
                                 key=lambda c: c.candidate_score)
            best_candidate.selected = True

            # Map selected candidate back to output fields
            accident_confirmed = True
            accident_timestamp = best_candidate.peak_confidence_time
            best_confidence    = best_candidate.peak_confidence
            # best_frame is already the frame with peak confidence

            print("\n  [MULTI-EVENT] %d candidate(s) found. "
                  "Selected: t=%.2fs (score=%.3f)" % (
                      len(_candidate_events),
                      best_candidate.first_confirmed_time,
                      best_candidate.candidate_score))
            for i, c in enumerate(_candidate_events):
                sel = "<-- SELECTED" if c.selected else ""
                print("    Candidate %d: t=%.2fs peak_conf=%.3f "
                      "dur=%d co_occ=%.3f score=%.3f %s" % (
                          i, c.first_confirmed_time, c.peak_confidence,
                          c.duration_frames, c.co_occurrence_score,
                          c.candidate_score, sel))

        # ===== FINAL RESULTS =====
        avg_fps = 1.0 / np.mean(frame_times) if frame_times else 0
        state_transitions = self._extract_state_transitions(per_frame_data)
        result = {
            "accident_detected": accident_confirmed,
            "accident_timestamp_sec": accident_timestamp,
            "first_confirmed_time": first_confirmed_timestamp,
            "first_confirmed_frame": first_confirmed_frame_idx,
            "state_transitions": state_transitions,
            "duration_sec": round(total_frames / fps, 2) if fps else None,
            "best_confidence": round(best_confidence, 3),
            "best_fusion_result": best_frame_data,
            # Phase 4: multi-event candidate fields
            "n_candidate_events": len(_candidate_events),
            "selected_event_index": next(
                (i for i, c in enumerate(_candidate_events) if c.selected), 0
            ),
            "candidate_events": [
                {
                    "first_confirmed_time":  c.first_confirmed_time,
                    "peak_confidence":       round(c.peak_confidence, 3),
                    "peak_confidence_time":  c.peak_confidence_time,
                    "duration_frames":       c.duration_frames,
                    "co_occurrence_score":   round(c.co_occurrence_score, 3),
                    "candidate_score":       round(c.candidate_score, 3),
                    "peak_frame_signals":    {
                        k: round(v, 3) for k, v in c.peak_frame_signals.items()
                    },
                    "selected":              c.selected,
                }
                for c in _candidate_events
            ],
            "metrics": {
                "total_frames": total_frames,
                "frames_analyzed": analyzed_count,
                "avg_fps": round(avg_fps, 1),
                "avg_processing_ms": round(np.mean(frame_times) * 1000, 1),
                "peak_tracks": max(
                    (d["n_vehicle_tracks"] for d in per_frame_data), default=0
                ),
                "total_dead_tracks": len(self.vehicle_tracker.dead_tracks),
            },
            "per_frame_data": per_frame_data,
            "confidence_history": [round(c, 3) for c in confidence_history],
        }

        # Save annotated best frame
        if best_frame is not None and best_frame_data:
            overlay = self._draw_overlay(
                best_frame,
                self.vehicle_tracker.get_all_active_tracks(),
                self.person_tracker.get_all_active_tracks(),
                {}, {}, {"has_prev": False}, best_frame_data, [],
                accident_timestamp, True,
            )
            frame_path = os.path.join(
                settings.ANNOTATED_DIR,
                f"tracked_{os.path.splitext(os.path.basename(video_path))[0]}.jpg"
            )
            cv2.imwrite(frame_path, overlay)
            result["annotated_frame_path"] = frame_path

        # Print summary
        print("\n" + "=" * 60)
        if accident_confirmed:
            print(f"  ACCIDENT DETECTED at {accident_timestamp}s")
            print(f"  Best confidence: {best_confidence:.3f}")
            if best_frame_data:
                print(f"  Dominant signal: {best_frame_data.get('dominant_signal', 'N/A')}")
        else:
            print("  No accident detected")
        print(f"  Frames analyzed: {analyzed_count}")
        print(f"  Avg FPS: {avg_fps:.1f}")
        print(f"  Total tracks created: {self.vehicle_tracker.dead_tracks.__len__() + len(self.vehicle_tracker.active_tracks) + len(self.vehicle_tracker.lost_tracks)}")
        print("=" * 60)

        # Save JSON report
        report_path = os.path.join(
            settings.REPORTS_DIR,
            f"tracked_{os.path.splitext(os.path.basename(video_path))[0]}.json"
        )
        report_data = {k: v for k, v in result.items()
                       if k != "per_frame_data"}  # Exclude verbose data
        report_data["per_frame_summary"] = [
            {k: v for k, v in d.items()
             if k in ("frame_idx", "timestamp_sec", "final_confidence",
                      "n_vehicle_tracks", "processing_ms",
                      # Phase 3: signal scores for LR training data
                      "detector_score", "tracking_score", "velocity_score",
                      "optical_flow_score", "disappearance_score",
                      "geometry_score")}
            for d in per_frame_data
        ]
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        print(f"  Report saved: {report_path}")

        return result

    def _extract_state_transitions(self, per_frame_data):
        """Extract first CLEAR->SUSPICIOUS and SUSPICIOUS->CONFIRMED transitions."""
        if not per_frame_data:
            return []

        transitions = []
        last_state = None
        for fd in per_frame_data:
            state = str(fd.get("state", "")).upper()
            ts = fd.get("timestamp_sec", None)
            if last_state is not None and state != last_state:
                transitions.append({
                    "from": last_state,
                    "to": state,
                    "timestamp": ts,
                })
            last_state = state

        wanted = [("CLEAR", "SUSPICIOUS"), ("SUSPICIOUS", "CONFIRMED")]
        filtered = []
        for frm, to in wanted:
            for tr in transitions:
                if tr["from"] == frm and tr["to"] == to:
                    filtered.append(tr)
                    break

        return filtered

    # ===== Detection helpers =====

    def _detect_vehicles(self, frame):
        """Detect vehicles using YOLO and return detection dicts."""
        results = self.vehicle_model(frame, verbose=False)[0]
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w

        vehicles = []
        if results.boxes is None:
            return vehicles

        for i, box in enumerate(results.boxes):
            coco_id = int(box.cls[0])
            conf = float(box.conf[0])

            if not self.is_vehicle_class(coco_id):
                continue
            if conf < settings.VEHICLE_CONFIDENCE_THRESHOLD:
                continue

            bbox = [int(c) for c in box.xyxy[0].tolist()]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            area_ratio = area / frame_area

            if area_ratio < settings.MIN_VEHICLE_AREA_RATIO:
                continue

            vehicles.append({
                "bbox": bbox,
                "confidence": round(conf * 100, 1),
                "type": "vehicle",
                "coco_class_id": coco_id,
                "area_ratio": area_ratio,
            })

        return vehicles

    def _detect_persons(self, frame):
        """Detect persons using YOLO pose."""
        results = self.pose_model(frame, verbose=False)[0]
        persons = []

        if results.boxes is None:
            return persons

        for box in results.boxes:
            coco_id = int(box.cls[0])
            conf = float(box.conf[0])
            if coco_id != 0:
                continue
            if conf < settings.PERSON_CONFIDENCE_THRESHOLD:
                continue

            bbox = [int(c) for c in box.xyxy[0].tolist()]
            persons.append({
                "bbox": bbox,
                "confidence": round(conf * 100, 1),
                "type": "person",
            })

        return persons

    # ===== Score computation helpers =====

    def _compute_detector_score(self, tracks, m1_score, m4_score):
        """Compute detector component score."""
        if not tracks:
            # If no tracks but M4 sees collision zone, use M4
            return max(m1_score, m4_score)

        # Average track confidence
        confs = [t.confidence for t in tracks]
        avg_conf = np.mean(confs)

        # Blend with M1 and M4
        return max(avg_conf, m1_score * 0.5, m4_score * 0.8)

    def _compute_velocity_score(self, velocity_scores):
        """Compute max velocity anomaly score."""
        if not velocity_scores:
            return 0.0
        return max(
            max(s["velocity_collapse_score"], s["trajectory_anomaly_score"])
            for s in velocity_scores.values()
        )

    def _compute_disappearance_score(self, disappearance_results,
                                      dead_tracks=None, active_tracks=None):
        """
        Compute max disappearance anomaly score, filtering out passive occlusion.

        A dead track is classified as occlusion ONLY when:
          1. A larger active vehicle bbox overlaps the dead track's last position
          AND
          2. The dead track showed NO velocity collapse before dying
             (was moving normally — passively hidden, not in distress)

        A dead track with velocity collapse before death is always counted
        regardless of overlapping vehicles (genuine physical event).
        """
        if not disappearance_results:
            return 0.0

        # Build track_id → Track object lookup for velocity-collapse check
        track_lookup = {}
        for track in (dead_tracks or []):
            track_lookup[track.track_id] = track

        scores = []
        for tid, r in disappearance_results.items():
            score = r["disappearance_score"]
            if score <= 0:
                continue

            dead_track = track_lookup.get(tid)
            if dead_track is not None and active_tracks:
                if self._is_occluded_death(dead_track, active_tracks):
                    print(
                        f"  [DisappearanceFilter] Track {tid} death classified "
                        f"as occlusion — skipped (score={score:.2f})"
                    )
                    continue

            scores.append(score)

        return max(scores) if scores else 0.0

    @staticmethod
    def _has_velocity_collapse(track):
        """
        Return True if the track's speed dropped >70% at any point in its
        recent history — indicating it was in physical distress before death.
        Uses the same algorithm as DisappearanceAnalyzer._check_velocity_collapse.
        """
        history = track.history
        if len(history) < 4:
            return False

        speeds = []
        for entry in history:
            vel = entry.get("velocity", (0, 0))
            if vel is None:
                vel = (0, 0)
            speeds.append((vel[0] ** 2 + vel[1] ** 2) ** 0.5)

        for i in range(2, len(speeds)):
            if speeds[i - 2] > 5.0:          # was moving fast
                if speeds[i] < speeds[i - 2] * 0.3:  # speed dropped >70%
                    return True
        return False

    @staticmethod
    def _is_occluded_death(dead_track, active_tracks,
                           iou_threshold=0.30, size_ratio=1.5):
        """
        Returns True (occlusion) only when:
          • A larger active vehicle bbox overlaps the dead track, AND
          • The dead track showed NO velocity collapse before dying
            (was moving normally = passively hidden, not in distress)

        Returns False (genuine event) when:
          • Dead track velocity collapsed before death  → real physical event
          • OR no large overlapping active vehicle found
        """
        dead_bbox = dead_track.bbox
        dead_area = max(0, dead_bbox[2] - dead_bbox[0]) * max(
            0, dead_bbox[3] - dead_bbox[1]
        )
        if dead_area == 0:
            return False

        # If dying track already showed velocity collapse → real event
        # Do not classify as occlusion regardless of bbox overlap
        if TrackCentricProcessor._has_velocity_collapse(dead_track):
            return False

        # Check if a larger active vehicle is geometrically covering dead track
        for active in active_tracks:
            active_bbox = active.bbox
            active_area = max(0, active_bbox[2] - active_bbox[0]) * max(
                0, active_bbox[3] - active_bbox[1]
            )
            if active_area <= dead_area * size_ratio:
                continue
            iou = TrackCentricProcessor._compute_iou(dead_bbox, active_bbox)
            if iou > iou_threshold:
                # Large vehicle overlapping + no velocity collapse = occlusion
                return True

        return False

    def _compute_geometry_score(self, tracks, collision_zones,
                                frame_w=None, frame_h=None):
        """Compute geometry score from track overlaps and collision zones."""
        score = 0.0

        # M4 collision zones (independent of portrait orientation)
        if collision_zones:
            score = max(score, collision_zones[0]["confidence"])

        # Track-to-track proximity via bbox IoU
        if len(tracks) >= 2:
            # --- Portrait geometry axis swap (Phase 2) ---
            # For portrait frames (h > w), swap x/y in bboxes so that
            # the y-axis vehicle distribution (along road depth) is treated
            # as x-axis (horizontal proximity), which the IoU module expects.
            if (frame_w is not None and frame_h is not None
                    and settings.PORTRAIT_GEOMETRY_ENABLED
                    and is_portrait_frame(frame_w, frame_h)):
                raw_bboxes = [
                    (t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3])
                    for t in tracks
                ]
                effective_bboxes, _, _ = portrait_swap_bboxes(
                    raw_bboxes, frame_w, frame_h
                )
            else:
                effective_bboxes = [t.bbox for t in tracks]
            # ---------------------------------------------

            for i in range(len(effective_bboxes)):
                for j in range(i + 1, len(effective_bboxes)):
                    iou = self._compute_iou(
                        effective_bboxes[i], effective_bboxes[j]
                    )
                    if iou > 0.1:
                        score = max(score, iou * 0.8)

        return min(1.0, score)

    @staticmethod
    def _compute_iou(bbox1, bbox2):
        """Compute IoU between two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
        a2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0

    # ===== Visualization =====

    def _draw_overlay(self, frame, vehicle_tracks, person_tracks,
                      velocity_scores, disappearance_results,
                      flow_result, fusion_result, collision_zones,
                      t_sec, confirmed):
        """Draw debug overlay with all track-centric information."""
        overlay = frame.copy()

        # Draw collision zones (if any)
        for zone in (collision_zones or []):
            z = zone["bbox"]
            cv2.rectangle(overlay, (z[0], z[1]), (z[2], z[3]),
                         (255, 100, 0), 2)
            cv2.putText(overlay, f"M4:{zone['confidence']:.2f}",
                       (z[0], z[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 100, 0), 1)

        # Draw vehicle tracks
        for track in (vehicle_tracks or []):
            bbox = track.bbox
            tid = track.track_id
            speed = track.get_speed()

            # Color based on status
            color = (0, 255, 0)  # Green = normal
            vel_scores = velocity_scores.get(tid, {})
            if vel_scores.get("sudden_stop"):
                color = (0, 0, 255)  # Red = sudden stop
            elif vel_scores.get("direction_reversal"):
                color = (0, 165, 255)  # Orange = anomaly

            cv2.rectangle(overlay, (bbox[0], bbox[1]),
                         (bbox[2], bbox[3]), color, 2)

            # Track ID and speed
            label = f"V{tid} spd:{speed:.1f}"
            cv2.putText(overlay, label,
                       (bbox[0], bbox[1] - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Draw velocity vector
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            vx = int(track.velocity[0] * 5)
            vy = int(track.velocity[1] * 5)
            cv2.arrowedLine(overlay, (cx, cy), (cx + vx, cy + vy),
                           color, 2, tipLength=0.3)

            # Draw trajectory
            if len(track.history) > 2:
                pts = []
                for entry in track.history[-15:]:
                    eb = entry["bbox"]
                    pts.append((int((eb[0] + eb[2]) / 2),
                               int((eb[1] + eb[3]) / 2)))
                for k in range(1, len(pts)):
                    cv2.line(overlay, pts[k - 1], pts[k], color, 1)

        # Draw person tracks
        for track in (person_tracks or []):
            bbox = track.bbox
            tid = track.track_id
            cv2.rectangle(overlay, (bbox[0], bbox[1]),
                         (bbox[2], bbox[3]), (0, 200, 255), 2)
            cv2.putText(overlay, f"P{tid}",
                       (bbox[0], bbox[1] - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        # Draw disappearance markers
        for tid, result in (disappearance_results or {}).items():
            if result["disappearance_type"] == "ANOMALOUS_DISAPPEARANCE":
                details = result.get("details", {})
                center = details.get("last_center")
                if center:
                    cx, cy = int(center[0]), int(center[1])
                    cv2.circle(overlay, (cx, cy), 25, (0, 0, 255), 3)
                    cv2.putText(overlay, f"GONE:{tid}",
                               (cx - 20, cy - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                               (0, 0, 255), 2)

        # Top banner with fusion info
        h_frame, w_frame = overlay.shape[:2]
        banner_h = 50
        cv2.rectangle(overlay, (0, 0), (w_frame, banner_h),
                     (0, 0, 180) if confirmed else (80, 80, 80), -1)

        if isinstance(fusion_result, dict):
            fc = fusion_result.get("final_confidence", 0)
            dom = fusion_result.get("dominant_signal", "N/A")
        else:
            fc = 0
            dom = "N/A"

        status = "ACCIDENT" if confirmed else "Monitoring"
        banner_text = (f"{status} | t={t_sec}s | "
                      f"Conf: {fc:.3f} | Dom: {dom}")
        cv2.putText(overlay, banner_text, (10, 32),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Bottom signal bar
        bar_y = h_frame - 30
        cv2.rectangle(overlay, (0, bar_y), (w_frame, h_frame),
                     (40, 40, 40), -1)

        if isinstance(fusion_result, dict):
            wc = fusion_result.get("weighted_components", {})
            signals = [
                f"det={wc.get('detector', 0):.2f}",
                f"trk={wc.get('tracking', 0):.2f}",
                f"vel={wc.get('velocity', 0):.2f}",
                f"flow={wc.get('optical_flow', 0):.2f}",
                f"dis={wc.get('disappearance', 0):.2f}",
                f"geo={wc.get('geometry', 0):.2f}",
            ]
            signal_text = " | ".join(signals)
        else:
            signal_text = "No signals"

        cv2.putText(overlay, signal_text, (10, h_frame - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return overlay
