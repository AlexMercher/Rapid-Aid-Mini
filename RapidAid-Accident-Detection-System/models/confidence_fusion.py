"""
RapidAid — Confidence Fusion Engine (Phase 7 + Phase 3 learned mode)

Replaces hardcoded heuristic confidence logic with a weighted
multi-signal fusion system.

Final confidence combines:
  - Detector confidence (YOLO detection quality)
  - Tracking consistency (how stable are the tracks)
  - Velocity anomaly (sudden stops, direction changes)
  - Optical flow anomaly (motion burst, flow patterns)
  - Disappearance score (anomalous vanishing)
  - Geometric overlap (vehicle-vehicle overlap signals)
  - Temporal consistency (M3 temporal classifier + detection window)

All weights are configurable and can be tuned per-scenario.

Phase 3 addition: FUSION_MODE = "learned" routes through a logistic
regression model trained on GT-labeled per-frame signals. Captures
co-occurrence patterns (geometry AND velocity simultaneously) that a
weighted average treats independently.

Usage:
    fusion = ConfidenceFusion()
    final = fusion.compute(
        detector_score=0.85,
        tracking_score=0.7,
        velocity_score=0.5,
        optical_flow_score=0.6,
        disappearance_score=0.3,
        geometry_score=0.8,
    )
"""
import os
import pickle
import numpy as np

from config import settings


class ConfidenceFusion:
    """
    Multi-signal confidence fusion for accident detection.

    Instead of relying on any single signal (like YOLO confidence alone),
    this combines multiple independent signals to produce a robust
    final accident confidence score.
    """

    # Default fusion weights
    # NOTE: disappearance weight reduced from 0.15 → 0.05 to dampen
    # occlusion false positives (large vehicle occluding tracked vehicle
    # causes spurious disappearance spike). Still retains 0.05 so genuine
    # track destruction events (truck overturn, car compression) are counted.
    # Full reintroduction deferred until occlusion death filter is complete.
    # Controlled by DISAPPEARANCE_WEIGHT in settings.py.
    DEFAULT_WEIGHTS = {
        "detector":       0.2235,   # was 0.20 → scaled by (0.95/0.85)
        "tracking":       0.2235,   # was 0.20 → scaled by (0.95/0.85)
        "velocity":       0.1676,   # was 0.15 → scaled by (0.95/0.85)
        "optical_flow":   0.1676,   # was 0.15 → scaled by (0.95/0.85)
        "disappearance":  0.05,     # reduced — occlusion FP dampening
        "geometry":       0.1676,   # was 0.15 → scaled by (0.95/0.85)
    }

    # Feature order for LR model input (must match training)
    _LR_FEATURES = ["detector", "tracking", "velocity", "optical_flow", "disappearance", "geometry"]

    def __init__(self, weights=None):
        """
        Args:
            weights: dict of signal_name -> weight. Must sum to ~1.0.
                     Uses DEFAULT_WEIGHTS if None.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

        if "geometry" not in self.weights:
            other_total = sum(self.weights.values())
            if other_total > 0:
                scale = (1.0 - 0.15) / other_total
                for k in self.weights:
                    self.weights[k] *= scale
            self.weights["geometry"] = 0.15

        # Validate weights sum
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            print(f"[ConfidenceFusion] WARNING: weights sum to {total:.3f}, "
                  f"normalizing to 1.0")
            for k in self.weights:
                self.weights[k] /= total

        # ── Phase 3: Load learned LR model if available ─────────────────
        self._lr_bundle = None
        if settings.FUSION_MODE == "learned":
            model_path = settings.FUSION_MODEL_PATH
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    self._lr_bundle = pickle.load(f)
                # Quick validation
                expected = self._LR_FEATURES
                loaded = self._lr_bundle.get("feature_names", expected)
                if loaded != expected:
                    print(f"[ConfidenceFusion] WARN: model feature order mismatch. "
                          f"Expected {expected}, got {loaded}. Falling back to weighted.")
                    self._lr_bundle = None
            else:
                print(f"[ConfidenceFusion] WARN: learned mode requested but model "
                      f"not found at {model_path}. Falling back to weighted average.")

    def compute(self, detector_score=0.0, tracking_score=0.0,
                velocity_score=0.0, optical_flow_score=0.0,
                disappearance_score=0.0, geometry_score=0.0,
                temporal_score=0.0):
        """
        Compute fused accident confidence.

        All input scores should be in [0.0, 1.0].

        Args:
            detector_score: YOLO detection confidence (avg across involved vehicles)
            tracking_score: track stability and consistency score
            velocity_score: max(velocity_collapse, trajectory_anomaly)
            optical_flow_score: max(motion_burst, optical_flow_anomaly)
            disappearance_score: anomalous disappearance score
            geometry_score: geometric overlap / crash score
            temporal_score: M3 temporal classifier bonus (added as extra signal)

        Returns:
            dict with:
                'final_confidence': float (0-1)
                'weighted_components': dict of signal -> weighted contribution
                'raw_scores': dict of signal -> raw input score
                'dominant_signal': str (which signal contributed most)
        """
        # Build signals dict for dispatch
        signals = {
            "detector":      detector_score,
            "tracking":      tracking_score,
            "velocity":      velocity_score,
            "optical_flow":  optical_flow_score,
            "disappearance": disappearance_score,
            "geometry":      geometry_score,
        }

        if settings.FUSION_MODE == "learned" and self._lr_bundle is not None:
            return self._compute_learned(signals, temporal_score)
        return self._compute_weighted(signals, temporal_score)

    # ── Phase 3: Learned mode ────────────────────────────────────────────

    def _compute_learned(self, signals: dict, temporal_score: float = 0.0) -> dict:
        """
        Logistic regression fusion — captures co-occurrence patterns.

        geometry=0.8 AND velocity=0.9 simultaneously produces a much higher
        probability than either alone. The weighted average cannot distinguish
        this from two independent moderate signals.

        Temporal bonus is added AFTER LR output (as in weighted mode) to
        preserve the independent temporal signal structure.
        """
        x = np.array([[self._get_signal(signals, k) for k in self._LR_FEATURES]])
        x_scaled = self._lr_bundle["scaler"].transform(x)
        prob = float(self._lr_bundle["model"].predict_proba(x_scaled)[0][1])

        # Add temporal bonus (same treatment as weighted mode)
        temporal_bonus = np.clip(temporal_score, 0, 1) * 0.10
        final = min(1.0, prob + temporal_bonus)

        # Build component breakdown (parallel to weighted format)
        weighted = {k: round(self._get_signal(signals, k) *
                              self.weights.get(k, 0), 4)
                    for k in self.weights}
        weighted["temporal_bonus"] = round(temporal_bonus, 4)
        dominant = max(weighted.items(), key=lambda x: x[1])

        raw_scores = {k: round(np.clip(v, 0, 1), 3) for k, v in signals.items()}

        return {
            "final_confidence": round(final, 3),
            "weighted_components": weighted,
            "raw_scores": raw_scores,
            "dominant_signal": dominant[0],
            "temporal_score": round(np.clip(temporal_score, 0, 1), 3),
            "fusion_mode": "learned",
        }

    @staticmethod
    def _get_signal(signals: dict, key: str) -> float:
        """Resolve signal value, trying _score suffix variants."""
        val = (signals.get(key)
               or signals.get(f"{key}_score")
               or signals.get(f"{key}Score"))
        return float(val) if val is not None else 0.0

    # ── Original weighted average mode (renamed from compute()) ─────────

    def _compute_weighted(self, signals: dict, temporal_score: float = 0.0) -> dict:
        """
        Compute fused accident confidence using hand-tuned weighted average.

        This is the ORIGINAL compute() logic moved verbatim into _compute_weighted().
        DO NOT change any logic inside this method.
        """
        # Clamp all inputs to [0, 1]
        scores = {
            "detector":      np.clip(signals.get("detector", 0), 0, 1),
            "tracking":      np.clip(signals.get("tracking", 0), 0, 1),
            "velocity":      np.clip(signals.get("velocity", 0), 0, 1),
            "optical_flow":  np.clip(signals.get("optical_flow", 0), 0, 1),
            "disappearance": np.clip(signals.get("disappearance", 0), 0, 1),
            "geometry":      np.clip(signals.get("geometry", 0), 0, 1),
        }

        # Compute weighted sum
        weighted = {}
        final = 0.0
        for signal, score in scores.items():
            w = self.weights.get(signal, 0)
            contribution = score * w
            weighted[signal] = round(contribution, 4)
            final += contribution

        # Add temporal bonus (not part of base weights, acts as multiplier)
        temporal_bonus = np.clip(temporal_score, 0, 1) * 0.10
        final += temporal_bonus
        weighted["temporal_bonus"] = round(temporal_bonus, 4)

        # Clamp final
        final = min(1.0, final)

        # Find dominant signal
        dominant = max(weighted.items(), key=lambda x: x[1])

        return {
            "final_confidence": round(final, 3),
            "weighted_components": weighted,
            "raw_scores": {k: round(v, 3) for k, v in scores.items()},
            "dominant_signal": dominant[0],
            "temporal_score": round(np.clip(temporal_score, 0, 1), 3),
        }

    def compute_tracking_score(self, active_tracks, involved_track_ids=None):
        """
        Compute tracking consistency score.

        A high score means:
          - Involved tracks have been visible for many frames
          - Track IDs are stable (not constantly changing)
          - Tracks have smooth, consistent detections

        Args:
            active_tracks: list of Track objects
            involved_track_ids: set of track IDs flagged as involved

        Returns:
            float: 0.0 to 1.0
        """
        if not active_tracks:
            return 0.0

        if involved_track_ids:
            relevant = [t for t in active_tracks
                       if t.track_id in involved_track_ids]
        else:
            relevant = active_tracks

        if not relevant:
            return 0.0

        scores = []
        for track in relevant:
            # Longevity: longer tracks are more reliable
            longevity = min(1.0, track.total_visible_frames / 15.0)

            # Consistency: low frames_missing means stable tracking
            consistency = 1.0 - min(1.0, track.frames_missing / 10.0)

            # Confidence stability: high detection confidence
            conf = track.confidence
            if conf > 1.0:
                conf /= 100.0
            conf_score = min(1.0, conf / 0.5)

            track_score = (longevity * 0.4 + consistency * 0.3 +
                          conf_score * 0.3)
            scores.append(track_score)

        return float(np.mean(scores))

    def compute_detector_score(self, involved_vehicles):
        """
        Compute aggregate detector confidence from involved vehicles.

        Args:
            involved_vehicles: list of vehicle dicts with 'confidence' field

        Returns:
            float: 0.0 to 1.0
        """
        if not involved_vehicles:
            return 0.0

        confs = []
        for v in involved_vehicles:
            conf = v.get("confidence", 0)
            if conf > 1.0:
                conf /= 100.0
            confs.append(conf)

        # Use geometric mean — penalizes having one low-confidence detection
        if len(confs) == 1:
            return confs[0]
        else:
            return float(np.mean(confs))

    def compute_geometry_score(self, involved_vehicles):
        """
        Compute geometry score from crash scores.

        Args:
            involved_vehicles: list of vehicle dicts with 'crash_score' field

        Returns:
            float: 0.0 to 1.0
        """
        if not involved_vehicles:
            return 0.0

        crash_scores = [v.get("crash_score", 0) for v in involved_vehicles]
        return float(min(1.0, max(crash_scores)))

    def adaptive_weights(self, available_signals):
        """
        Adapt weights when some signals are unavailable.

        Redistributes weight from missing signals to available ones
        proportionally.

        Args:
            available_signals: set of signal names that have valid data

        Returns:
            dict of adjusted weights
        """
        adjusted = {}
        missing_weight = 0.0

        for signal, weight in self.weights.items():
            if signal in available_signals:
                adjusted[signal] = weight
            else:
                missing_weight += weight

        # Redistribute missing weight
        if adjusted and missing_weight > 0:
            scale = 1.0 / (1.0 - missing_weight)
            for signal in adjusted:
                adjusted[signal] *= scale

        return adjusted
