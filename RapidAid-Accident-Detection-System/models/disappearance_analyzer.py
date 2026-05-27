"""
RapidAid — Disappearance Analyzer (Phase 5)

Analyzes tracked objects that have disappeared to classify them
into three categories:

  1. NORMAL_EXIT: Object left the frame naturally
     - Near frame border, velocity directed outward, smooth motion

  2. TEMP_OCCLUSION: Object is temporarily hidden
     - Tracker maintains memory, expects reappearance

  3. ANOMALOUS_DISAPPEARANCE: Object vanished suspiciously
     - Disappeared mid-frame (not near border)
     - Collision nearby, velocity collapse occurred
     - Overlap with truck/large vehicle
     - Sudden loss after impact

CRITICAL PRINCIPLE: A disappearing vehicle IS an accident signal.
Vehicles don't vanish in mid-frame without reason.

Usage:
    analyzer = DisappearanceAnalyzer(frame_w=1920, frame_h=1080)
    results = analyzer.analyze(dead_tracks, lost_tracks, active_tracks, frame_idx)
"""
import math
import numpy as np


class DisappearanceType:
    NORMAL_EXIT = "NORMAL_EXIT"
    TEMP_OCCLUSION = "TEMP_OCCLUSION"
    ANOMALOUS = "ANOMALOUS_DISAPPEARANCE"


class DisappearanceAnalyzer:
    """
    Classifies track disappearances into normal, occlusion, or anomalous.

    Anomalous disappearances are accident signals — a vehicle that
    vanishes mid-frame after impact likely indicates:
      - Vehicle crushed/flipped behind another
      - Vehicle went under a truck
      - Vehicle flew off road/bridge
      - Vehicle obscured by debris/smoke from impact
    """

    def __init__(self, frame_w=1920, frame_h=1080,
                 border_margin_ratio=0.08,
                 occlusion_max_frames=20,
                 min_speed_for_exit=3.0,
                 overlap_threshold=0.2):
        """
        Args:
            frame_w, frame_h: frame dimensions
            border_margin_ratio: fraction of frame size considered "near border"
            occlusion_max_frames: max lost frames before occlusion→anomalous
            min_speed_for_exit: min speed to consider outward motion
            overlap_threshold: min overlap with active vehicle for occlusion
        """
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.border_margin_ratio = border_margin_ratio
        self.occlusion_max_frames = occlusion_max_frames
        self.min_speed_for_exit = min_speed_for_exit
        self.overlap_threshold = overlap_threshold

    def update_frame_size(self, frame_w, frame_h):
        """Update frame dimensions (call when processing new video)."""
        self.frame_w = frame_w
        self.frame_h = frame_h

    def analyze(self, dead_tracks, lost_tracks=None, active_tracks=None,
                frame_idx=0):
        """
        Analyze all dead and lost tracks for disappearance classification.

        Args:
            dead_tracks: list of Track objects that have died
            lost_tracks: list of Track objects currently lost
            active_tracks: list of Track objects currently active (for overlap check)
            frame_idx: current frame index

        Returns:
            dict mapping track_id -> {
                'disappearance_type': str,
                'disappearance_score': float (0=normal, 1=very anomalous),
                'reason': str,
                'details': dict,
            }
        """
        results = {}

        all_disappeared = list(dead_tracks or [])
        if lost_tracks:
            all_disappeared.extend(lost_tracks)

        active = active_tracks or []

        for track in all_disappeared:
            result = self._classify_disappearance(track, active, frame_idx)
            results[track.track_id] = result

        return results

    def _classify_disappearance(self, track, active_tracks, frame_idx):
        """Classify a single track's disappearance."""
        bbox = track.bbox
        center = track.last_known_center
        velocity = track.last_known_velocity
        speed = track.get_speed()
        history = track.history
        frames_missing = frame_idx - track.last_seen_frame

        # Border margins
        margin_x = self.frame_w * self.border_margin_ratio
        margin_y = self.frame_h * self.border_margin_ratio

        # Check 1: Is the object near the frame border?
        near_left = bbox[0] < margin_x
        near_right = bbox[2] > self.frame_w - margin_x
        near_top = bbox[1] < margin_y
        near_bottom = bbox[3] > self.frame_h - margin_y
        near_border = near_left or near_right or near_top or near_bottom

        # Check 2: Is velocity directed outward?
        outward_velocity = False
        if speed >= self.min_speed_for_exit:
            if near_left and velocity[0] < 0:
                outward_velocity = True
            elif near_right and velocity[0] > 0:
                outward_velocity = True
            elif near_top and velocity[1] < 0:
                outward_velocity = True
            elif near_bottom and velocity[1] > 0:
                outward_velocity = True

        # Check 3: Was the motion smooth before disappearance?
        smooth_motion = self._check_smooth_motion(history)

        # Check 4: Is the disappearance point overlapping with active vehicles?
        occluded_by = self._check_occlusion(track, active_tracks)

        # Check 5: Was there a velocity collapse before disappearance?
        had_velocity_collapse = self._check_velocity_collapse(history)

        # Check 6: Was the track short-lived (possibly a detection artifact)?
        is_short_track = track.total_visible_frames < 3

        # ============ Classification Logic ============

        # NORMAL_EXIT: near border + outward velocity + smooth motion
        if near_border and outward_velocity and smooth_motion:
            return {
                "disappearance_type": DisappearanceType.NORMAL_EXIT,
                "disappearance_score": 0.0,
                "reason": "Object exited frame normally",
                "details": {
                    "near_border": True,
                    "outward_velocity": True,
                    "smooth_motion": True,
                    "border_side": self._get_border_side(bbox),
                },
            }

        # TEMP_OCCLUSION: hidden behind another active object
        if occluded_by and frames_missing < self.occlusion_max_frames:
            return {
                "disappearance_type": DisappearanceType.TEMP_OCCLUSION,
                "disappearance_score": 0.1,
                "reason": f"Occluded by track {occluded_by.track_id}",
                "details": {
                    "occluded_by_track": occluded_by.track_id,
                    "occluded_by_type": occluded_by.obj_type,
                    "frames_missing": frames_missing,
                },
            }

        # Very short track near border — likely a detection artifact
        if is_short_track and near_border:
            return {
                "disappearance_type": DisappearanceType.NORMAL_EXIT,
                "disappearance_score": 0.05,
                "reason": "Short-lived detection near frame edge",
                "details": {"total_visible_frames": track.total_visible_frames},
            }

        # ANOMALOUS_DISAPPEARANCE: everything else
        score = self._compute_anomaly_score(
            track, near_border, outward_velocity, smooth_motion,
            occluded_by, had_velocity_collapse, frames_missing
        )

        reasons = []
        if not near_border:
            reasons.append("disappeared mid-frame")
        if had_velocity_collapse:
            reasons.append("velocity collapse before disappearance")
        if not smooth_motion:
            reasons.append("erratic motion before disappearance")
        if occluded_by and frames_missing >= self.occlusion_max_frames:
            reasons.append(f"prolonged occlusion ({frames_missing} frames)")

        return {
            "disappearance_type": DisappearanceType.ANOMALOUS,
            "disappearance_score": round(score, 3),
            "reason": "; ".join(reasons) if reasons else "unexplained disappearance",
            "details": {
                "near_border": near_border,
                "outward_velocity": outward_velocity,
                "smooth_motion": smooth_motion,
                "had_velocity_collapse": had_velocity_collapse,
                "frames_missing": frames_missing,
                "last_speed": round(speed, 2),
                "last_center": track.last_known_center,
                "last_bbox": list(bbox),   # needed by occlusion death filter
            },
        }

    def _compute_anomaly_score(self, track, near_border, outward_velocity,
                                smooth_motion, occluded_by, velocity_collapse,
                                frames_missing):
        """
        Compute anomaly score (0 = definitely normal, 1 = highly anomalous).
        """
        score = 0.0

        # Mid-frame disappearance is the strongest signal
        if not near_border:
            score += 0.4

        # Velocity collapse before disappearance
        if velocity_collapse:
            score += 0.25

        # Erratic motion before disappearance
        if not smooth_motion:
            score += 0.10

        # No outward velocity when near border (stopped at edge = suspicious)
        if near_border and not outward_velocity:
            score += 0.10

        # Long-duration disappearance with no occlusion
        if not occluded_by and frames_missing > 10:
            score += 0.10

        # Track was visible for many frames (established track suddenly vanishes)
        if track.total_visible_frames > 10:
            score += 0.05

        return min(1.0, score)

    def _check_smooth_motion(self, history):
        """Check if the object had smooth motion before disappearing."""
        if len(history) < 4:
            return True  # Not enough data to judge

        # Look at the last 6 velocity entries
        recent = history[-6:]
        velocities = [
            entry.get("velocity", (0, 0))
            for entry in recent
            if entry.get("velocity") is not None
        ]

        if len(velocities) < 3:
            return True

        # Compute direction changes
        directions = []
        for vx, vy in velocities:
            speed = (vx ** 2 + vy ** 2) ** 0.5
            if speed > 1.0:
                directions.append(math.degrees(math.atan2(vy, vx)))

        if len(directions) < 3:
            return True

        # Check for large direction changes
        max_change = 0
        for i in range(1, len(directions)):
            delta = abs(directions[i] - directions[i - 1])
            if delta > 180:
                delta = 360 - delta
            max_change = max(max_change, delta)

        return max_change < 60  # Less than 60 degrees = smooth

    def _check_velocity_collapse(self, history):
        """Check if the object had a sudden speed drop before disappearing."""
        if len(history) < 4:
            return False

        speeds = []
        for entry in history:
            vel = entry.get("velocity", (0, 0))
            if vel is None:
                vel = (0, 0)
            speed = (vel[0] ** 2 + vel[1] ** 2) ** 0.5
            speeds.append(speed)

        if len(speeds) < 4:
            return False

        # Check if any speed drop > 70% occurred
        for i in range(2, len(speeds)):
            if speeds[i - 2] > 5.0:  # Was moving fast
                if speeds[i] < speeds[i - 2] * 0.3:  # Speed dropped > 70%
                    return True

        return False

    def _check_occlusion(self, disappeared_track, active_tracks):
        """
        Check if the disappeared track's last position overlaps
        with any active track (suggesting occlusion).
        """
        if not active_tracks:
            return None

        d_bbox = disappeared_track.bbox
        for track in active_tracks:
            if track.track_id == disappeared_track.track_id:
                continue

            a_bbox = track.bbox
            # Compute overlap
            x1 = max(d_bbox[0], a_bbox[0])
            y1 = max(d_bbox[1], a_bbox[1])
            x2 = min(d_bbox[2], a_bbox[2])
            y2 = min(d_bbox[3], a_bbox[3])

            inter = max(0, x2 - x1) * max(0, y2 - y1)
            d_area = max(1, (d_bbox[2] - d_bbox[0]) * (d_bbox[3] - d_bbox[1]))

            if inter / d_area > self.overlap_threshold:
                return track

        return None

    def _get_border_side(self, bbox):
        """Determine which border side the object is near."""
        margin_x = self.frame_w * self.border_margin_ratio
        margin_y = self.frame_h * self.border_margin_ratio

        sides = []
        if bbox[0] < margin_x:
            sides.append("left")
        if bbox[2] > self.frame_w - margin_x:
            sides.append("right")
        if bbox[1] < margin_y:
            sides.append("top")
        if bbox[3] > self.frame_h - margin_y:
            sides.append("bottom")
        return sides if sides else ["none"]

    def get_anomalous_disappearances(self, results, threshold=0.3):
        """
        Filter for anomalous disappearances above threshold.

        Returns: list of (track_id, result_dict) tuples sorted by score
        """
        anomalous = [
            (tid, r) for tid, r in results.items()
            if r["disappearance_type"] == DisappearanceType.ANOMALOUS
            and r["disappearance_score"] >= threshold
        ]
        anomalous.sort(key=lambda x: x[1]["disappearance_score"], reverse=True)
        return anomalous
