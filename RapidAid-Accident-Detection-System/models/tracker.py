"""
RapidAid — Multi-Object Tracker (Phase 3)

Integrates ByteTrack-style multi-object tracking for persistent
vehicle and pedestrian IDs across video frames.

Key features:
  - Persistent object IDs surviving temporary occlusion
  - Track lifecycle management (active → lost → dead)
  - Per-track state: bbox, velocity, acceleration, history
  - IoU-based association with Kalman-style prediction

This module does NOT modify any existing files. It provides a
TrackManager class that can be layered on top of existing detectors.

Usage:
    tracker = TrackManager(max_lost_frames=15)
    for frame_idx, detections in enumerate(all_detections):
        active_tracks = tracker.update(detections, frame_idx)
"""
import numpy as np
from collections import defaultdict


class Track:
    """Single object track with temporal state."""

    _next_id = 1

    def __init__(self, bbox, frame_idx, obj_type="vehicle", confidence=0.0,
                 extra_data=None):
        """
        Initialize a new track.

        Args:
            bbox: [x1, y1, x2, y2]
            frame_idx: frame index where track was created
            obj_type: 'vehicle' or 'person'
            confidence: detection confidence
            extra_data: dict of additional data to carry (class, polygon, etc.)
        """
        self.track_id = Track._next_id
        Track._next_id += 1

        self.bbox = list(bbox)
        self.obj_type = obj_type
        self.confidence = confidence
        self.extra_data = extra_data or {}

        # Lifecycle
        self.status = "active"  # active | lost | dead
        self.created_frame = frame_idx
        self.last_seen_frame = frame_idx
        self.frames_missing = 0
        self.total_visible_frames = 1

        # History
        self.history = [{"frame": frame_idx, "bbox": list(bbox),
                         "confidence": confidence}]

        # Velocity & acceleration (pixels per frame)
        self.velocity = (0.0, 0.0)
        self.acceleration = (0.0, 0.0)
        self._prev_velocity = (0.0, 0.0)

        # Disappearance tracking
        self.disappearance_type = None  # Set when track transitions to dead
        self.last_known_center = self._center(bbox)
        self.last_known_velocity = (0.0, 0.0)

    @staticmethod
    def _center(bbox):
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    @staticmethod
    def _area(bbox):
        return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

    def update(self, bbox, frame_idx, confidence=0.0, extra_data=None):
        """Update track with new detection."""
        old_center = self._center(self.bbox)
        new_center = self._center(bbox)

        # Compute frame gap (handles skipped frames)
        frame_gap = max(1, frame_idx - self.last_seen_frame)

        # Update velocity (pixels per frame)
        self._prev_velocity = self.velocity
        self.velocity = (
            (new_center[0] - old_center[0]) / frame_gap,
            (new_center[1] - old_center[1]) / frame_gap,
        )

        # Update acceleration
        self.acceleration = (
            (self.velocity[0] - self._prev_velocity[0]) / frame_gap,
            (self.velocity[1] - self._prev_velocity[1]) / frame_gap,
        )

        self.bbox = list(bbox)
        self.confidence = confidence
        self.last_seen_frame = frame_idx
        self.frames_missing = 0
        self.status = "active"
        self.total_visible_frames += 1
        self.last_known_center = new_center
        self.last_known_velocity = self.velocity

        if extra_data:
            self.extra_data.update(extra_data)

        self.history.append({
            "frame": frame_idx,
            "bbox": list(bbox),
            "confidence": confidence,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
        })

        # Cap history to prevent memory bloat
        if len(self.history) > 120:
            self.history = self.history[-120:]

    def predict_bbox(self, frame_idx):
        """
        Predict bbox at a future frame using last known velocity.
        Used for association when the object is temporarily lost.
        """
        dt = frame_idx - self.last_seen_frame
        cx, cy = self.last_known_center
        pred_cx = cx + self.velocity[0] * dt
        pred_cy = cy + self.velocity[1] * dt

        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]

        return [
            pred_cx - w / 2,
            pred_cy - h / 2,
            pred_cx + w / 2,
            pred_cy + h / 2,
        ]

    def mark_missing(self, frame_idx):
        """Mark track as missing for one frame."""
        self.frames_missing = frame_idx - self.last_seen_frame
        if self.status == "active":
            self.status = "lost"

    def get_speed(self):
        """Get current speed (magnitude of velocity) in pixels per frame."""
        return (self.velocity[0] ** 2 + self.velocity[1] ** 2) ** 0.5

    def get_direction(self):
        """Get current direction angle in degrees (0=right, 90=down)."""
        if self.velocity == (0.0, 0.0):
            return 0.0
        return np.degrees(np.arctan2(self.velocity[1], self.velocity[0]))

    def get_state(self):
        """Get serializable track state dict."""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "history": self.history[-10:],  # Last 10 entries
            "last_seen_frame": self.last_seen_frame,
            "frames_missing": self.frames_missing,
            "status": self.status,
            "obj_type": self.obj_type,
            "confidence": self.confidence,
            "speed": round(self.get_speed(), 2),
            "direction": round(self.get_direction(), 1),
            "total_visible_frames": self.total_visible_frames,
        }


def compute_iou(bbox1, bbox2):
    """Compute IoU between two bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
    area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


class TrackManager:
    """
    ByteTrack-style multi-object tracker.

    Manages persistent tracks across video frames using IoU-based
    association with two-stage matching:
      1. High-confidence detections matched to active tracks
      2. Low-confidence detections matched to lost tracks
    """

    def __init__(self, max_lost_frames=15, iou_threshold_high=0.3,
                 iou_threshold_low=0.15, min_track_length=2):
        """
        Args:
            max_lost_frames: max frames before lost track becomes dead
            iou_threshold_high: min IoU for primary matching
            iou_threshold_low: min IoU for secondary matching (lost tracks)
            min_track_length: min visible frames before track is considered valid
        """
        self.max_lost_frames = max_lost_frames
        self.iou_threshold_high = iou_threshold_high
        self.iou_threshold_low = iou_threshold_low
        self.min_track_length = min_track_length

        self.active_tracks = []   # Currently visible
        self.lost_tracks = []     # Temporarily missing
        self.dead_tracks = []     # Permanently gone (for disappearance analysis)

        # Reset track ID counter for each new video
        Track._next_id = 1

    def reset(self):
        """Reset all tracks (call at start of each video)."""
        self.active_tracks = []
        self.lost_tracks = []
        self.dead_tracks = []
        Track._next_id = 1

    def update(self, detections, frame_idx):
        """
        Update tracks with new detections.

        Args:
            detections: list of dicts with 'bbox', 'confidence',
                        'type' (or 'obj_type'), and optional extra fields
            frame_idx: current frame index

        Returns:
            list of active Track objects with updated state
        """
        if not detections:
            # No detections — mark all active tracks as missing
            for track in self.active_tracks:
                track.mark_missing(frame_idx)

            # Move expired lost tracks to dead
            self._expire_tracks(frame_idx)
            self.lost_tracks.extend(self.active_tracks)
            self.active_tracks = []
            return []

        # Separate high and low confidence detections
        high_conf_dets = []
        low_conf_dets = []
        for det in detections:
            conf = det.get("confidence", 0)
            # Normalize to 0-1 if in 0-100 range
            if conf > 1.0:
                conf = conf / 100.0
            if conf >= 0.3:
                high_conf_dets.append(det)
            else:
                low_conf_dets.append(det)

        # Stage 1: Match high-confidence detections to active tracks
        all_current_tracks = self.active_tracks + self.lost_tracks
        matched_track_indices, matched_det_indices, unmatched_tracks, unmatched_dets = \
            self._match_detections(
                all_current_tracks, high_conf_dets, frame_idx,
                self.iou_threshold_high
            )

        # Update matched tracks
        new_active = []
        for t_idx, d_idx in zip(matched_track_indices, matched_det_indices):
            track = all_current_tracks[t_idx]
            det = high_conf_dets[d_idx]
            conf = det.get("confidence", 0)
            if conf > 1.0:
                conf /= 100.0
            extra = {k: v for k, v in det.items()
                     if k not in ("bbox", "confidence", "type", "obj_type")}
            track.update(det["bbox"], frame_idx, confidence=conf,
                        extra_data=extra)
            new_active.append(track)

        # Stage 2: Match low-confidence detections to remaining lost tracks
        remaining_tracks = [all_current_tracks[i] for i in unmatched_tracks]
        if low_conf_dets and remaining_tracks:
            m_t, m_d, still_unmatched_tracks, _ = self._match_detections(
                remaining_tracks, low_conf_dets, frame_idx,
                self.iou_threshold_low
            )
            for t_idx, d_idx in zip(m_t, m_d):
                track = remaining_tracks[t_idx]
                det = low_conf_dets[d_idx]
                conf = det.get("confidence", 0)
                if conf > 1.0:
                    conf /= 100.0
                extra = {k: v for k, v in det.items()
                         if k not in ("bbox", "confidence", "type", "obj_type")}
                track.update(det["bbox"], frame_idx, confidence=conf,
                            extra_data=extra)
                new_active.append(track)
            remaining_tracks = [remaining_tracks[i]
                               for i in still_unmatched_tracks]
        else:
            remaining_tracks = [all_current_tracks[i] for i in unmatched_tracks]

        # Mark unmatched tracks as missing
        for track in remaining_tracks:
            track.mark_missing(frame_idx)

        # Create new tracks for unmatched high-conf detections
        for d_idx in unmatched_dets:
            det = high_conf_dets[d_idx]
            conf = det.get("confidence", 0)
            if conf > 1.0:
                conf /= 100.0
            obj_type = det.get("type", det.get("obj_type", "unknown"))
            extra = {k: v for k, v in det.items()
                     if k not in ("bbox", "confidence", "type", "obj_type")}
            new_track = Track(
                det["bbox"], frame_idx, obj_type=obj_type,
                confidence=conf, extra_data=extra
            )
            new_active.append(new_track)

        # Separate active vs still-lost
        self.active_tracks = [t for t in new_active if t.status == "active"]
        newly_lost = [t for t in remaining_tracks if t.status == "lost"]
        self.lost_tracks = [t for t in new_active if t.status == "lost"] + newly_lost

        # Expire old lost tracks
        self._expire_tracks(frame_idx)

        return self.active_tracks

    def _match_detections(self, tracks, detections, frame_idx, iou_threshold):
        """
        Hungarian-style greedy IoU matching between tracks and detections.

        Returns:
            matched_track_indices, matched_det_indices,
            unmatched_track_indices, unmatched_det_indices
        """
        if not tracks or not detections:
            return (
                [],
                [],
                list(range(len(tracks))),
                list(range(len(detections))),
            )

        # Build IoU cost matrix
        n_tracks = len(tracks)
        n_dets = len(detections)
        iou_matrix = np.zeros((n_tracks, n_dets))

        for t_idx, track in enumerate(tracks):
            # Use predicted bbox for lost tracks
            if track.status == "lost":
                t_bbox = track.predict_bbox(frame_idx)
            else:
                t_bbox = track.bbox
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = compute_iou(t_bbox, det["bbox"])

        # Greedy matching (sorted by IoU, highest first)
        matched_t = []
        matched_d = []
        used_tracks = set()
        used_dets = set()

        # Flatten and sort
        pairs = []
        for t_idx in range(n_tracks):
            for d_idx in range(n_dets):
                if iou_matrix[t_idx, d_idx] >= iou_threshold:
                    pairs.append((iou_matrix[t_idx, d_idx], t_idx, d_idx))

        pairs.sort(reverse=True)

        for iou_val, t_idx, d_idx in pairs:
            if t_idx in used_tracks or d_idx in used_dets:
                continue
            matched_t.append(t_idx)
            matched_d.append(d_idx)
            used_tracks.add(t_idx)
            used_dets.add(d_idx)

        unmatched_t = [i for i in range(n_tracks) if i not in used_tracks]
        unmatched_d = [i for i in range(n_dets) if i not in used_dets]

        return matched_t, matched_d, unmatched_t, unmatched_d

    def _expire_tracks(self, frame_idx):
        """Move lost tracks that exceed max_lost_frames to dead."""
        still_lost = []
        for track in self.lost_tracks:
            if (frame_idx - track.last_seen_frame) > self.max_lost_frames:
                track.status = "dead"
                self.dead_tracks.append(track)
            else:
                still_lost.append(track)
        self.lost_tracks = still_lost

    def get_all_active_tracks(self):
        """Get all currently active tracks."""
        return self.active_tracks

    def get_all_lost_tracks(self):
        """Get all currently lost (temporarily missing) tracks."""
        return self.lost_tracks

    def get_recently_dead_tracks(self, frame_idx, lookback=30):
        """Get tracks that died within the last N frames."""
        return [
            t for t in self.dead_tracks
            if (frame_idx - t.last_seen_frame) <= lookback
        ]

    def get_track_by_id(self, track_id):
        """Find a track by its ID across all lists."""
        for track in self.active_tracks + self.lost_tracks + self.dead_tracks:
            if track.track_id == track_id:
                return track
        return None

    def get_all_states(self):
        """Get serializable state dicts for all active and lost tracks."""
        states = []
        for track in self.active_tracks:
            states.append(track.get_state())
        for track in self.lost_tracks:
            states.append(track.get_state())
        return states
