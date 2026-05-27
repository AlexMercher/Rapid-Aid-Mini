"""
Storyboard Generator — Saves individual high-resolution event-state frames.

REPLACES the old composite horizontal strip with 5 individual full-resolution
frames, each labeled with its semantic role and timestamp.

bakllava receives each frame INDEPENDENTLY at high resolution, preserving:
  - vehicle overlap / deformation
  - debris visibility
  - victim posture
  - impact damage detail

A composite summary image is ALSO saved for human review, but is NOT
sent to bakllava.
"""
import os
import cv2
import numpy as np

from orchestration.frame_selector import FRAME_ROLES

# Semantic frame filenames (ordered)
FRAME_FILENAMES = [
    "pre_anomaly.jpg",
    "convergence.jpg",
    "impact.jpg",
    "disruption.jpg",
    "aftermath.jpg",
]

# Role display labels for annotation
ROLE_LABELS = {
    "pre_anomaly_trajectory":  "A: Pre-Anomaly Trajectory",
    "trajectory_convergence":  "B: Trajectory Convergence",
    "impact_moment":           "C: Impact Moment",
    "peak_disruption":         "D: Peak Disruption",
    "stabilized_aftermath":    "E: Stabilized Aftermath",
}

# Target resolution for semantic frames sent to bakllava
# Balances fidelity vs memory/inference speed
SEMANTIC_FRAME_WIDTH = 768
SEMANTIC_FRAME_HEIGHT = 432


class StoryboardGenerator:
    """
    Generates individual high-resolution event-state frames.

    Saves:
      event_dir/event_state_frames/
        pre_anomaly.jpg
        convergence.jpg
        impact.jpg
        disruption.jpg
        aftermath.jpg

    Also saves a composite storyboard.jpg for human review.
    """

    def __init__(self, frame_width=SEMANTIC_FRAME_WIDTH,
                 frame_height=SEMANTIC_FRAME_HEIGHT):
        self.frame_width = frame_width
        self.frame_height = frame_height

    def generate(self, frames, timestamps, event_data=None,
                 output_path=None, roles=None, annotated_frames=None):
        """
        Generate individual event-state frame images.

        Args:
            frames: list of BGR frames (5 event-state frames)
            timestamps: list of float timestamps
            event_data: optional per-frame signal data
            output_path: base path (e.g. event_dir/storyboard.jpg)
            roles: list of role strings from FrameSelector

        Returns:
            dict with:
              'individual_frames': list of resized BGR arrays (for bakllava)
              'frame_paths': list of saved file paths
              'roles': role labels
              'composite': composite BGR image (for human review)
        """
        if not frames:
            return None

        if roles is None:
            roles = FRAME_ROLES[:len(frames)]

        # Pad/trim to 5
        while len(frames) < 5:
            frames.append(frames[-1].copy())
            timestamps.append(timestamps[-1] if timestamps else 0.0)
            if roles and len(roles) < 5:
                roles.append(FRAME_ROLES[len(roles)] if len(roles) < len(FRAME_ROLES)
                             else "unknown")
        frames = frames[:5]
        timestamps = timestamps[:5]
        roles = roles[:5]

        if annotated_frames is None:
            annotated_frames = frames
        while len(annotated_frames) < 5:
            if annotated_frames:
                annotated_frames.append(annotated_frames[-1].copy())
            else:
                annotated_frames.append(frames[-1].copy())
        annotated_frames = annotated_frames[:5]

        # ── Save individual high-res frames ──
        clean_dir = None
        debug_dir = None
        legacy_dir = None
        clean_paths = []
        debug_paths = []
        if output_path:
            event_dir = os.path.dirname(output_path)
            clean_dir = os.path.join(event_dir, "clean_event_frames")
            debug_dir = os.path.join(event_dir, "overlay_debug_frames")
            legacy_dir = os.path.join(event_dir, "event_state_frames")
            os.makedirs(clean_dir, exist_ok=True)
            os.makedirs(debug_dir, exist_ok=True)
            os.makedirs(legacy_dir, exist_ok=True)

        clean_frames = []
        debug_frames = []
        for i, (frame, ts, role) in enumerate(zip(frames, timestamps, roles)):
            # Resize to semantic resolution with letterboxing
            resized = self._resize_with_letterbox(frame)
            annotated_base = annotated_frames[i] if i < len(annotated_frames) else frame
            annotated_resized = self._resize_with_letterbox(annotated_base)

            # Clean frame for bakllava (no overlays)
            clean_frames.append(resized)

            # Debug overlay frame for humans
            labeled = annotated_resized.copy()
            label_text = ROLE_LABELS.get(role, role)
            label_full = f"{label_text} | t={ts:.1f}s"

            cv2.rectangle(labeled, (0, 0), (len(label_full) * 11 + 10, 30),
                         (0, 0, 0), -1)
            cv2.putText(labeled, label_full, (5, 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
                       cv2.LINE_AA)

            if event_data and i < len(event_data):
                fd = event_data[i]
                conf = fd.get("final_confidence", 0)
                state = fd.get("state", "?")

                bar_w = int(conf * (self.frame_width - 10))
                color = ((0, 255, 0) if conf < 0.4
                         else (0, 165, 255) if conf < 0.6
                         else (0, 0, 255))
                cv2.rectangle(labeled,
                             (5, self.frame_height - 15),
                             (5 + bar_w, self.frame_height - 5),
                             color, -1)
                cv2.putText(labeled, f"[{state}] {conf:.2f}",
                           (5, self.frame_height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                           (255, 255, 255), 1)

            if role == "impact_moment":
                cv2.rectangle(labeled, (0, 0),
                             (self.frame_width - 1, self.frame_height - 1),
                             (0, 0, 255), 3)

            debug_frames.append(labeled)

            if clean_dir:
                fname = FRAME_FILENAMES[i] if i < len(FRAME_FILENAMES) else f"frame_{i}.jpg"
                clean_path = os.path.join(clean_dir, fname)
                debug_path = os.path.join(debug_dir, fname)
                legacy_path = os.path.join(legacy_dir, fname)
                cv2.imwrite(clean_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
                cv2.imwrite(debug_path, labeled, [cv2.IMWRITE_JPEG_QUALITY, 92])
                cv2.imwrite(legacy_path, labeled, [cv2.IMWRITE_JPEG_QUALITY, 92])
                clean_paths.append(clean_path)
                debug_paths.append(debug_path)

        # ── Generate composite for human review ──
        composite = self._build_composite(debug_frames, timestamps, roles)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, composite)

        return {
            "individual_frames": clean_frames,
            "frame_paths": clean_paths,
            "debug_frame_paths": debug_paths,
            "roles": roles,
            "composite": composite,
        }

    def _build_composite(self, frames, timestamps, roles):
        """Build a composite review image from individual frames (for humans)."""
        # Use smaller thumbnails for the composite
        thumb_w = 384
        thumb_h = 216
        thumbs = []
        for f in frames:
            t = cv2.resize(f, (thumb_w, thumb_h))
            thumbs.append(t)

        # Horizontal strip
        strip = np.hstack(thumbs)

        # Title bar
        title_h = 35
        title_bar = np.zeros((title_h, strip.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_bar,
                    "EVENT STATE FRAMES - Causal Event Progression",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)

        return np.vstack([title_bar, strip])

    def _resize_with_letterbox(self, frame):
        """Resize while preserving aspect ratio, padding to target size."""
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return cv2.resize(frame, (self.frame_width, self.frame_height))

        scale = min(self.frame_width / w, self.frame_height / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        canvas = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        x = (self.frame_width - new_w) // 2
        y = (self.frame_height - new_h) // 2
        canvas[y:y + new_h, x:x + new_w] = resized
        return canvas
