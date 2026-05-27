"""
Event Extractor — Rolling frame buffer + event clip extraction.

Maintains a deque of recent frames. When RapidAid escalates an event,
extracts T-8s to T+12s around the event timestamp and saves:
  - raw_event_clip.mp4
  - overlay_event_clip.mp4 (with tracking overlays)
"""
import os
import cv2
import json
from collections import deque
from datetime import datetime
from shared.constants import EVENTS_DIR, EVENT_PRE_SECONDS, EVENT_POST_SECONDS
from config import settings


class EventExtractor:
    """Rolling frame buffer that extracts event clips on escalation."""

    def __init__(self, fps=30.0, buffer_seconds=25.0):
        self.fps = fps
        max_frames = int(fps * buffer_seconds)
        self.raw_buffer = deque(maxlen=max_frames)
        self.overlay_buffer = deque(maxlen=max_frames)
        self.timestamps = deque(maxlen=max_frames)
        self.frame_count = 0

    def add_frame(self, raw_frame, overlay_frame=None, timestamp_sec=None):
        """Add a frame to the rolling buffer."""
        self.raw_buffer.append(raw_frame.copy())
        self.overlay_buffer.append(
            overlay_frame.copy() if overlay_frame is not None else raw_frame.copy()
        )
        self.timestamps.append(timestamp_sec or self.frame_count / self.fps)
        self.frame_count += 1

    def extract_event_clip(self, rapidaid_output, event_id=None):
        """
        Extract event clip centered around the selected anchor time.

        Returns:
            dict with paths to raw and overlay clips, plus metadata
        """
        if not self.raw_buffer:
            return None

        event_time = rapidaid_output.get("event_time")
        if event_time is None:
            event_time = rapidaid_output.get("accident_timestamp_sec", 0) or 0
        first_confirmed = rapidaid_output.get("first_confirmed_time")
        video_duration = rapidaid_output.get("duration_sec")

        anchor_time = event_time
        anchor_source = "event_time"
        if (first_confirmed is not None
                and event_time is not None
                and (event_time - first_confirmed) > settings.ANCHOR_DIFF_MIN_SEC):
            anchor_time = first_confirmed
            anchor_source = "first_confirmed"

        if video_duration is None:
            video_duration = self.timestamps[-1] if self.timestamps else anchor_time

        event_id = event_id or datetime.now().strftime("EVT_%Y%m%d_%H%M%S")
        event_dir = os.path.join(EVENTS_DIR, event_id)
        os.makedirs(event_dir, exist_ok=True)
        os.makedirs(os.path.join(event_dir, "raw_frames"), exist_ok=True)
        os.makedirs(os.path.join(event_dir, "overlay_frames"), exist_ok=True)

        # Find frame indices for the clip
        clip_start = max(0.0, anchor_time - EVENT_PRE_SECONDS)
        clip_end = min(video_duration, anchor_time + EVENT_POST_SECONDS)
        t_start = clip_start
        t_end = clip_end

        raw_clip_frames = []
        overlay_clip_frames = []
        clip_timestamps = []

        for i, ts in enumerate(self.timestamps):
            if t_start <= ts <= t_end:
                raw_clip_frames.append(self.raw_buffer[i])
                overlay_clip_frames.append(self.overlay_buffer[i])
                clip_timestamps.append(ts)

        if not raw_clip_frames:
            return None

        h, w = raw_clip_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        clip_fps = max(1.0, len(raw_clip_frames) / max(
            clip_timestamps[-1] - clip_timestamps[0], 1.0
        ))

        # Save raw clip
        raw_path = os.path.join(event_dir, "raw_event_clip.mp4")
        writer = cv2.VideoWriter(raw_path, fourcc, clip_fps, (w, h))
        for f in raw_clip_frames:
            writer.write(f)
        writer.release()

        # Save overlay clip
        overlay_path = os.path.join(event_dir, "overlay_event_clip.mp4")
        writer = cv2.VideoWriter(overlay_path, fourcc, clip_fps, (w, h))
        for f in overlay_clip_frames:
            writer.write(f)
        writer.release()

        # Save key frames as images
        n = len(raw_clip_frames)
        key_indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        for idx in key_indices:
            if idx < n:
                cv2.imwrite(
                    os.path.join(event_dir, "raw_frames", f"frame_{idx:04d}.jpg"),
                    raw_clip_frames[idx]
                )
                cv2.imwrite(
                    os.path.join(event_dir, "overlay_frames", f"frame_{idx:04d}.jpg"),
                    overlay_clip_frames[idx]
                )

        metadata = {
            "event_id": event_id,
            "event_time_sec": event_time,
            "anchor_time": anchor_time,
            "anchor_source": anchor_source,
            "clip_start": clip_start,
            "clip_end": clip_end,
            "clip_start_sec": clip_timestamps[0] if clip_timestamps else 0,
            "clip_end_sec": clip_timestamps[-1] if clip_timestamps else 0,
            "n_frames": len(raw_clip_frames),
            "clip_fps": round(clip_fps, 1),
            "resolution": f"{w}x{h}",
            "raw_clip": raw_path,
            "overlay_clip": overlay_path,
        }

        with open(os.path.join(event_dir, "clip_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata
