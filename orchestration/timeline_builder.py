"""
Timeline Builder — Constructs event timelines from RapidAid metadata.

Builds a structured timeline of what happened and when, for consumption
by Groq's synthesis engine.
"""
import json
import os
from datetime import datetime


class TimelineBuilder:
    """Builds structured event timelines from RapidAid processing results."""

    def build_timeline(self, rapidaid_result, per_frame_data=None, video_file=None):
        """
        Build an event timeline from RapidAid output.

        Args:
            rapidaid_result: dict from TrackCentricProcessor.process_video()
            per_frame_data: optional per-frame signal data

        Returns:
            dict with structured timeline
        """
        metrics = rapidaid_result.get("metrics", {})
        best_fusion = rapidaid_result.get("best_fusion_result", {}) or {}
        timeline = {
            "generated_at": datetime.now().isoformat(),
            "video_file": video_file or rapidaid_result.get("video_file", "unknown"),
            "accident_detected": self._coerce_bool(
                rapidaid_result.get("accident_detected", False)
            ),
            "event_time_sec": rapidaid_result.get("accident_timestamp_sec", None),
            "best_confidence": rapidaid_result.get("best_confidence", 0),
            "dominant_signal": best_fusion.get("dominant_signal", "unknown"),
            "total_tracks": metrics.get("peak_tracks", 0),
            "dead_tracks": metrics.get("total_dead_tracks", 0),
            "avg_fps": metrics.get("avg_fps", 0),
            "phases": [],
        }

        if not per_frame_data:
            return timeline

        # Build phases from per-frame data
        current_phase = None
        for fd in per_frame_data:
            state = fd.get("state", "CLEAR")
            t = fd.get("timestamp_sec", 0)
            conf = fd.get("final_confidence", 0)

            if state != current_phase:
                timeline["phases"].append({
                    "state": state,
                    "start_time": t,
                    "start_confidence": round(conf, 3),
                    "causal_present": self._coerce_bool(
                        fd.get("causal_present", False)
                    ),
                    "n_families": fd.get("n_families", 0),
                })
                current_phase = state

        # Signal summary at impact
        if timeline["event_time_sec"]:
            impact_frames = [
                fd for fd in per_frame_data
                if abs(fd.get("timestamp_sec", 0) - timeline["event_time_sec"]) < 1.0
            ]
            if impact_frames:
                best = max(impact_frames, key=lambda x: x.get("final_confidence", 0))
                timeline["impact_signals"] = {
                    "detector": best.get("detector_score", 0),
                    "tracking": best.get("tracking_score", 0),
                    "velocity": best.get("velocity_score", 0),
                    "optical_flow": best.get("optical_flow_score", 0),
                    "disappearance": best.get("disappearance_score", 0),
                    "geometry": best.get("geometry_score", 0),
                }

        return timeline

    def _coerce_bool(self, value):
        """Normalize boolean-like values for metadata integrity."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)

    def save_timeline(self, timeline, output_path):
        """Save timeline to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(timeline, f, indent=2, default=str)
        return output_path
