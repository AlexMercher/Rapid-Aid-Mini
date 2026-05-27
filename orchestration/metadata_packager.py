"""
Metadata Packager — Symbolic causal metadata for LLM consumption.

Packages RapidAid causal signals, event-state frame roles, timeline
evolution, and track lifecycle data into structured metadata.

bakllava receives: visual evidence (frames)
Groq receives:     symbolic + semantic evidence (this package)
"""
import json
import math
import os


class MetadataPackager:
    """Packages multi-source metadata with symbolic causal grounding."""

    def package(self, timeline, rapidaid_result, storyboard_path=None,
                bakllava_output=None, event_clip_meta=None,
                frame_selection=None):
        """
        Create a comprehensive metadata package with symbolic causal data.

        Args:
            timeline: dict from TimelineBuilder
            rapidaid_result: raw dict from TrackCentricProcessor
            storyboard_path: path to composite image (human review)
            bakllava_output: bakllava narration text
            event_clip_meta: clip extraction metadata
            frame_selection: dict from FrameSelector with roles + frame_data

        Returns:
            dict with all packaged metadata
        """
        metrics = rapidaid_result.get("metrics", {})
        best_fusion = rapidaid_result.get("best_fusion_result", {}) or {}
        per_frame = rapidaid_result.get("per_frame_data", [])
        event_time = self._safe_float(
            rapidaid_result.get("accident_timestamp_sec", None)
        )
        first_confirmed_time = self._safe_float(
            rapidaid_result.get("first_confirmed_time", None)
        )
        state_transitions = self._sanitize_state_transitions(
            rapidaid_result.get("state_transitions", [])
        )
        safe_event_clip = self._sanitize_event_clip(event_clip_meta)

        # ── Core RapidAid summary ──
        package = {
            "rapidaid": {
                "accident_detected": self._coerce_bool(
                    rapidaid_result.get("accident_detected", False)
                ),
                "confidence": rapidaid_result.get("best_confidence", 0),
                "event_time": event_time,
                "first_confirmed_time": first_confirmed_time,
                "state_transitions": state_transitions,
                "dominant_signal": best_fusion.get("dominant_signal", "unknown"),
                "total_tracks": metrics.get("peak_tracks", 0),
                "dead_tracks": metrics.get("total_dead_tracks", 0),
                "confirmation_reason": best_fusion.get("confirmation_reason", ""),
            },
            "timeline": timeline,
            "storyboard_path": storyboard_path,
            "bakllava_narration": bakllava_output,
            "event_clip": safe_event_clip,
        }

        # ── Symbolic causal metadata (NEW) ──
        impact_frame_signals = self._extract_impact_signals(
            frame_selection, per_frame
        )
        package["impact_frame_signals"] = impact_frame_signals

        # ── Impact signals from timeline ──
        impact_signals = timeline.get("impact_signals", {}) or {}
        geometry_signal = self._get_geometry_signal(
            impact_frame_signals.get("geometry_overlap", 0),
            event_clip_meta,
            impact_frame_signals.get("n_vehicle_tracks", 0),
        )
        impact_signals["geometry"] = geometry_signal

        package["signal_summary"] = {
            "causal_signals_active": (
                impact_signals.get("velocity", 0) > 0.1 or
                impact_signals.get("optical_flow", 0) > 0.2 or
                impact_signals.get("disappearance", 0) > 0.1
            ),
            "strongest_signal": (
                max(impact_signals.items(), key=lambda x: x[1])[0]
                if impact_signals else "none"
            ),
            "signal_values": impact_signals,
        }
        package["event_state_labels"] = self._extract_event_state_labels(
            frame_selection
        )
        package["confidence_progression"] = self._extract_confidence_curve(
            per_frame
        )
        package["track_lifecycle"] = self._extract_track_lifecycle(
            per_frame, metrics
        )

        return package

    def _get_geometry_signal(self, geometry_overlap, event_clip_meta, vehicle_count):
        """Return geometry signal with portrait fallback."""
        if geometry_overlap and geometry_overlap > 0.0:
            return float(geometry_overlap)

        frame_w, frame_h = self._parse_resolution(event_clip_meta)
        if frame_h > frame_w and vehicle_count >= 2:
            return min(0.35, vehicle_count * 0.10)

        return 0.0

    def _parse_resolution(self, event_clip_meta):
        """Parse resolution from event clip metadata."""
        if not event_clip_meta:
            return 0, 0
        resolution = event_clip_meta.get("resolution", "")
        if isinstance(resolution, str) and "x" in resolution:
            parts = resolution.lower().split("x")
            if len(parts) == 2:
                try:
                    return int(parts[0]), int(parts[1])
                except ValueError:
                    return 0, 0
        return 0, 0

    def _extract_impact_signals(self, frame_selection, per_frame):
        """
        Extract detailed causal signals at the impact frame.

        This is the symbolic data Groq uses to understand WHAT happened
        physically, without needing to see the video.
        """
        if not frame_selection:
            # Fallback: use per_frame peak
            if not per_frame:
                return {}
            best = max(per_frame, key=lambda fd: fd.get("final_confidence", 0))
            return self._signals_from_frame_data(best)

        impact_fd = frame_selection.get("impact_frame_data", {})
        return self._signals_from_frame_data(impact_fd)

    def _signals_from_frame_data(self, fd):
        """Extract a symbolic signal dict from a single frame's data."""
        return {
            "velocity_spike": round(fd.get("velocity_score", 0), 3),
            "optical_flow_spike": round(fd.get("optical_flow_score", 0), 3),
            "geometry_overlap": round(fd.get("geometry_score", 0), 3),
            "disappearance_score": round(fd.get("disappearance_score", 0), 3),
            "tracking_score": round(fd.get("tracking_score", 0), 3),
            "detector_score": round(fd.get("detector_score", 0), 3),
            "death_density": round(fd.get("death_density", 0), 3),
            "final_confidence": round(fd.get("final_confidence", 0), 3),
            "causal_present": self._coerce_bool(fd.get("causal_present", False)),
            "n_families": fd.get("n_families", 0),
            "timestamp_sec": fd.get("timestamp_sec", 0),
            "n_vehicle_tracks": fd.get("n_vehicle_tracks", 0),
            "n_dead_vehicles": fd.get("n_dead_vehicles", 0),
        }

    def _coerce_bool(self, value):
        """Normalize boolean-like values for metadata integrity."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)

    def _extract_event_state_labels(self, frame_selection):
        """
        Extract the event-state role labels and their timestamps.

        This tells Groq WHICH frames bakllava narrated, and what
        each frame semantically represents.
        """
        if not frame_selection:
            return []

        roles = frame_selection.get("roles", [])
        timestamps = frame_selection.get("timestamps", [])
        frame_data = frame_selection.get("frame_data", [])
        impact_zone_source = frame_selection.get("impact_zone_source")

        labels = []
        for i, role in enumerate(roles):
            entry = {
                "role": role,
                "timestamp_sec": self._safe_float(
                    timestamps[i] if i < len(timestamps) else 0
                ),
                "impact_zone_source": impact_zone_source,
            }
            if i < len(frame_data):
                fd = frame_data[i]
                entry["confidence"] = round(fd.get("final_confidence", 0), 3)
                entry["state"] = fd.get("state", "?")
            labels.append(entry)

        return labels

    def _safe_float(self, value):
        """Return a finite float or None for invalid values."""
        if value is None:
            return None
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            return None
        return as_float if math.isfinite(as_float) else None

    def _sanitize_state_transitions(self, transitions):
        if not transitions:
            return []
        safe = []
        for tr in transitions:
            if not isinstance(tr, dict):
                continue
            safe.append({
                "from": tr.get("from"),
                "to": tr.get("to"),
                "timestamp": self._safe_float(tr.get("timestamp")),
            })
        return safe

    def _sanitize_event_clip(self, event_clip_meta):
        if not event_clip_meta:
            return event_clip_meta
        safe = dict(event_clip_meta)
        for key in [
            "event_time_sec",
            "anchor_time",
            "clip_start",
            "clip_end",
            "clip_start_sec",
            "clip_end_sec",
        ]:
            if key in safe:
                safe[key] = self._safe_float(safe.get(key))
        return safe

    def _extract_confidence_curve(self, per_frame):
        """
        Extract a sparse confidence progression for temporal context.

        Groq sees how confidence evolved — not just a single peak.
        """
        if not per_frame:
            return []

        # Sample every few frames to keep it compact
        step = max(1, len(per_frame) // 12)
        curve = []
        for i in range(0, len(per_frame), step):
            fd = per_frame[i]
            curve.append({
                "t": round(fd.get("timestamp_sec", 0), 1),
                "conf": round(fd.get("final_confidence", 0), 3),
                "state": fd.get("state", "?"),
            })
        return curve

    def _extract_track_lifecycle(self, per_frame, metrics):
        """Extract track count evolution for Groq's temporal reasoning."""
        if not per_frame:
            return {"peak_tracks": 0, "final_dead": 0, "evolution": []}

        step = max(1, len(per_frame) // 8)
        evolution = []
        for i in range(0, len(per_frame), step):
            fd = per_frame[i]
            evolution.append({
                "t": round(fd.get("timestamp_sec", 0), 1),
                "active": fd.get("n_vehicle_tracks", 0),
                "dead": fd.get("n_dead_vehicles", 0),
                "lost": fd.get("n_lost_vehicles", 0),
            })

        return {
            "peak_tracks": metrics.get("peak_tracks", 0),
            "final_dead": metrics.get("total_dead_tracks", 0),
            "evolution": evolution,
        }

    def save_package(self, package, output_path):
        """Save metadata package to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(package, f, indent=2, default=str)
        return output_path
