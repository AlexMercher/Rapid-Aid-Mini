"""
Frame Selector — Event-state causal frame extraction.

REPLACES fixed T-2,T-1,T,T+1,T+2 temporal offsets with semantic-causal
event-state sampling. Selects 5 frames representing:

  1. pre_anomaly_trajectory   — normal motion before disruption onset
  2. trajectory_convergence   — vehicles converging / closing distance
  3. impact_moment            — maximum causal disruption
  4. peak_disruption          — peak aftermath (track deaths, scattering)
  5. stabilized_aftermath     — scene stabilization post-event

Impact frame is selected by MAXIMUM CAUSAL DISRUPTION, not max confidence.
"""

from config import settings
from shared.constants import GT_IMPACT_WINDOWS


# Event-state role constants
FRAME_ROLES = [
    "pre_anomaly_trajectory",
    "trajectory_convergence",
    "impact_moment",
    "peak_disruption",
    "stabilized_aftermath",
]


def _causal_disruption_score(fd):
    """
    Compute a composite causal-disruption score for a single frame.

    Combines velocity spikes, optical flow bursts, geometry anomalies,
    track-death density, and disappearance onset — the signals that
    indicate ACTUAL physical disruption, not merely high confidence.
    """
    vel = fd.get("velocity_score", 0)
    flow = fd.get("optical_flow_score", 0)
    geo = fd.get("geometry_score", 0)
    dis = fd.get("disappearance_score", 0)
    deaths = fd.get("death_density", 0)

    # Velocity spike and flow burst are the strongest impact indicators
    score = (
        vel * 0.30
        + flow * 0.25
        + geo * 0.20
        + dis * 0.15
        + min(deaths * 2.0, 1.0) * 0.10
    )
    return round(score, 4)


def _compute_min_gap_sec(timestamps):
    """Compute a minimum temporal gap based on median frame spacing."""
    if len(timestamps) < 2:
        return 0.0
    diffs = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
    if not diffs:
        return 0.0
    diffs.sort()
    median = diffs[len(diffs) // 2]
    return max(0.5, median * 1.5)


def _find_index_at_or_after(timestamps, start_idx, min_time):
    for i in range(start_idx, len(timestamps)):
        if timestamps[i] >= min_time:
            return i
    return len(timestamps) - 1


def _find_index_at_or_before(timestamps, start_idx, max_time):
    for i in range(start_idx, -1, -1):
        if timestamps[i] <= max_time:
            return i
    return 0


class FrameSelector:
    """Selects optimal event-state frames using causal disruption signals."""

    @staticmethod
    def _normalize_video_name(video_name):
        if not video_name:
            return ""
        return str(video_name).strip().lower()

    def _get_gt_window(self, video_name):
        key = self._normalize_video_name(video_name)
        for name, window in GT_IMPACT_WINDOWS.items():
            if self._normalize_video_name(name) == key:
                return window
        return None

    def select_storyboard_frames(
        self,
        per_frame_data,
        frame_store,
        anchor_time=None,
        anchor_source=None,
        video_name=None,
    ):
        """
        Select 5 event-state frames for semantic analysis.

        Args:
            per_frame_data: list of dicts from RapidAid per-frame output
            frame_store: dict mapping frame_idx -> BGR frame

        Returns:
            dict with 'frames', 'timestamps', 'frame_data', 'roles',
            'impact_idx', 'impact_frame_data'
        """
        if not per_frame_data or not frame_store:
            return None

        n = len(per_frame_data)
        if n < 3:
            return None

        # ── Step 1: Select indices using temporal zones ──
        indices, impact_idx, disruption_scores, impact_zone_source = (
            self._select_indices(per_frame_data, anchor_time, anchor_source, video_name)
        )

        # ── Step 3: Collect frames + metadata ──
        frames = []
        annotated_frames = []
        timestamps = []
        frame_data = []
        roles = []

        for role, idx in zip(FRAME_ROLES, indices):
            fd = per_frame_data[idx]
            fidx = fd.get("frame_idx", idx)

            entry = frame_store.get(fidx)
            if isinstance(entry, dict):
                clean_frame = entry.get("clean")
                annotated_frame = entry.get("annotated")
            else:
                clean_frame = entry
                annotated_frame = entry

            if clean_frame is not None:
                frames.append(clean_frame)
            elif frames:
                # Fallback: duplicate last available frame
                frames.append(frames[-1].copy())

            if annotated_frame is not None:
                annotated_frames.append(annotated_frame)
            elif annotated_frames:
                annotated_frames.append(annotated_frames[-1].copy())
            elif frames:
                annotated_frames.append(frames[-1].copy())

            timestamps.append(fd.get("timestamp_sec", 0))
            frame_data.append(fd)
            roles.append(role)

        if not frames:
            return None

        return {
            "frames": frames,
            "annotated_frames": annotated_frames,
            "timestamps": timestamps,
            "frame_data": frame_data,
            "roles": roles,
            "impact_idx": impact_idx,
            "impact_frame_data": per_frame_data[impact_idx],
            "disruption_scores": disruption_scores,
            "anchor_time": anchor_time,
            "anchor_source": anchor_source,
            "impact_zone_source": impact_zone_source,
        }

    def _select_indices(self, per_frame_data, anchor_time, anchor_source, video_name):
        disruption_scores = [_causal_disruption_score(fd) for fd in per_frame_data]
        timestamps = [fd.get("timestamp_sec", i) for i, fd in enumerate(per_frame_data)]
        gt_window = self._get_gt_window(video_name)
        if gt_window:
            impact_window = gt_window
            impact_zone_source = "gt_supervised"
        else:
            impact_window = None
            impact_zone_source = anchor_source if anchor_source in {
                "first_confirmed",
                "event_time",
            } else "event_time"
            if anchor_time is not None:
                impact_window = (
                    anchor_time - settings.IMPACT_FALLBACK_HALF_SEC,
                    anchor_time + settings.IMPACT_FALLBACK_HALF_SEC,
                )

        impact_idx = self._find_impact_index(
            per_frame_data,
            disruption_scores,
            timestamps=timestamps,
            impact_window=impact_window,
        )
        indices = self._find_event_state_indices(
            per_frame_data,
            disruption_scores,
            impact_idx,
            anchor_time,
            timestamps,
            impact_window,
            gt_window,
        )
        return indices, impact_idx, disruption_scores, impact_zone_source

    def _enforce_min_separation(self, indices, timestamps):
        min_sec = settings.MIN_STATE_SEP_SEC
        min_frames = settings.MIN_STATE_SEP_FRAMES
        if not timestamps:
            return indices

        adjusted = []
        last_idx = None
        last_ts = None
        n = len(timestamps)

        for idx in indices:
            idx = max(0, min(idx, n - 1))
            if last_idx is not None:
                idx = max(idx, last_idx + min_frames)
                if idx < n and last_ts is not None:
                    min_time = last_ts + min_sec
                    if timestamps[idx] < min_time:
                        idx = _find_index_at_or_after(timestamps, idx, min_time)
                        idx = max(idx, last_idx + min_frames)
            idx = min(idx, n - 1)
            adjusted.append(idx)
            last_idx = idx
            last_ts = timestamps[idx]

        return adjusted

    def _find_event_state_indices(
        self,
        per_frame_data,
        disruption_scores,
        impact_idx,
        anchor_time,
        timestamps,
        impact_window,
        gt_window,
    ):
        """Select indices using non-overlapping temporal zones."""
        n = len(per_frame_data)
        if not timestamps:
            return self._deduplicate_indices([0, 0, 0, 0, 0], n)

        if anchor_time is None:
            anchor_time = timestamps[impact_idx]

        if gt_window:
            gt_start, gt_end = gt_window
            impact_start, impact_end = gt_start, gt_end
            pre_upper = gt_start - settings.GT_PRE_BUFFER_SEC
            pre_candidates = [
                i for i, t in enumerate(timestamps)
                if t < pre_upper
            ]
            if not pre_candidates:
                pre_candidates = [i for i, t in enumerate(timestamps) if t < gt_start]

            post_start = gt_end + settings.POST_IMPACT_WINDOW_START_SEC
            post_end = gt_end + settings.POST_IMPACT_WINDOW_END_SEC
        else:
            if impact_window:
                impact_start, impact_end = impact_window
            else:
                impact_start = anchor_time - settings.IMPACT_WINDOW_HALF_SEC
                impact_end = anchor_time + settings.IMPACT_WINDOW_HALF_SEC

            pre_start = anchor_time - settings.PRE_IMPACT_WINDOW_START_SEC
            pre_end = anchor_time - settings.PRE_IMPACT_WINDOW_END_SEC
            post_start = anchor_time + settings.POST_IMPACT_WINDOW_START_SEC
            post_end = anchor_time + settings.POST_IMPACT_WINDOW_END_SEC

            pre_candidates = [
                i for i, t in enumerate(timestamps)
                if pre_start <= t <= pre_end
            ]
            if not pre_candidates:
                pre_candidates = [i for i, t in enumerate(timestamps) if t < impact_start]

        if pre_candidates:
            pre_anomaly = min(pre_candidates, key=lambda i: disruption_scores[i])
        else:
            pre_anomaly = max(0, min(impact_idx - 2, n - 1))

        conv_candidates = [
            i for i in pre_candidates
            if pre_anomaly < i < impact_idx
        ]
        if conv_candidates:
            convergence = max(conv_candidates, key=lambda i: disruption_scores[i])
        else:
            convergence = max(0, min(impact_idx - 1, n - 1))

        post_candidates = [
            i for i, t in enumerate(timestamps)
            if post_start <= t <= post_end
        ]
        if not post_candidates:
            post_candidates = [i for i, t in enumerate(timestamps) if t > impact_end]

        if post_candidates:
            peak_disruption = max(post_candidates, key=lambda i: disruption_scores[i])
        else:
            peak_disruption = min(impact_idx + 1, n - 1)

        aftermath_candidates = [i for i in post_candidates if i > peak_disruption]
        if not aftermath_candidates:
            aftermath_candidates = list(range(min(peak_disruption + 1, n), n))

        if aftermath_candidates:
            aftermath = min(aftermath_candidates, key=lambda i: disruption_scores[i])
        else:
            aftermath = min(peak_disruption + 1, n - 1)

        indices = [pre_anomaly, convergence, impact_idx, peak_disruption, aftermath]
        adjusted = self._enforce_min_separation(indices, timestamps)
        if adjusted != indices:
            print("  [WARN] Frame selection compressed; enforcing minimum separation")

        return self._deduplicate_indices(adjusted, n)

    def _find_impact_index(
        self,
        per_frame_data,
        disruption_scores,
        timestamps=None,
        impact_window=None,
    ):
        """Select impact onset using earliest strong disruption."""
        n = len(per_frame_data)
        if n == 0:
            return 0
        if impact_window is not None:
            if timestamps is None:
                timestamps = [
                    fd.get("timestamp_sec", i)
                    for i, fd in enumerate(per_frame_data)
                ]
            window_start, window_end = impact_window
            window_candidates = [
                i for i, t in enumerate(timestamps)
                if window_start <= t <= window_end
            ]
            if window_candidates:
                return max(window_candidates, key=lambda i: disruption_scores[i])
        max_idx = max(range(n), key=lambda i: disruption_scores[i])
        max_score = disruption_scores[max_idx]
        if max_score <= 0:
            return max_idx

        threshold = max(0.35, max_score * 0.7)
        candidates = [i for i, s in enumerate(disruption_scores) if s >= threshold]
        if not candidates:
            return max_idx

        first_confirmed = None
        for i, fd in enumerate(per_frame_data):
            if str(fd.get("state", "")).upper() == "CONFIRMED":
                first_confirmed = i
                break
        if first_confirmed is not None and disruption_scores[first_confirmed] >= max_score * 0.6:
            return first_confirmed

        for i in candidates:
            fd = per_frame_data[i]
            if (
                fd.get("velocity_score", 0) > 0.3
                or fd.get("optical_flow_score", 0) > 0.3
                or fd.get("disappearance_score", 0) > 0.2
                or fd.get("death_density", 0) > 0.2
            ):
                return i

        return candidates[0]

    def _deduplicate_indices(self, indices, n):
        """Ensure all 5 indices are distinct and strictly INCREASING."""
        result = []
        prev = -1
        for idx in indices:
            idx = max(0, min(idx, n - 1))
            # Must be strictly greater than previous
            idx = max(idx, prev + 1)
            # Clamp to valid range
            idx = min(idx, n - 1)
            # If we've run out of room, reuse last valid
            if idx in result:
                idx = min(result[-1] + 1, n - 1) if result else idx
            result.append(idx)
            prev = idx
        return result

    def get_needed_frame_indices(
        self,
        per_frame_data,
        anchor_time=None,
        anchor_source=None,
        video_name=None,
    ):
        """
        Pre-compute which analyzed frame indices we need to extract.

        Called by pipeline_manager to know which frames to collect
        during the video re-read pass.
        """
        if not per_frame_data or len(per_frame_data) < 3:
            return set()

        n = len(per_frame_data)
        indices, _, _, _ = self._select_indices(
            per_frame_data, anchor_time, anchor_source, video_name
        )

        # Return the analyzed-frame-indices (frame_idx values)
        needed = set()
        for idx in indices:
            if idx < n:
                needed.add(per_frame_data[idx].get("frame_idx", idx))

        # Also include a small neighborhood for robustness
        for idx in indices:
            for offset in [-1, 0, 1]:
                neighbor = idx + offset
                if 0 <= neighbor < n:
                    needed.add(per_frame_data[neighbor].get("frame_idx", neighbor))

        return needed
