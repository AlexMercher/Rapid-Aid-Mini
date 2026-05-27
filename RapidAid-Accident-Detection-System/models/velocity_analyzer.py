"""
RapidAid — Velocity & Trajectory Analyzer (Phase 4)

Analyzes tracked object velocity and trajectory patterns to
detect anomalous behaviors indicative of accidents:
  - Abrupt deceleration (sudden stop)
  - Velocity collapse (impact-induced halt)
  - Trajectory anomaly (unrealistic direction change)
  - Sudden impact motion patterns

Input: list of Track objects from TrackManager
Output: per-track velocity_collapse_score and trajectory_anomaly_score

Usage:
    analyzer = VelocityAnalyzer()
    scores = analyzer.analyze_tracks(active_tracks, dead_tracks, frame_idx)
"""
import math
import numpy as np


class VelocityAnalyzer:
    """
    Analyzes velocity and trajectory patterns in tracked objects.

    Scoring outputs:
        velocity_collapse_score: 0.0 (normal) to 1.0 (sudden stop from high speed)
        trajectory_anomaly_score: 0.0 (smooth path) to 1.0 (abrupt direction reversal)
    """

    def __init__(self,
                 min_history_frames=4,
                 high_speed_threshold=8.0,
                 decel_threshold=0.7,
                 direction_change_threshold=90.0,
                 smoothing_window=3):
        """
        Args:
            min_history_frames: min frames of history before analysis
            high_speed_threshold: pixels/frame to be considered 'moving fast'
            decel_threshold: fraction of speed lost to be considered 'sudden stop'
            direction_change_threshold: degrees of direction change for anomaly
            smoothing_window: frames for velocity smoothing
        """
        self.min_history_frames = min_history_frames
        self.high_speed_threshold = high_speed_threshold
        self.decel_threshold = decel_threshold
        self.direction_change_threshold = direction_change_threshold
        self.smoothing_window = smoothing_window

    def analyze_tracks(self, active_tracks, lost_tracks=None,
                       dead_tracks=None, frame_idx=0):
        """
        Analyze all tracks for velocity and trajectory anomalies.

        Args:
            active_tracks: list of active Track objects
            lost_tracks: list of lost Track objects (optional)
            dead_tracks: list of recently dead Track objects (optional)
            frame_idx: current frame index

        Returns:
            dict mapping track_id -> {
                'velocity_collapse_score': float,
                'trajectory_anomaly_score': float,
                'max_speed': float,
                'current_speed': float,
                'speed_history': list,
                'sudden_stop': bool,
                'direction_reversal': bool,
            }
        """
        all_tracks = list(active_tracks or [])
        if lost_tracks:
            all_tracks.extend(lost_tracks)
        if dead_tracks:
            all_tracks.extend(dead_tracks)

        results = {}
        for track in all_tracks:
            if len(track.history) < self.min_history_frames:
                results[track.track_id] = self._empty_result()
                continue

            results[track.track_id] = self._analyze_single_track(track)

        return results

    def _analyze_single_track(self, track):
        """Analyze a single track's velocity and trajectory."""
        history = track.history

        # Extract speed and direction history
        speeds = []
        directions = []
        for entry in history:
            vel = entry.get("velocity", (0, 0))
            if vel is None:
                vel = (0, 0)
            speed = (vel[0] ** 2 + vel[1] ** 2) ** 0.5
            direction = math.degrees(math.atan2(vel[1], vel[0])) if speed > 0.5 else None
            speeds.append(speed)
            directions.append(direction)

        # Smooth speeds
        smoothed_speeds = self._smooth(speeds)

        # Velocity collapse score
        velocity_collapse = self._compute_velocity_collapse(smoothed_speeds)

        # Trajectory anomaly score
        trajectory_anomaly = self._compute_trajectory_anomaly(directions, smoothed_speeds)

        # Derived flags
        max_speed = max(smoothed_speeds) if smoothed_speeds else 0
        current_speed = smoothed_speeds[-1] if smoothed_speeds else 0
        sudden_stop = (velocity_collapse > 0.6)
        direction_reversal = (trajectory_anomaly > 0.6)

        return {
            "velocity_collapse_score": round(velocity_collapse, 3),
            "trajectory_anomaly_score": round(trajectory_anomaly, 3),
            "max_speed": round(max_speed, 2),
            "current_speed": round(current_speed, 2),
            "speed_history": [round(s, 2) for s in smoothed_speeds[-10:]],
            "sudden_stop": sudden_stop,
            "direction_reversal": direction_reversal,
        }

    def _compute_velocity_collapse(self, speeds):
        """
        Score how abruptly the object decelerated.

        A high score means the object was moving fast and suddenly stopped.
        This is a key accident indicator — vehicles don't normally stop
        instantaneously.

        Returns: 0.0 to 1.0
        """
        if len(speeds) < 3:
            return 0.0

        max_speed = max(speeds)
        if max_speed < self.high_speed_threshold:
            # Object was never moving fast enough to care about
            return 0.0

        # Find the biggest speed drop in the history
        max_drop_ratio = 0.0
        for i in range(1, len(speeds)):
            if speeds[i - 1] > self.high_speed_threshold:
                drop = (speeds[i - 1] - speeds[i]) / speeds[i - 1]
                max_drop_ratio = max(max_drop_ratio, drop)

        # Also check if object went from fast to near-zero
        recent_avg = np.mean(speeds[-3:])
        peak_speed = max(speeds[:-3]) if len(speeds) > 3 else max_speed
        if peak_speed > self.high_speed_threshold and recent_avg < 2.0:
            overall_collapse = 1.0 - (recent_avg / peak_speed)
            max_drop_ratio = max(max_drop_ratio, overall_collapse)

        # Clamp and scale
        if max_drop_ratio < self.decel_threshold:
            return max_drop_ratio * 0.4  # Mild deceleration
        else:
            # Strong deceleration — scale to 0.5-1.0 range
            return 0.5 + 0.5 * min(1.0, (max_drop_ratio - self.decel_threshold)
                                    / (1.0 - self.decel_threshold))

    def _compute_trajectory_anomaly(self, directions, speeds):
        """
        Score how abruptly the object changed direction.

        A high score means the object was moving in one direction and
        suddenly changed course — classic post-impact behavior.

        Returns: 0.0 to 1.0
        """
        valid_dirs = [(i, d) for i, d in enumerate(directions)
                      if d is not None and speeds[i] > 2.0]

        if len(valid_dirs) < 3:
            return 0.0

        # Compute direction changes
        max_change = 0.0
        for j in range(1, len(valid_dirs)):
            idx_prev, dir_prev = valid_dirs[j - 1]
            idx_curr, dir_curr = valid_dirs[j]

            # Angular difference (handle wraparound)
            delta = abs(dir_curr - dir_prev)
            if delta > 180:
                delta = 360 - delta

            # Weight by speed — fast direction changes are more anomalous
            speed_at_change = speeds[idx_curr]
            weighted_change = delta * min(1.0, speed_at_change / self.high_speed_threshold)
            max_change = max(max_change, weighted_change)

        # Scale to 0-1
        if max_change < self.direction_change_threshold * 0.5:
            return 0.0
        elif max_change < self.direction_change_threshold:
            return (max_change - self.direction_change_threshold * 0.5) / \
                   (self.direction_change_threshold * 0.5) * 0.5
        else:
            return min(1.0, 0.5 + 0.5 *
                       (max_change - self.direction_change_threshold) /
                       self.direction_change_threshold)

    def _smooth(self, values):
        """Apply simple moving average smoothing."""
        if len(values) <= self.smoothing_window:
            return list(values)

        smoothed = []
        for i in range(len(values)):
            start = max(0, i - self.smoothing_window + 1)
            window = values[start:i + 1]
            smoothed.append(np.mean(window))
        return smoothed

    def _empty_result(self):
        return {
            "velocity_collapse_score": 0.0,
            "trajectory_anomaly_score": 0.0,
            "max_speed": 0.0,
            "current_speed": 0.0,
            "speed_history": [],
            "sudden_stop": False,
            "direction_reversal": False,
        }

    def get_anomalous_tracks(self, track_scores, threshold=0.5):
        """
        Filter tracks with significant anomalies.

        Returns: list of (track_id, combined_score) tuples
        """
        anomalous = []
        for track_id, scores in track_scores.items():
            combined = max(
                scores["velocity_collapse_score"],
                scores["trajectory_anomaly_score"],
            )
            if combined >= threshold:
                anomalous.append((track_id, combined))

        anomalous.sort(key=lambda x: x[1], reverse=True)
        return anomalous
