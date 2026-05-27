"""
RapidAid — Optical Flow Analyzer (Phase 6)

Independent motion analysis using Farneback Optical Flow.
Detects motion anomalies WITHOUT relying on YOLO detections:
  - Motion bursts (sudden large-area motion)
  - Violent motion patterns (high-magnitude flow)
  - Regional disturbances (localized intense motion)

This provides a detection signal independent of YOLO — even if
YOLO misses a vehicle, optical flow can still detect the crash
impact through the motion disturbance it creates.

Usage:
    analyzer = OpticalFlowAnalyzer()
    for frame in video_frames:
        scores = analyzer.process_frame(frame)
        # scores contains motion_burst_score and optical_flow_score
"""
import cv2
import numpy as np


class OpticalFlowAnalyzer:
    """
    Farneback Optical Flow analyzer for motion anomaly detection.

    Key outputs:
        motion_burst_score: sudden spike in overall motion (0-1)
        optical_flow_score: abnormal flow pattern score (0-1)
        regional_scores: per-region motion intensity map
    """

    def __init__(self,
                 grid_size=4,
                 flow_scale=1.0,
                 burst_threshold=8.0,
                 history_length=30,
                 pyr_scale=0.5,
                 levels=3,
                 winsize=15,
                 iterations=3,
                 poly_n=5,
                 poly_sigma=1.2):
        """
        Args:
            grid_size: divide frame into NxN grid for regional analysis
            flow_scale: scale factor for frame before computing flow
            burst_threshold: magnitude threshold for "burst" classification
            history_length: frames of flow history to maintain
            pyr_scale, levels, winsize, etc.: Farneback parameters
        """
        self.grid_size = grid_size
        self.flow_scale = flow_scale
        self.burst_threshold = burst_threshold
        self.history_length = history_length

        # Farneback parameters
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma

        # State
        self.prev_gray = None
        self.magnitude_history = []
        self.regional_history = []

    def reset(self):
        """Reset state (call at start of each video)."""
        self.prev_gray = None
        self.magnitude_history = []
        self.regional_history = []

    def process_frame(self, frame):
        """
        Compute optical flow and motion scores for a single frame.

        Args:
            frame: BGR numpy array

        Returns:
            dict with:
                'motion_burst_score': float (0-1)
                'optical_flow_score': float (0-1)
                'mean_magnitude': float
                'max_magnitude': float
                'regional_scores': 2D list of per-grid-cell magnitudes
                'flow_field': (mag, angle) tuple for visualization (optional)
                'has_prev': bool (False for first frame)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optionally scale down for speed
        if self.flow_scale != 1.0:
            h, w = gray.shape
            new_w = int(w * self.flow_scale)
            new_h = int(h * self.flow_scale)
            gray = cv2.resize(gray, (new_w, new_h))

        if self.prev_gray is None:
            self.prev_gray = gray
            return self._empty_result()

        # Compute Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            None,
            self.pyr_scale, self.levels, self.winsize,
            self.iterations, self.poly_n, self.poly_sigma,
            0
        )

        self.prev_gray = gray

        # Compute magnitude and angle
        mag, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Global statistics
        mean_mag = float(np.mean(mag))
        max_mag = float(np.max(mag))

        # Regional analysis: divide into grid
        h, w = mag.shape
        cell_h = h // self.grid_size
        cell_w = w // self.grid_size
        regional = []
        for row in range(self.grid_size):
            row_scores = []
            for col in range(self.grid_size):
                y1 = row * cell_h
                y2 = min((row + 1) * cell_h, h)
                x1 = col * cell_w
                x2 = min((col + 1) * cell_w, w)
                cell_mag = mag[y1:y2, x1:x2]
                cell_mean = float(np.mean(cell_mag))
                row_scores.append(round(cell_mean, 2))
            regional.append(row_scores)

        # Update history
        self.magnitude_history.append(mean_mag)
        self.regional_history.append(regional)
        if len(self.magnitude_history) > self.history_length:
            self.magnitude_history.pop(0)
            self.regional_history.pop(0)

        # Compute scores
        motion_burst_score = self._compute_burst_score(mean_mag)
        optical_flow_score = self._compute_flow_anomaly_score(
            mean_mag, max_mag, regional
        )

        return {
            "motion_burst_score": round(motion_burst_score, 3),
            "optical_flow_score": round(optical_flow_score, 3),
            "mean_magnitude": round(mean_mag, 2),
            "max_magnitude": round(max_mag, 2),
            "regional_scores": regional,
            "flow_field": flow,  # Raw (H,W,2) for camera stabilizer
            "flow_polar": (mag, angle),  # For visualization
            "has_prev": True,
        }

    def compute_regional_score(self, frame, roi_bbox):
        """
        Compute optical flow score for a specific region of interest.

        Useful for analyzing flow around:
          - Collision zones
          - Disappearance locations
          - Accident zones

        Args:
            frame: BGR numpy array
            roi_bbox: [x1, y1, x2, y2] region of interest

        Returns:
            float: mean flow magnitude in the ROI (0 if no prev frame)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            return 0.0

        # Match sizes
        if gray.shape != self.prev_gray.shape:
            return 0.0

        # Compute flow on the full frame (already cached in process_frame)
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            None,
            self.pyr_scale, self.levels, self.winsize,
            self.iterations, self.poly_n, self.poly_sigma,
            0
        )

        # Extract ROI
        x1, y1, x2, y2 = roi_bbox
        h, w = gray.shape

        # Scale ROI if flow was computed on scaled frame
        if self.flow_scale != 1.0:
            x1 = int(x1 * self.flow_scale)
            y1 = int(y1 * self.flow_scale)
            x2 = int(x2 * self.flow_scale)
            y2 = int(y2 * self.flow_scale)
            h, w = int(h * self.flow_scale), int(w * self.flow_scale)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        roi_flow = flow[y1:y2, x1:x2]
        mag = np.sqrt(roi_flow[..., 0] ** 2 + roi_flow[..., 1] ** 2)

        return float(np.mean(mag))

    def _compute_burst_score(self, current_mag):
        """
        Detect sudden motion spike relative to recent baseline.

        Returns: 0.0 (no burst) to 1.0 (extreme burst)
        """
        if len(self.magnitude_history) < 5:
            return 0.0

        # Baseline: median of history excluding current
        baseline = np.median(self.magnitude_history[:-1])
        baseline_std = max(np.std(self.magnitude_history[:-1]), 0.5)

        if baseline < 0.5:
            # Very low baseline — use absolute threshold
            if current_mag > self.burst_threshold:
                return min(1.0, current_mag / (self.burst_threshold * 2))
            return 0.0

        # Relative spike
        z_score = (current_mag - baseline) / baseline_std
        if z_score < 2.0:
            return 0.0
        elif z_score < 4.0:
            return (z_score - 2.0) / 2.0 * 0.5  # 0.0 → 0.5
        else:
            return min(1.0, 0.5 + (z_score - 4.0) / 4.0 * 0.5)  # 0.5 → 1.0

    def _compute_flow_anomaly_score(self, mean_mag, max_mag, regional):
        """
        Detect anomalous flow patterns.

        Anomaly indicators:
          1. Very high mean magnitude (whole-frame motion)
          2. Extreme max magnitude (localized violent motion)
          3. High variance between grid cells (concentrated motion)

        Returns: 0.0 to 1.0
        """
        score = 0.0

        # 1. Absolute magnitude
        if mean_mag > self.burst_threshold:
            score += min(0.4, (mean_mag - self.burst_threshold) /
                        (self.burst_threshold * 3) * 0.4)

        # 2. Localized intense motion (max >> mean)
        if mean_mag > 0.5 and max_mag > mean_mag * 3:
            ratio = max_mag / mean_mag
            score += min(0.3, (ratio - 3) / 10 * 0.3)

        # 3. Regional variance (motion concentrated in some cells)
        flat_regional = [v for row in regional for v in row]
        if len(flat_regional) > 1:
            regional_std = np.std(flat_regional)
            regional_mean = np.mean(flat_regional)
            if regional_mean > 0.5:
                cv = regional_std / max(regional_mean, 0.01)
                if cv > 1.0:
                    score += min(0.3, (cv - 1.0) / 3.0 * 0.3)

        return min(1.0, score)

    def _empty_result(self):
        return {
            "motion_burst_score": 0.0,
            "optical_flow_score": 0.0,
            "mean_magnitude": 0.0,
            "max_magnitude": 0.0,
            "regional_scores": [[0.0] * self.grid_size
                                for _ in range(self.grid_size)],
            "flow_field": None,
            "has_prev": False,
        }
