"""
RapidAid — Camera Shake Suppression (Feature 5)

Estimates global frame motion to detect camera shake.
When most optical flow vectors move uniformly, it means
the camera moved — NOT the scene objects.

Outputs a suppression factor (0.0 = full shake, 1.0 = stable camera)
that multiplies the optical flow score before fusion.
"""
import cv2
import numpy as np


class CameraStabilizer:
    """
    Detects camera shake by analyzing global motion coherence.

    If >60% of flow vectors point in the same direction with
    similar magnitude, the motion is camera-induced, not scene-induced.
    """

    def __init__(self, coherence_threshold=0.6, min_magnitude=1.0,
                 grid_size=8):
        """
        Args:
            coherence_threshold: fraction of vectors that must agree
                                 to classify as camera shake
            min_magnitude: minimum average flow magnitude to bother checking
            grid_size: grid divisions for sampling flow field
        """
        self.coherence_threshold = coherence_threshold
        self.min_magnitude = min_magnitude
        self.grid_size = grid_size

    def compute_suppression(self, flow_field):
        """
        Compute camera shake suppression factor from a flow field.

        Args:
            flow_field: numpy array (H, W, 2) from cv2.calcOpticalFlowFarneback

        Returns:
            dict with:
                suppression_factor: 0.0 (full shake) to 1.0 (stable)
                is_camera_shake: bool
                global_motion: (dx, dy) average motion vector
                coherence: fraction of vectors that agree
        """
        if flow_field is None:
            return {
                "suppression_factor": 1.0,
                "is_camera_shake": False,
                "global_motion": (0, 0),
                "coherence": 0.0,
            }

        h, w = flow_field.shape[:2]

        # Sample flow at grid points
        step_y = max(1, h // self.grid_size)
        step_x = max(1, w // self.grid_size)
        samples_x = flow_field[::step_y, ::step_x, 0].flatten()
        samples_y = flow_field[::step_y, ::step_x, 1].flatten()

        if len(samples_x) == 0:
            return {
                "suppression_factor": 1.0,
                "is_camera_shake": False,
                "global_motion": (0, 0),
                "coherence": 0.0,
            }

        # Compute global motion (median is more robust than mean)
        global_dx = float(np.median(samples_x))
        global_dy = float(np.median(samples_y))
        global_mag = (global_dx**2 + global_dy**2)**0.5

        if global_mag < self.min_magnitude:
            return {
                "suppression_factor": 1.0,
                "is_camera_shake": False,
                "global_motion": (round(global_dx, 2), round(global_dy, 2)),
                "coherence": 0.0,
            }

        # Check coherence: how many vectors agree with global motion?
        # Compute angle of each sample vs global angle
        global_angle = np.arctan2(global_dy, global_dx)
        sample_angles = np.arctan2(samples_y, samples_x)
        sample_mags = np.sqrt(samples_x**2 + samples_y**2)

        # Only consider samples with meaningful magnitude
        valid = sample_mags > self.min_magnitude * 0.5
        if np.sum(valid) < 4:
            return {
                "suppression_factor": 1.0,
                "is_camera_shake": False,
                "global_motion": (round(global_dx, 2), round(global_dy, 2)),
                "coherence": 0.0,
            }

        valid_angles = sample_angles[valid]
        angle_diffs = np.abs(valid_angles - global_angle)
        # Wrap to [0, pi]
        angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)

        # Vectors within 30 degrees of global direction = agreeing
        agreeing = np.sum(angle_diffs < np.radians(30))
        coherence = agreeing / len(valid_angles)

        is_shake = coherence >= self.coherence_threshold

        if is_shake:
            # Strong shake = strong suppression
            suppression = max(0.0, 1.0 - coherence)
        else:
            suppression = 1.0

        return {
            "suppression_factor": round(suppression, 3),
            "is_camera_shake": is_shake,
            "global_motion": (round(global_dx, 2), round(global_dy, 2)),
            "coherence": round(coherence, 3),
        }
