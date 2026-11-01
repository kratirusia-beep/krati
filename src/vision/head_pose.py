"""
Head Pose Estimation & Gaze Direction Classifier.
Epic 5: Visual Engagement & Attention Dynamics (Task 9)
Uses cv2.solvePnP with canonical 3D facial geometry to compute Yaw, Pitch, Roll angles.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import cv2

from src.config import get_config


@dataclass
class HeadPoseResult:
    yaw: float          # Left (-) / Right (+) in degrees
    pitch: float        # Down (-) / Up (+) in degrees
    roll: float         # Tilt Left (-) / Tilt Right (+) in degrees
    gaze_direction: str # 'LOOKING_FORWARD', 'LOOKING_DOWN', 'LOOKING_AWAY'
    attention_score: float # 0.0 to 1.0


class HeadPoseEstimator:
    """Computes Euler angles from 2D landmarks using Perspective-n-Point."""

    def __init__(self):
        self.config = get_config()
        self.yaw_limit = self.config.get("vision.head_pose.yaw_limit", 28.0)
        self.pitch_min = self.config.get("vision.head_pose.pitch_min", -22.0)
        self.pitch_max = self.config.get("vision.head_pose.pitch_max", 18.0)

        # Standard canonical 3D facial model coordinates (in millimeters)
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye outer corner
            (225.0, 170.0, -135.0),   # Right eye outer corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ], dtype=np.float32)

    def estimate_pose(
        self, landmarks_2d: np.ndarray, frame_shape: Tuple[int, int]
    ) -> HeadPoseResult:
        """
        Estimates yaw, pitch, roll and evaluates board/screen attention.
        frame_shape: (height, width)
        """
        if landmarks_2d is None or len(landmarks_2d) < 6:
            return HeadPoseResult(
                yaw=0.0, pitch=0.0, roll=0.0,
                gaze_direction="LOOKING_FORWARD",
                attention_score=0.85
            )

        h, w = frame_shape[:2]
        focal_length = float(w)
        center = (float(w / 2.0), float(h / 2.0))
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            self.model_points_3d,
            landmarks_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return HeadPoseResult(
                yaw=0.0, pitch=0.0, roll=0.0,
                gaze_direction="LOOKING_FORWARD",
                attention_score=0.80
            )

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Decompose projection matrix to extract Euler angles
        proj_matrix = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = float(euler_angles[0, 0])
        yaw = float(euler_angles[1, 0])
        roll = float(euler_angles[2, 0])

        # Normalize angles to [-180, 180]
        pitch = ((pitch + 180) % 360) - 180
        yaw = ((yaw + 180) % 360) - 180
        roll = ((roll + 180) % 360) - 180

        # Classify attention
        is_yaw_ok = abs(yaw) <= self.yaw_limit
        is_pitch_ok = (pitch >= self.pitch_min) and (pitch <= self.pitch_max)

        if is_yaw_ok and is_pitch_ok:
            direction = "LOOKING_FORWARD"
            # Higher attention when looking directly at center
            att_score = 1.0 - (0.35 * (abs(yaw) / self.yaw_limit) + 0.15 * (abs(pitch) / max(abs(self.pitch_min), 1.0)))
            att_score = float(np.clip(att_score, 0.70, 1.0))
        elif pitch < self.pitch_min and abs(yaw) <= (self.yaw_limit * 1.3):
            direction = "LOOKING_DOWN"
            att_score = 0.65  # Constructive engagement (e.g., notebook or laptop work)
        else:
            direction = "LOOKING_AWAY"
            excess_yaw = max(0.0, abs(yaw) - self.yaw_limit)
            att_score = float(max(0.15, 0.45 - (excess_yaw * 0.01)))

        return HeadPoseResult(
            yaw=round(yaw, 2),
            pitch=round(pitch, 2),
            roll=round(roll, 2),
            gaze_direction=direction,
            attention_score=round(att_score, 3)
        )
