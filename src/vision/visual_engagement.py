"""
Visual Engagement & Behavior Dynamics Analyzer.
Epic 5: Visual Engagement & Attention Dynamics (Task 9 & 10)
Fuses Head Pose, Eye Aspect Ratio, Inactivity Duration, and Movement Dynamics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.config import get_config
from .head_pose import HeadPoseEstimator, HeadPoseResult
from .face_tracker import TrackedTarget


@dataclass
class StudentVisualState:
    track_id: int
    student_id: Optional[str]
    student_name: str
    head_pose: HeadPoseResult
    eye_aspect_ratio: float
    motion_energy: float
    stillness_seconds: float
    drowsiness_flag: bool
    visual_score: float  # 0.0 to 100.0
    status_label: str    # 'Attentive', 'Taking Notes', 'Distracted', 'Drowsy / Inactive'


class VisualEngagementAnalyzer:
    """Analyzes continuous visual attention cues per tracked student."""

    def __init__(self):
        self.config = get_config()
        self.pose_estimator = HeadPoseEstimator()
        self.stillness_threshold = self.config.get("vision.inactivity.stillness_threshold", 0.04)
        self.inactivity_alert_sec = self.config.get("vision.inactivity.inactivity_alert_sec", 45.0)
        self.ear_threshold = self.config.get("vision.inactivity.drowsiness_ear_threshold", 0.20)

        # Map track_id -> persistent state tracking
        self.stillness_accumulators: Dict[int, float] = {}
        self.last_update_times: Dict[int, float] = {}
        self.smoothed_scores: Dict[int, float] = {}

    def analyze(
        self,
        tracked_targets: List[TrackedTarget],
        frame_shape: Tuple[int, int],
        current_time: float
    ) -> List[StudentVisualState]:
        """Processes each tracked face and computes its behavioral engagement metrics."""
        results: List[StudentVisualState] = []

        for target in tracked_targets:
            tid = target.track_id
            dt = 0.0
            if tid in self.last_update_times:
                dt = max(0.0, current_time - self.last_update_times[tid])
            self.last_update_times[tid] = current_time

            # 1. Head Pose & Gaze
            pose = self.pose_estimator.estimate_pose(target.landmarks_2d, frame_shape)

            # 2. Movement & Stillness accumulation
            motion = float(target.motion_energy)
            prev_stillness = self.stillness_accumulators.get(tid, 0.0)
            if motion < self.stillness_threshold:
                new_stillness = prev_stillness + dt
            else:
                new_stillness = max(0.0, prev_stillness - dt * 2.0)
            self.stillness_accumulators[tid] = new_stillness

            # 3. Drowsiness / Inactivity evaluation
            ear = float(target.eye_aspect_ratio)
            is_drowsy = (ear < self.ear_threshold) or (new_stillness > self.inactivity_alert_sec)

            # 4. Composite Visual Engagement Score (0 - 100)
            # Base attention from head pose (0.0 to 1.0)
            base_att = pose.attention_score

            # Alertness multiplier
            if is_drowsy:
                alertness = 0.25
            elif new_stillness > (self.inactivity_alert_sec * 0.6):
                alertness = 0.60
            else:
                alertness = 1.0

            # Dynamic motion factor (healthy natural subtle movement is positive; extreme restlessness is distraction)
            if 0.04 <= motion <= 0.35:
                motion_factor = 1.0
            elif motion > 0.35:
                motion_factor = 0.70  # Excessive fidgeting
            else:
                motion_factor = 0.85  # Slightly static

            instant_score = (base_att * 0.60 + alertness * 0.30 + motion_factor * 0.10) * 100.0
            instant_score = float(np.clip(instant_score, 5.0, 100.0))

            # Temporal smoothing (EMA)
            prev_smooth = self.smoothed_scores.get(tid, instant_score)
            smoothed = 0.35 * instant_score + 0.65 * prev_smooth
            self.smoothed_scores[tid] = smoothed

            # Categorize status label
            if is_drowsy:
                label = "Drowsy / Inactive"
            elif pose.gaze_direction == "LOOKING_FORWARD":
                label = "Attentive"
            elif pose.gaze_direction == "LOOKING_DOWN":
                label = "Taking Notes"
            else:
                label = "Distracted"

            results.append(
                StudentVisualState(
                    track_id=tid,
                    student_id=target.student_id,
                    student_name=target.student_name,
                    head_pose=pose,
                    eye_aspect_ratio=round(ear, 3),
                    motion_energy=round(motion, 3),
                    stillness_seconds=round(new_stillness, 1),
                    drowsiness_flag=is_drowsy,
                    visual_score=round(smoothed, 1),
                    status_label=label,
                )
            )

        return results
