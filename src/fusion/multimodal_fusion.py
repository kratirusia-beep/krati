"""
Multimodal Signal Fusion Engine & Session Dynamics Analytics.
Epic 7: Multimodal Signal Fusion (Task 13 & 14)
Fuses Computer Vision (Head Pose, Motion, Inactivity) and Audio Signal Dynamics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

from src.config import get_config
from src.vision.visual_engagement import StudentVisualState
from src.audio.audio_analyzer import AudioAcousticState


@dataclass
class SessionIntervalMetrics:
    interval_index: int
    timestamp_iso: str
    aggregate_engagement_score: float  # 0.0 to 100.0
    visual_attention_avg: float
    acoustic_factor: float
    active_students_count: int
    attentive_pct: float
    taking_notes_pct: float
    distracted_pct: float
    drowsy_pct: float
    acoustic_environment: str
    is_attention_drop: bool = False
    is_interaction_peak: bool = False


@dataclass
class ClassroomDynamicsSnapshot:
    timestamp_ts: float
    timestamp_iso: str
    overall_engagement_score: float
    smoothed_engagement_score: float
    active_students_count: int
    audio_state: AudioAcousticState
    student_scores: Dict[str, float]  # student_id or track_id -> score
    student_states: List[StudentVisualState]
    attentive_pct: float
    distracted_pct: float
    drowsy_pct: float
    taking_notes_pct: float
    trend_direction: str  # 'RISING', 'STABLE', 'FALLING'
    interval_history: List[SessionIntervalMetrics] = field(default_factory=list)


class MultimodalFusionEngine:
    """Fuses multi-sensory features into normalized engagement scores and session dynamics."""

    def __init__(self):
        self.config = get_config()
        self.w_visual = self.config.get("fusion.weights.visual_attention", 0.45)
        self.w_alertness = self.config.get("fusion.weights.alertness_motion", 0.25)
        self.w_audio = self.config.get("fusion.weights.acoustic_alignment", 0.30)
        self.smoothing_factor = self.config.get("fusion.smoothing_factor", 0.35)
        self.interval_sec = self.config.get("fusion.interval_window_sec", 5.0)

        self.last_interval_ts = 0.0
        self.interval_counter = 0
        self.smoothed_classroom_score = 75.0
        self.interval_history: List[SessionIntervalMetrics] = []
        self.score_window: List[float] = []

    def fuse(
        self,
        student_states: List[StudentVisualState],
        audio_state: AudioAcousticState,
        current_time_ts: float
    ) -> ClassroomDynamicsSnapshot:
        """Fuses vision and audio streams into classroom-level and student-level engagement."""
        now_iso = datetime.fromtimestamp(current_time_ts).strftime("%H:%M:%S")
        total_students = len(student_states)

        student_scores: Dict[str, float] = {}
        attentive_count = 0
        notes_count = 0
        distracted_count = 0
        drowsy_count = 0

        audio_factor = audio_state.audio_engagement_factor

        for st in student_states:
            # 1. Base visual score (0 - 100)
            vis = st.visual_score

            # 2. Modulate with acoustic context
            # In a discussion, high acoustic factor elevates score; during unwanted chatter, it decreases
            fused_student_score = (vis * 0.75) + (audio_factor * 100.0 * 0.25)
            fused_student_score = float(np.clip(fused_student_score, 0.0, 100.0))

            key = st.student_id or f"Track-{st.track_id}"
            student_scores[key] = round(fused_student_score, 1)

            if st.drowsiness_flag:
                drowsy_count += 1
            elif st.status_label == "Attentive":
                attentive_count += 1
            elif st.status_label == "Taking Notes":
                notes_count += 1
            else:
                distracted_count += 1

        # Calculate percentages
        if total_students > 0:
            att_pct = (attentive_count / total_students) * 100.0
            notes_pct = (notes_count / total_students) * 100.0
            dist_pct = (distracted_count / total_students) * 100.0
            drowsy_pct = (drowsy_count / total_students) * 100.0
            raw_classroom_score = float(np.mean(list(student_scores.values())))
        else:
            # Default baseline when no students currently in camera FOV
            att_pct, notes_pct, dist_pct, drowsy_pct = 0.0, 0.0, 0.0, 0.0
            raw_classroom_score = 70.0 * audio_factor

        # Temporal smoothing
        self.smoothed_classroom_score = (
            self.smoothing_factor * raw_classroom_score
            + (1.0 - self.smoothing_factor) * self.smoothed_classroom_score
        )

        # Track trend direction
        self.score_window.append(self.smoothed_classroom_score)
        if len(self.score_window) > 10:
            self.score_window.pop(0)

        trend = "STABLE"
        if len(self.score_window) >= 5:
            delta = self.score_window[-1] - self.score_window[0]
            if delta > 3.5:
                trend = "RISING"
            elif delta < -3.5:
                trend = "FALLING"

        # Check periodic interval recording (every interval_sec)
        if (current_time_ts - self.last_interval_ts) >= self.interval_sec:
            self.last_interval_ts = current_time_ts
            self.interval_counter += 1

            # Anomaly checks
            is_drop = self.smoothed_classroom_score < 48.0 or (trend == "FALLING" and self.smoothed_classroom_score < 55.0)
            is_peak = self.smoothed_classroom_score >= 82.0 or (audio_state.acoustic_environment == "CLASSROOM_DISCUSSION" and self.smoothed_classroom_score >= 75.0)

            interval_metric = SessionIntervalMetrics(
                interval_index=self.interval_counter,
                timestamp_iso=now_iso,
                aggregate_engagement_score=round(self.smoothed_classroom_score, 1),
                visual_attention_avg=round(float(np.mean([s.visual_score for s in student_states])) if student_states else 70.0, 1),
                acoustic_factor=round(audio_factor, 2),
                active_students_count=total_students,
                attentive_pct=round(att_pct, 1),
                taking_notes_pct=round(notes_pct, 1),
                distracted_pct=round(dist_pct, 1),
                drowsy_pct=round(drowsy_pct, 1),
                acoustic_environment=audio_state.acoustic_environment,
                is_attention_drop=is_drop,
                is_interaction_peak=is_peak,
            )
            self.interval_history.append(interval_metric)
            if len(self.interval_history) > 150:
                self.interval_history.pop(0)

        return ClassroomDynamicsSnapshot(
            timestamp_ts=current_time_ts,
            timestamp_iso=now_iso,
            overall_engagement_score=round(raw_classroom_score, 1),
            smoothed_engagement_score=round(self.smoothed_classroom_score, 1),
            active_students_count=total_students,
            audio_state=audio_state,
            student_scores=student_scores,
            student_states=student_states,
            attentive_pct=round(att_pct, 1),
            distracted_pct=round(dist_pct, 1),
            drowsy_pct=round(drowsy_pct, 1),
            taking_notes_pct=round(notes_pct, 1),
            trend_direction=trend,
            interval_history=self.interval_history,
        )
