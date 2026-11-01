"""
Pedagogical Intervention & Explainable Teaching Suggestions.
Epic 8: AI Agent Reasoning & Decision Support (Task 15 & 16)
Defines structured actionable pedagogical interventions with explainable multimodal rationale.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TeachingSuggestion:
    suggestion_id: str
    timestamp_iso: str
    category: str      # 'Pacing', 'Interaction', 'Fatigue / Attention', 'Classroom Management', 'Reinforcement'
    urgency: str       # 'HIGH', 'MEDIUM', 'LOW', 'INFO'
    title: str
    action_prompt: str
    pedagogical_rationale: str
    multimodal_evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.suggestion_id,
            "timestamp": self.timestamp_iso,
            "category": self.category,
            "urgency": self.urgency,
            "title": self.title,
            "action_prompt": self.action_prompt,
            "rationale": self.pedagogical_rationale,
            "evidence": self.multimodal_evidence,
        }


class TeachingAdvisor:
    """Catalog of evidence-based pedagogical heuristics based on modern instructional design."""

    @staticmethod
    def create_attention_dip_suggestion(
        timestamp_iso: str, score: float, drowsy_pct: float, distracted_pct: float
    ) -> TeachingSuggestion:
        return TeachingSuggestion(
            suggestion_id=f"SUG-ATTN-{timestamp_iso.replace(':', '')}",
            timestamp_iso=timestamp_iso,
            category="Interaction",
            urgency="HIGH" if score < 40.0 else "MEDIUM",
            title="Classroom Attention Drop Detected",
            action_prompt="Launch a 2-minute Think-Pair-Share or an interactive concept poll to re-engage the room.",
            pedagogical_rationale=(
                "Learner attention naturally dips every 15-20 minutes during continuous passive listening. "
                "Active retrieval practices rapidly reset learner cognitive readiness."
            ),
            multimodal_evidence={
                "aggregate_engagement": f"{score}%",
                "distracted_students": f"{distracted_pct}%",
                "drowsy_students": f"{drowsy_pct}%",
                "visual_gaze_alignment": "Deviated from board / screen",
            },
        )

    @staticmethod
    def create_fatigue_suggestion(
        timestamp_iso: str, drowsy_pct: float, stillness_sec: float
    ) -> TeachingSuggestion:
        return TeachingSuggestion(
            suggestion_id=f"SUG-FATIGUE-{timestamp_iso.replace(':', '')}",
            timestamp_iso=timestamp_iso,
            category="Fatigue / Attention",
            urgency="HIGH",
            title="Elevated Drowsiness & Stillness Cluster",
            action_prompt="Initiate a 60-second physical stretch break or invite quick verbal responses from back rows.",
            pedagogical_rationale=(
                "Physiological stillness combined with lowered eye aspect ratio indicates mental fatigue and oxygenation slump. "
                "Kinesthetic breaks immediately enhance cerebral blood flow."
            ),
            multimodal_evidence={
                "drowsy_ratio": f"{drowsy_pct}%",
                "avg_stillness_duration": f"{stillness_sec}s",
                "behavioral_indicator": "Low eye blink activity + slumped head pitch",
            },
        )

    @staticmethod
    def create_monologue_suggestion(
        timestamp_iso: str, monologue_min: float
    ) -> TeachingSuggestion:
        return TeachingSuggestion(
            suggestion_id=f"SUG-MONO-{timestamp_iso.replace(':', '')}",
            timestamp_iso=timestamp_iso,
            category="Pacing",
            urgency="MEDIUM",
            title="Pacing Warning: Extended Monologue",
            action_prompt="Pause your lecture delivery. Pose an open-ended comprehension question to a student.",
            pedagogical_rationale=(
                "Continuous teacher talk time exceeding 10 minutes leads to passive disassociation. "
                "Socratic questioning rebalances speaking ratios."
            ),
            multimodal_evidence={
                "uninterrupted_speech_duration": f"{monologue_min} mins",
                "acoustic_environment": "INSTRUCTOR_MONOLOGUE",
                "student_verbal_interactions": 0,
            },
        )

    @staticmethod
    def create_chatter_suggestion(
        timestamp_iso: str, chatter_index: float, distracted_pct: float
    ) -> TeachingSuggestion:
        return TeachingSuggestion(
            suggestion_id=f"SUG-CHATTER-{timestamp_iso.replace(':', '')}",
            timestamp_iso=timestamp_iso,
            category="Classroom Management",
            urgency="MEDIUM",
            title="Ambient Side-Chatter Spiking",
            action_prompt="Gently refocus student attention with a clear transition phrase (e.g. 'Eyes on the main slide').",
            pedagogical_rationale=(
                "Side chatter during instructional explanation disrupts working memory for neighboring students. "
                "Early polite redirection prevents session derailment."
            ),
            multimodal_evidence={
                "acoustic_chatter_index": f"{int(chatter_index * 100)}%",
                "distracted_students": f"{distracted_pct}%",
                "audio_profile": "High spectral flux + sideways head yaw",
            },
        )

    @staticmethod
    def create_optimal_praise(
        timestamp_iso: str, score: float
    ) -> TeachingSuggestion:
        return TeachingSuggestion(
            suggestion_id=f"SUG-PEAK-{timestamp_iso.replace(':', '')}",
            timestamp_iso=timestamp_iso,
            category="Reinforcement",
            urgency="INFO",
            title="Optimal Flow & Engagement Zone",
            action_prompt="Current instructional format is exceptionally effective. Maintain this delivery pacing.",
            pedagogical_rationale=(
                "Synchronized high visual attention and active acoustic discussion indicates peak pedagogical resonance."
            ),
            multimodal_evidence={
                "aggregate_engagement": f"{score}%",
                "interaction_peak": "Confirmed",
            },
        )
