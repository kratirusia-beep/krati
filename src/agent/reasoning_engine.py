"""
Agent Reasoning Core for Classroom Intelligence.
Epic 8: AI Agent Reasoning & Teaching Decision Support (Task 15 & 16)
Evaluates temporal patterns and generates context-aware pedagogical suggestions.
"""

import time
from typing import List, Dict, Optional
from src.config import get_config
from src.fusion.multimodal_fusion import ClassroomDynamicsSnapshot
from .teaching_advisor import TeachingSuggestion, TeachingAdvisor


class AgentReasoningEngine:
    """Intelligent reasoning core that evaluates classroom dynamics against pedagogical rules."""

    def __init__(self):
        self.config = get_config()
        self.advisor = TeachingAdvisor()
        self.last_eval_time = 0.0
        self.eval_interval = self.config.get("agent_reasoning.evaluation_interval_sec", 8.0)

        # Cooldown trackers per alert type (in seconds) to avoid notification spam
        self.cooldowns: Dict[str, float] = {
            "attention_drop": 0.0,
            "fatigue_cluster": 0.0,
            "lecture_monologue": 0.0,
            "unwanted_chatter": 0.0,
            "peak_engagement": 0.0,
        }
        self.COOLDOWN_PERIOD = 45.0  # seconds between repeated recommendations

        self.active_suggestions: List[TeachingSuggestion] = []
        self.history_suggestions: List[TeachingSuggestion] = []

    def evaluate(self, snapshot: ClassroomDynamicsSnapshot) -> Optional[TeachingSuggestion]:
        """Evaluates current snapshot and generates a suggestion if thresholds are triggered."""
        now = time.time()
        if (now - self.last_eval_time) < self.eval_interval:
            return None
        self.last_eval_time = now

        new_suggestion: Optional[TeachingSuggestion] = None

        # Rule 1: Attention Drop Alert
        drop_rule = self.config.get("agent_reasoning.rules.attention_drop", {})
        thresh_score = drop_rule.get("threshold_engagement", 45.0)
        if snapshot.smoothed_engagement_score < thresh_score:
            if (now - self.cooldowns["attention_drop"]) > self.COOLDOWN_PERIOD:
                self.cooldowns["attention_drop"] = now
                new_suggestion = self.advisor.create_attention_dip_suggestion(
                    timestamp_iso=snapshot.timestamp_iso,
                    score=snapshot.smoothed_engagement_score,
                    drowsy_pct=snapshot.drowsy_pct,
                    distracted_pct=snapshot.distracted_pct,
                )

        # Rule 2: Fatigue / Drowsiness Cluster
        fatigue_rule = self.config.get("agent_reasoning.rules.fatigue_cluster", {})
        thresh_inactive = fatigue_rule.get("threshold_inactive_ratio", 0.30)
        if not new_suggestion and (snapshot.drowsy_pct / 100.0) >= thresh_inactive:
            if (now - self.cooldowns["fatigue_cluster"]) > self.COOLDOWN_PERIOD:
                self.cooldowns["fatigue_cluster"] = now
                avg_stillness = 50.0
                new_suggestion = self.advisor.create_fatigue_suggestion(
                    timestamp_iso=snapshot.timestamp_iso,
                    drowsy_pct=snapshot.drowsy_pct,
                    stillness_sec=avg_stillness,
                )

        # Rule 3: Extended Lecture Monologue Pacing Alert
        mono_rule = self.config.get("agent_reasoning.rules.lecture_monologue", {})
        max_speech_sec = mono_rule.get("max_speech_duration_sec", 600.0)
        if (
            not new_suggestion
            and snapshot.audio_state.acoustic_environment == "INSTRUCTOR_MONOLOGUE"
            and snapshot.audio_state.speech_duration_sec >= max_speech_sec
        ):
            if (now - self.cooldowns["lecture_monologue"]) > self.COOLDOWN_PERIOD:
                self.cooldowns["lecture_monologue"] = now
                mins = round(snapshot.audio_state.speech_duration_sec / 60.0, 1)
                new_suggestion = self.advisor.create_monologue_suggestion(
                    timestamp_iso=snapshot.timestamp_iso,
                    monologue_min=mins,
                )

        # Rule 4: Side-Chatter Classroom Management
        chatter_rule = self.config.get("agent_reasoning.rules.unwanted_chatter", {})
        thresh_chatter = chatter_rule.get("chatter_threshold", 0.10)
        if (
            not new_suggestion
            and snapshot.audio_state.acoustic_environment == "SIDE_CHATTER"
            and snapshot.audio_state.chatter_index >= thresh_chatter
            and snapshot.distracted_pct > 25.0
        ):
            if (now - self.cooldowns["unwanted_chatter"]) > self.COOLDOWN_PERIOD:
                self.cooldowns["unwanted_chatter"] = now
                new_suggestion = self.advisor.create_chatter_suggestion(
                    timestamp_iso=snapshot.timestamp_iso,
                    chatter_index=snapshot.audio_state.chatter_index,
                    distracted_pct=snapshot.distracted_pct,
                )

        # Rule 5: Peak Engagement Reinforcement
        peak_rule = self.config.get("agent_reasoning.rules.peak_engagement", {})
        thresh_peak = peak_rule.get("threshold_engagement", 82.0)
        if not new_suggestion and snapshot.smoothed_engagement_score >= thresh_peak:
            if (now - self.cooldowns["peak_engagement"]) > (self.COOLDOWN_PERIOD * 2.0):
                self.cooldowns["peak_engagement"] = now
                new_suggestion = self.advisor.create_optimal_praise(
                    timestamp_iso=snapshot.timestamp_iso,
                    score=snapshot.smoothed_engagement_score,
                )

        if new_suggestion:
            self.active_suggestions.insert(0, new_suggestion)
            self.history_suggestions.insert(0, new_suggestion)
            if len(self.active_suggestions) > 5:
                self.active_suggestions.pop()
            if len(self.history_suggestions) > 50:
                self.history_suggestions.pop()

        return new_suggestion

    def get_latest_suggestions(self) -> List[Dict]:
        return [s.to_dict() for s in self.active_suggestions]

    def get_all_suggestions(self) -> List[Dict]:
        return [s.to_dict() for s in self.history_suggestions]
