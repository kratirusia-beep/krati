"""
Configuration loader and validator for Multimodal Attendance & Engagement Agent.
Epic 1: System Foundation & Configuration (Task 1 & 2)
"""

import os
from pathlib import Path
from typing import Any, Dict
import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "system": {
        "app_name": "Multimodal Attendance & Engagement Agent",
        "version": "1.0.0",
        "offline_mode": True,
        "privacy_preservation": {
            "anonymize_biometrics": True,
            "hash_algorithm": "sha256",
            "store_raw_photos": False,
            "data_retention_days": 30,
        },
    },
    "vision": {
        "detector_backend": "opencv_haar",
        "min_face_size": [40, 40],
        "detection_confidence": 0.60,
        "track_history_len": 30,
        "max_missing_frames": 25,
        "head_pose": {
            "yaw_limit": 28.0,
            "pitch_min": -22.0,
            "pitch_max": 18.0,
            "roll_limit": 25.0,
        },
        "inactivity": {
            "stillness_threshold": 0.04,
            "inactivity_alert_sec": 45.0,
            "drowsiness_ear_threshold": 0.20,
        },
    },
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
        "chunk_duration_sec": 1.0,
        "device_index": None,
        "silence_rms_threshold": 0.015,
        "voice_activity_threshold": 0.040,
        "classroom_chatter_threshold": 0.090,
    },
    "attendance": {
        "min_sustained_seconds": 15.0,
        "sustained_ratio_threshold": 0.50,
        "session_grace_period_sec": 300,
        "status_categories": {
            "present": "PRESENT",
            "late": "LATE",
            "partial": "PARTIAL",
            "absent": "ABSENT",
        },
    },
    "fusion": {
        "interval_window_sec": 5.0,
        "weights": {
            "visual_attention": 0.45,
            "alertness_motion": 0.25,
            "acoustic_alignment": 0.30,
        },
        "smoothing_factor": 0.35,
    },
    "agent_reasoning": {
        "evaluation_interval_sec": 10.0,
        "rules": {
            "attention_drop": {
                "threshold_engagement": 45.0,
                "duration_sec": 15.0,
                "suggestion": "Attention Dip Detected: Transition to an interactive think-pair-share or live question poll.",
            },
            "fatigue_cluster": {
                "threshold_inactive_ratio": 0.35,
                "suggestion": "Fatigue/Drowsiness Cluster: Consider a 60-second stand-and-stretch or call on students for brief input.",
            },
            "lecture_monologue": {
                "max_speech_duration_sec": 600,
                "suggestion": "Extended Monologue Alert: You have been lecturing for 10+ mins without student interaction. Pose a concept check question.",
            },
            "unwanted_chatter": {
                "chatter_threshold": 0.10,
                "suggestion": "Side-Chatter Alert: Elevated ambient chatter detected during focus interval. Refocus learner attention.",
            },
            "peak_engagement": {
                "threshold_engagement": 80.0,
                "suggestion": "Optimal Engagement Zone: Current delivery format is highly effective. Maintain current session pace.",
            },
        },
    },
}


class Config:
    """Singleton wrapper for system configuration."""
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load(config_path)
        return cls._instance

    def _load(self, config_path: str = None):
        if config_path is None:
            # Look relative to workspace
            base_dir = Path(__file__).resolve().parent.parent
            config_path = base_dir / "config" / "settings.yaml"

        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    user_conf = yaml.safe_load(f) or {}
                    self._config = self._deep_merge(DEFAULT_CONFIG, user_conf)
            except Exception as e:
                print(f"[Config] Error loading {config_path}: {e}. Using defaults.")
                self._config = DEFAULT_CONFIG.copy()
        else:
            self._config = DEFAULT_CONFIG.copy()

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = base.copy()
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = self._deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        val = self._config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    @property
    def raw(self) -> Dict[str, Any]:
        return self._config


def get_config(config_path: str = None) -> Config:
    return Config(config_path)
