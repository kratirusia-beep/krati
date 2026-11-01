"""
=============================================================================
Multimodal Attendance & Engagement Agent - Data Models
Epic 1: Project Architecture & Environment Initialization
Task 2: Privacy-Preserving Data Models & Schemas
=============================================================================
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
import json


@dataclass
class Student:
    student_id: str
    name: str
    roll_number: str
    email: str = ""
    face_embedding: Optional[List[float]] = None
    enrolled_at: str = field(default_factory=lambda: datetime.now().isoformat())
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get("face_embedding") is not None:
            # Mask embedding for safe json serialization in UI
            d["has_embedding"] = True
            d["face_embedding"] = None
        else:
            d["has_embedding"] = False
        return d


@dataclass
class Session:
    session_id: str
    title: str
    course: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ended_at: Optional[str] = None
    is_active: bool = True
    total_students_present: int = 0
    avg_engagement: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttendanceRecord:
    record_id: Optional[int]
    session_id: str
    student_id: str
    student_name: str
    first_seen: str
    last_seen: str
    sustained_seconds: float
    attendance_status: str  # 'PRESENT', 'ABSENT', 'LATE', 'INCOMPLETE'
    presence_ratio: float   # sustained_seconds / session_elapsed_seconds
    marked_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EngagementSnapshot:
    session_id: str
    student_id: str
    timestamp: str
    head_pose_yaw: float
    head_pose_pitch: float
    head_pose_score: float      # 0 to 100
    movement_score: float       # 0 to 100
    eye_alertness_score: float  # 0 to 100
    audio_energy: float         # 0 to 1.0
    engagement_score: float     # Combined 0 to 100
    status_label: str           # FOCUSED, DISTRACTED, DROWSY, RESTLESS

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemporalInterval:
    interval_id: Optional[int]
    session_id: str
    interval_index: int
    start_time: str
    end_time: str
    avg_engagement: float
    visual_attention_avg: float
    audio_activity_avg: float
    trend: str                  # 'STABLE', 'DROPPING', 'PEAKING'
    active_student_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TeachingSuggestion:
    suggestion_id: Optional[int]
    session_id: str
    timestamp: str
    category: str               # 'PACING', 'INTERACTION', 'BREAK', 'ATTENTION_CHECK'
    urgency: str                # 'INFO', 'SUGGESTION', 'ALERT'
    message: str
    trigger_metric: str
    action_recommended: str
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
