"""
Automated Sustained Presence Attendance Engine.
Epic 4: Attendance Engine & Sustained Presence (Task 7 & 8)
Filters transient frame spikes, tracks time-in-seat, and exports verifiable attendance logs.
"""

import os
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.config import get_config
from src.enrollment.student_registry import StudentRegistry
import numpy as np


@dataclass
class AttendanceRecord:
    student_id: str
    student_name: str
    roll_no: str
    department: str
    first_seen_iso: Optional[str] = None
    last_seen_iso: Optional[str] = None
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0
    cumulative_seconds_present: float = 0.0
    total_frames_seen: int = 0
    status: str = "ABSENT"  # 'PRESENT', 'LATE', 'PARTIAL', 'ABSENT'
    sustained_presence_verified: bool = False
    average_engagement_score: float = 0.0
    engagement_samples: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "student_id": self.student_id,
            "student_name": self.student_name,
            "roll_no": self.roll_no,
            "department": self.department,
            "first_seen": self.first_seen_iso or "N/A",
            "last_seen": self.last_seen_iso or "N/A",
            "active_duration_min": round(self.cumulative_seconds_present / 60.0, 2),
            "status": self.status,
            "sustained_verified": self.sustained_presence_verified,
            "avg_engagement": round(self.average_engagement_score, 1),
        }


@dataclass
class AttendanceSummary:
    session_id: str
    session_start_iso: str
    session_duration_sec: float
    total_enrolled: int
    present_count: int
    late_count: int
    partial_count: int
    absent_count: int
    overall_attendance_rate: float
    records: List[AttendanceRecord]


class AttendanceEngine:
    """Calculates sustained presence and logs attendance with temporal verification."""

    def __init__(self, registry: StudentRegistry):
        self.config = get_config()
        self.registry = registry
        self.min_sustained_sec = self.config.get("attendance.min_sustained_seconds", 15.0)
        self.grace_period_sec = self.config.get("attendance.session_grace_period_sec", 300.0)

        self.session_id = f"SESSION-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.session_start_time = datetime.now()
        self.session_start_ts = self.session_start_time.timestamp()

        # Initialize student records from registry
        self.records: Dict[str, AttendanceRecord] = {}
        self._init_records()

    def _init_records(self):
        for student in self.registry.students.values():
            self.records[student.student_id] = AttendanceRecord(
                student_id=student.student_id,
                student_name=student.name,
                roll_no=student.roll_no,
                department=student.department,
            )

    def refresh_enrolled_students(self):
        """Ensures any newly enrolled students during live session are tracked."""
        for s_id, student in self.registry.students.items():
            if s_id not in self.records:
                self.records[s_id] = AttendanceRecord(
                    student_id=student.student_id,
                    student_name=student.name,
                    roll_no=student.roll_no,
                    department=student.department,
                )

    def update_presence(
        self,
        student_id: str,
        current_time_ts: float,
        dt_seconds: float,
        current_engagement: float = 75.0
    ):
        """
        Registers presence increment for an identified student.
        Prevents single-frame false positives via sustained duration accumulation.
        """
        self.refresh_enrolled_students()
        if student_id not in self.records:
            return

        rec = self.records[student_id]
        now_iso = datetime.fromtimestamp(current_time_ts).strftime("%Y-%m-%d %H:%M:%S")

        if rec.first_seen_ts == 0.0:
            rec.first_seen_ts = current_time_ts
            rec.first_seen_iso = now_iso

        rec.last_seen_ts = current_time_ts
        rec.last_seen_iso = now_iso
        rec.cumulative_seconds_present += dt_seconds
        rec.total_frames_seen += 1

        rec.engagement_samples.append(current_engagement)
        if len(rec.engagement_samples) > 200:
            rec.engagement_samples.pop(0)
        rec.average_engagement_score = float(np.mean(rec.engagement_samples))

        # Check sustained presence rule
        if rec.cumulative_seconds_present >= self.min_sustained_sec:
            rec.sustained_presence_verified = True
            time_since_start = rec.first_seen_ts - self.session_start_ts
            if time_since_start <= self.grace_period_sec:
                rec.status = "PRESENT"
            else:
                rec.status = "LATE"
        elif rec.cumulative_seconds_present >= (self.min_sustained_sec * 0.4):
            rec.status = "PARTIAL"
        else:
            rec.status = "ABSENT"

    def get_summary(self, current_ts: Optional[float] = None) -> AttendanceSummary:
        """Returns session attendance breakdown."""
        self.refresh_enrolled_students()
        cur_ts = current_ts or datetime.now().timestamp()
        duration = max(0.0, cur_ts - self.session_start_ts)

        present = sum(1 for r in self.records.values() if r.status == "PRESENT")
        late = sum(1 for r in self.records.values() if r.status == "LATE")
        partial = sum(1 for r in self.records.values() if r.status == "PARTIAL")
        absent = sum(1 for r in self.records.values() if r.status == "ABSENT")
        total = len(self.records)

        rate = ((present + late) / total * 100.0) if total > 0 else 0.0

        return AttendanceSummary(
            session_id=self.session_id,
            session_start_iso=self.session_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            session_duration_sec=duration,
            total_enrolled=total,
            present_count=present,
            late_count=late,
            partial_count=partial,
            absent_count=absent,
            overall_attendance_rate=round(rate, 1),
            records=list(self.records.values()),
        )

    def export_csv(self, output_dir: Optional[str] = None) -> str:
        """Exports verified attendance records to CSV."""
        if output_dir is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            output_dir = str(base_dir / "data" / "attendance_logs")
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{self.session_id}_attendance.csv"
        file_path = os.path.join(output_dir, filename)

        summary = self.get_summary()
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Session ID", summary.session_id])
            writer.writerow(["Session Start", summary.session_start_iso])
            writer.writerow(["Attendance Rate", f"{summary.overall_attendance_rate}%"])
            writer.writerow([])
            writer.writerow([
                "Student ID", "Name", "Roll No", "Department",
                "First Seen", "Last Seen", "Active Duration (Mins)",
                "Attendance Status", "Sustained Verified", "Avg Engagement (%)"
            ])
            for rec in summary.records:
                writer.writerow([
                    rec.student_id,
                    rec.student_name,
                    rec.roll_no,
                    rec.department,
                    rec.first_seen_iso or "N/A",
                    rec.last_seen_iso or "N/A",
                    round(rec.cumulative_seconds_present / 60.0, 2),
                    rec.status,
                    "YES" if rec.sustained_presence_verified else "NO",
                    round(rec.average_engagement_score, 1)
                ])

        print(f"[AttendanceEngine] Attendance exported to: {file_path}")
        return file_path
