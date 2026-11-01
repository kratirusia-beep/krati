"""
=============================================================================
Multimodal Attendance & Engagement Agent - Local Database Engine
Epic 1: Project Architecture & Environment Initialization
Task 2: Privacy-Preserving SQLite Local Database Architecture
=============================================================================
"""

import sqlite3
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from config import DATABASE_PATH
from core.models import (
    Student,
    Session,
    AttendanceRecord,
    EngagementSnapshot,
    TemporalInterval,
    TeachingSuggestion,
)
from core.logger import logger


class DatabaseManager:
    """Thread-safe SQLite database manager for offline privacy-preserving storage."""

    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = str(db_path)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes tables for students, sessions, attendance, engagement, intervals, and suggestions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Students Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    roll_number TEXT UNIQUE NOT NULL,
                    email TEXT,
                    face_embedding TEXT,  -- JSON serialized float vector
                    enrolled_at TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1
                )
            """)

            # Sessions Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    course TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    is_active INTEGER DEFAULT 1,
                    total_students_present INTEGER DEFAULT 0,
                    avg_engagement REAL DEFAULT 0.0
                )
            """)

            # Attendance Records Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    student_id TEXT NOT NULL,
                    student_name TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    sustained_seconds REAL DEFAULT 0.0,
                    attendance_status TEXT NOT NULL,
                    presence_ratio REAL DEFAULT 0.0,
                    marked_at TEXT,
                    UNIQUE(session_id, student_id),
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id),
                    FOREIGN KEY(student_id) REFERENCES students(student_id)
                )
            """)

            # Engagement Snapshots Table (Time-series per student)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS engagement_snapshots (
                    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    student_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    head_pose_yaw REAL,
                    head_pose_pitch REAL,
                    head_pose_score REAL,
                    movement_score REAL,
                    eye_alertness_score REAL,
                    audio_energy REAL,
                    engagement_score REAL,
                    status_label TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            """)

            # Temporal Intervals Table (Aggregated classroom dynamics)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS temporal_intervals (
                    interval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    interval_index INTEGER NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    avg_engagement REAL NOT NULL,
                    visual_attention_avg REAL NOT NULL,
                    audio_activity_avg REAL NOT NULL,
                    trend TEXT NOT NULL,
                    active_student_count INTEGER NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            """)

            # Teaching Suggestions Table (AI Reasoning logs)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS teaching_suggestions (
                    suggestion_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL,
                    urgency TEXT NOT NULL,
                    message TEXT NOT NULL,
                    trigger_metric TEXT NOT NULL,
                    action_recommended TEXT NOT NULL,
                    acknowledged INTEGER DEFAULT 0,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            """)

            conn.commit()
            logger.info("SQLite Database initialized successfully.")

    # ---------------- Student Management ----------------
    def add_student(self, student: Student) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                embedding_json = json.dumps(student.face_embedding) if student.face_embedding else None
                cursor.execute("""
                    INSERT INTO students (student_id, name, roll_number, email, face_embedding, enrolled_at, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(student_id) DO UPDATE SET
                        name=excluded.name,
                        roll_number=excluded.roll_number,
                        email=excluded.email,
                        face_embedding=coalesce(excluded.face_embedding, students.face_embedding),
                        is_active=excluded.is_active
                """, (
                    student.student_id,
                    student.name,
                    student.roll_number,
                    student.email,
                    embedding_json,
                    student.enrolled_at,
                    1 if student.is_active else 0,
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error adding student: {e}")
            return False

    def get_student(self, student_id: str) -> Optional[Student]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
            row = cursor.fetchone()
            if row:
                embedding = json.loads(row["face_embedding"]) if row["face_embedding"] else None
                return Student(
                    student_id=row["student_id"],
                    name=row["name"],
                    roll_number=row["roll_number"],
                    email=row["email"] or "",
                    face_embedding=embedding,
                    enrolled_at=row["enrolled_at"],
                    is_active=bool(row["is_active"]),
                )
        return None

    def get_all_students(self) -> List[Student]:
        students = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM students WHERE is_active = 1 ORDER BY name ASC")
            for row in cursor.fetchall():
                embedding = json.loads(row["face_embedding"]) if row["face_embedding"] else None
                students.append(
                    Student(
                        student_id=row["student_id"],
                        name=row["name"],
                        roll_number=row["roll_number"],
                        email=row["email"] or "",
                        face_embedding=embedding,
                        enrolled_at=row["enrolled_at"],
                        is_active=bool(row["is_active"]),
                    )
                )
        return students

    # ---------------- Session Management ----------------
    def create_session(self, session: Session):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (session_id, title, course, started_at, ended_at, is_active, total_students_present, avg_engagement)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.title,
                session.course,
                session.started_at,
                session.ended_at,
                1 if session.is_active else 0,
                session.total_students_present,
                session.avg_engagement,
            ))
            conn.commit()

    def get_active_session(self) -> Optional[Session]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE is_active = 1 ORDER BY started_at DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                return Session(
                    session_id=row["session_id"],
                    title=row["title"],
                    course=row["course"],
                    started_at=row["started_at"],
                    ended_at=row["ended_at"],
                    is_active=bool(row["is_active"]),
                    total_students_present=row["total_students_present"],
                    avg_engagement=row["avg_engagement"],
                )
        return None

    def end_session(self, session_id: str, avg_engagement: float = 0.0):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Count total present
            cursor.execute("SELECT COUNT(*) as count FROM attendance WHERE session_id = ? AND attendance_status = 'PRESENT'", (session_id,))
            count = cursor.fetchone()["count"]

            cursor.execute("""
                UPDATE sessions 
                SET is_active = 0, ended_at = ?, total_students_present = ?, avg_engagement = ?
                WHERE session_id = ?
            """, (datetime.now().isoformat(), count, avg_engagement, session_id))
            conn.commit()

    def get_all_sessions(self) -> List[Session]:
        sessions = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions ORDER BY started_at DESC")
            for row in cursor.fetchall():
                sessions.append(
                    Session(
                        session_id=row["session_id"],
                        title=row["title"],
                        course=row["course"],
                        started_at=row["started_at"],
                        ended_at=row["ended_at"],
                        is_active=bool(row["is_active"]),
                        total_students_present=row["total_students_present"],
                        avg_engagement=row["avg_engagement"],
                    )
                )
        return sessions

    # ---------------- Attendance Management (Scenario 1) ----------------
    def upsert_attendance(self, record: AttendanceRecord):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO attendance (session_id, student_id, student_name, first_seen, last_seen, sustained_seconds, attendance_status, presence_ratio, marked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, student_id) DO UPDATE SET
                    student_name = excluded.student_name,
                    last_seen = excluded.last_seen,
                    sustained_seconds = excluded.sustained_seconds,
                    attendance_status = excluded.attendance_status,
                    presence_ratio = excluded.presence_ratio,
                    marked_at = coalesce(attendance.marked_at, excluded.marked_at)
            """, (
                record.session_id,
                record.student_id,
                record.student_name,
                record.first_seen,
                record.last_seen,
                record.sustained_seconds,
                record.attendance_status,
                record.presence_ratio,
                record.marked_at,
            ))
            conn.commit()

    def get_session_attendance(self, session_id: str) -> List[AttendanceRecord]:
        records = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM attendance WHERE session_id = ? ORDER BY student_name ASC", (session_id,))
            for row in cursor.fetchall():
                records.append(
                    AttendanceRecord(
                        record_id=row["record_id"],
                        session_id=row["session_id"],
                        student_id=row["student_id"],
                        student_name=row["student_name"],
                        first_seen=row["first_seen"],
                        last_seen=row["last_seen"],
                        sustained_seconds=row["sustained_seconds"],
                        attendance_status=row["attendance_status"],
                        presence_ratio=row["presence_ratio"],
                        marked_at=row["marked_at"],
                    )
                )
        return records

    # ---------------- Engagement Snapshots & Intervals (Scenario 2) ----------------
    def insert_engagement_snapshot(self, snapshot: EngagementSnapshot):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO engagement_snapshots (
                    session_id, student_id, timestamp, head_pose_yaw, head_pose_pitch,
                    head_pose_score, movement_score, eye_alertness_score, audio_energy,
                    engagement_score, status_label
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.session_id,
                snapshot.student_id,
                snapshot.timestamp,
                snapshot.head_pose_yaw,
                snapshot.head_pose_pitch,
                snapshot.head_pose_score,
                snapshot.movement_score,
                snapshot.eye_alertness_score,
                snapshot.audio_energy,
                snapshot.engagement_score,
                snapshot.status_label,
            ))
            conn.commit()

    def insert_temporal_interval(self, interval: TemporalInterval):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO temporal_intervals (
                    session_id, interval_index, start_time, end_time, avg_engagement,
                    visual_attention_avg, audio_activity_avg, trend, active_student_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                interval.session_id,
                interval.interval_index,
                interval.start_time,
                interval.end_time,
                interval.avg_engagement,
                interval.visual_attention_avg,
                interval.audio_activity_avg,
                interval.trend,
                interval.active_student_count,
            ))
            conn.commit()

    def get_session_intervals(self, session_id: str) -> List[TemporalInterval]:
        intervals = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM temporal_intervals WHERE session_id = ? ORDER BY interval_index ASC", (session_id,))
            for row in cursor.fetchall():
                intervals.append(
                    TemporalInterval(
                        interval_id=row["interval_id"],
                        session_id=row["session_id"],
                        interval_index=row["interval_index"],
                        start_time=row["start_time"],
                        end_time=row["end_time"],
                        avg_engagement=row["avg_engagement"],
                        visual_attention_avg=row["visual_attention_avg"],
                        audio_activity_avg=row["audio_activity_avg"],
                        trend=row["trend"],
                        active_student_count=row["active_student_count"],
                    )
                )
        return intervals

    # ---------------- Teaching Suggestions (Scenario 3) ----------------
    def insert_teaching_suggestion(self, suggestion: TeachingSuggestion):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO teaching_suggestions (
                    session_id, timestamp, category, urgency, message, trigger_metric, action_recommended, acknowledged
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                suggestion.session_id,
                suggestion.timestamp,
                suggestion.category,
                suggestion.urgency,
                suggestion.message,
                suggestion.trigger_metric,
                suggestion.action_recommended,
                1 if suggestion.acknowledged else 0,
            ))
            conn.commit()

    def get_session_suggestions(self, session_id: str) -> List[TeachingSuggestion]:
        suggestions = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM teaching_suggestions WHERE session_id = ? ORDER BY timestamp DESC", (session_id,))
            for row in cursor.fetchall():
                suggestions.append(
                    TeachingSuggestion(
                        suggestion_id=row["suggestion_id"],
                        session_id=row["session_id"],
                        timestamp=row["timestamp"],
                        category=row["category"],
                        urgency=row["urgency"],
                        message=row["message"],
                        trigger_metric=row["trigger_metric"],
                        action_recommended=row["action_recommended"],
                        acknowledged=bool(row["acknowledged"]),
                    )
                )
        return suggestions


db = DatabaseManager()
