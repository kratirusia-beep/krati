"""
Generates sample student roster, runs a simulated session, and outputs sample attendance logs.
"""

import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.enrollment.student_registry import StudentRegistry
from src.attendance.attendance_engine import AttendanceEngine


def main():
    print("[Sample Data] Seeding student registry...")
    registry = StudentRegistry()
    registry.seed_sample_students()
    students = registry.list_all()
    print(f"Registered {len(students)} students in offline store.")

    print("\n[Sample Data] Simulating classroom session...")
    engine = AttendanceEngine(registry)

    # Simulate 20 seconds of sustained presence for 4 students, and short presence for 1
    t0 = time.time()
    for tick in range(18):
        current_t = t0 + tick
        # STU-101 (Present)
        engine.update_presence("STU-101", current_t, 1.0, current_engagement=88.0)
        # STU-102 (Present)
        engine.update_presence("STU-102", current_t, 1.0, current_engagement=82.0)
        # STU-103 (Present)
        engine.update_presence("STU-103", current_t, 1.0, current_engagement=72.0)
        # STU-104 (Present)
        engine.update_presence("STU-104", current_t, 1.0, current_engagement=65.0)

    # STU-105 (walked past camera briefly: 4 seconds -> partial/absent, not sustained)
    for tick in range(4):
        engine.update_presence("STU-105", t0 + tick, 1.0, current_engagement=50.0)

    summary = engine.get_summary(t0 + 20)
    print("\n[Sample Data] Session Summary Generated:")
    print(f"   Session ID: {summary.session_id}")
    print(f"   Total Enrolled: {summary.total_enrolled}")
    print(f"   Present: {summary.present_count}")
    print(f"   Partial: {summary.partial_count}")
    print(f"   Absent: {summary.absent_count}")
    print(f"   Attendance Rate: {summary.overall_attendance_rate}%")

    csv_path = engine.export_csv()
    print(f"\n[SUCCESS] Sample verified attendance log created at: {csv_path}")


if __name__ == "__main__":
    main()
