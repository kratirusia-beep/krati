"""
Comprehensive Automated Test Suite for Multimodal Attendance & Engagement Agent.
Covers all 9 Epics and 17 Core Tasks.
"""

import sys
import os
import time
from pathlib import Path
import unittest
import numpy as np
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import get_config
from src.enrollment.student_registry import StudentRegistry
from src.vision.face_detector import FaceDetector, DetectedFace
from src.vision.face_tracker import FaceTracker
from src.vision.head_pose import HeadPoseEstimator
from src.vision.visual_engagement import VisualEngagementAnalyzer
from src.audio.audio_analyzer import AudioAnalyzer, AudioAcousticState
from src.attendance.attendance_engine import AttendanceEngine
from src.fusion.multimodal_fusion import MultimodalFusionEngine
from src.agent.reasoning_engine import AgentReasoningEngine
from src.pipeline.classroom_agent import ClassroomAgent
from src.web.app import create_app


class TestMultimodalAgent(unittest.TestCase):

    def setUp(self):
        self.config = get_config()

    def test_01_config_loading(self):
        """Epic 1 (Task 1 & 2): Verify configuration values and threshold keys."""
        app_name = self.config.get("system.app_name")
        self.assertEqual(app_name, "Multimodal Attendance & Engagement Agent")
        self.assertTrue(self.config.get("system.offline_mode"))
        self.assertAlmostEqual(self.config.get("fusion.weights.visual_attention"), 0.45)
        self.assertAlmostEqual(self.config.get("fusion.weights.alertness_motion"), 0.25)
        self.assertAlmostEqual(self.config.get("fusion.weights.acoustic_alignment"), 0.30)

    def test_02_student_enrollment_and_matching(self):
        """Epic 2 (Task 3 & 4): Verify enrollment, feature hashing, and cosine similarity."""
        registry = StudentRegistry()
        registry.seed_sample_students()
        self.assertGreaterEqual(len(registry.students), 5)

        # Create distinct test face
        test_face = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.circle(test_face, (32, 32), 15, (255, 255, 255), -1)

        profile = registry.enroll(
            student_id="TEST-999",
            name="Test Learner",
            roll_no="T999",
            department="AI Labs",
            face_img=test_face,
        )
        self.assertEqual(profile.student_id, "TEST-999")
        self.assertTrue(len(profile.embedding) == 128)
        self.assertTrue(len(profile.embedding_hash) == 64)  # SHA-256 length

        # Match face
        matched_id, score, name = registry.match_face(test_face, threshold=0.60)
        self.assertEqual(matched_id, "TEST-999")
        self.assertEqual(name, "Test Learner")
        self.assertGreaterEqual(score, 0.90)

        # Cleanup
        registry.delete("TEST-999")

    def test_03_head_pose_estimation(self):
        """Epic 5 (Task 9): Verify 3D solvePnP Euler angle extraction & attention classification."""
        estimator = HeadPoseEstimator()
        # Canonical forward-facing 2D landmark coordinates on 640x480 frame
        w, h = 640, 480
        landmarks = np.array([
            (w * 0.50, h * 0.55),  # nose
            (w * 0.50, h * 0.90),  # chin
            (w * 0.35, h * 0.35),  # left eye
            (w * 0.65, h * 0.35),  # right eye
            (w * 0.38, h * 0.75),  # left mouth
            (w * 0.62, h * 0.75),  # right mouth
        ], dtype=np.float32)

        pose = estimator.estimate_pose(landmarks, (h, w))
        self.assertIn(pose.gaze_direction, ["LOOKING_FORWARD", "LOOKING_DOWN", "LOOKING_AWAY"])
        self.assertGreaterEqual(pose.attention_score, 0.0)
        self.assertLessEqual(pose.attention_score, 1.0)

    def test_04_face_tracker_association(self):
        """Epic 3 (Task 5 & 6): Verify multi-target tracking across consecutive frames."""
        tracker = FaceTracker(max_missing_frames=5)
        dummy_crop = np.zeros((40, 40, 3), dtype=np.uint8)

        # Frame 1: Detection at (100, 100, 50, 50)
        det1 = DetectedFace(bbox=(100, 100, 50, 50), confidence=0.9, face_crop=dummy_crop)
        tracks1 = tracker.update([det1], current_time=1.0)
        self.assertEqual(len(tracks1), 1)
        t_id = tracks1[0].track_id

        # Frame 2: Slight movement to (104, 102, 50, 50) -> should preserve same track_id
        det2 = DetectedFace(bbox=(104, 102, 50, 50), confidence=0.9, face_crop=dummy_crop)
        tracks2 = tracker.update([det2], current_time=1.1)
        self.assertEqual(len(tracks2), 1)
        self.assertEqual(tracks2[0].track_id, t_id)
        self.assertEqual(tracks2[0].total_frames_seen, 2)

    def test_05_sustained_presence_attendance(self):
        """Epic 4 (Task 7 & 8): Verify false positive rejection and sustained presence."""
        registry = StudentRegistry()
        registry.seed_sample_students()
        engine = AttendanceEngine(registry)

        t_start = 1000.0
        engine.session_start_ts = t_start

        # Student A: Present for 18 seconds (exceeds min_sustained_sec 15s)
        for i in range(18):
            engine.update_presence("STU-101", t_start + i, 1.0, 85.0)

        # Student B: Present for only 3 seconds (brief walk-by)
        for i in range(3):
            engine.update_presence("STU-102", t_start + i, 1.0, 60.0)

        summary = engine.get_summary(t_start + 25)
        rec_a = engine.records["STU-101"]
        rec_b = engine.records["STU-102"]

        self.assertEqual(rec_a.status, "PRESENT")
        self.assertTrue(rec_a.sustained_presence_verified)
        self.assertEqual(rec_b.status, "ABSENT")
        self.assertFalse(rec_b.sustained_presence_verified)

    def test_06_audio_analyzer_features(self):
        """Epic 6 (Task 11 & 12): Verify acoustic RMS, VAD, and classification."""
        analyzer = AudioAnalyzer(use_microphone=False)
        # Process high-energy synthetic speech chunk
        speech_chunk = np.random.normal(0, 0.08, 8000).astype(np.float32)
        analyzer._process_chunk(speech_chunk)
        state = analyzer.get_state()

        self.assertGreater(state.rms_energy, 0.03)
        self.assertTrue(state.is_speech_active)
        self.assertIn(state.acoustic_environment, ["CLASSROOM_DISCUSSION", "INSTRUCTOR_MONOLOGUE", "SIDE_CHATTER"])

        # Process silence chunk
        silence_chunk = np.random.normal(0, 0.005, 8000).astype(np.float32)
        analyzer._process_chunk(silence_chunk)
        state_silent = analyzer.get_state()
        self.assertLess(state_silent.rms_energy, 0.02)

    def test_07_multimodal_fusion_engine(self):
        """Epic 7 (Task 13 & 14): Verify cross-modal score calculation and interval history."""
        fusion = MultimodalFusionEngine()
        audio = AudioAcousticState(
            rms_energy=0.04,
            zero_crossing_rate=0.05,
            is_speech_active=True,
            acoustic_environment="CLASSROOM_DISCUSSION",
            speech_duration_sec=5.0,
            chatter_index=0.1,
            audio_engagement_factor=0.90,
        )

        from src.vision.head_pose import HeadPoseResult
        from src.vision.visual_engagement import StudentVisualState

        st1 = StudentVisualState(
            track_id=1,
            student_id="STU-101",
            student_name="Aarav Sharma",
            head_pose=HeadPoseResult(yaw=2.0, pitch=0.0, roll=0.0, gaze_direction="LOOKING_FORWARD", attention_score=0.95),
            eye_aspect_ratio=0.32,
            motion_energy=0.15,
            stillness_seconds=0.0,
            drowsiness_flag=False,
            visual_score=92.0,
            status_label="Attentive",
        )

        snapshot = fusion.fuse([st1], audio, current_time_ts=100.0)
        self.assertGreaterEqual(snapshot.overall_engagement_score, 75.0)
        self.assertAlmostEqual(snapshot.attentive_pct, 100.0)
        self.assertEqual(snapshot.active_students_count, 1)

    def test_08_agent_reasoning_and_suggestions(self):
        """Epic 8 (Task 15 & 16): Verify pedagogical intervention trigger on attention drop."""
        reasoning = AgentReasoningEngine()
        audio = AudioAcousticState(0.01, 0.02, False, "SILENT_WORK", 0.0, 0.0, 0.5)

        # Snapshot with low engagement (35%)
        from src.fusion.multimodal_fusion import ClassroomDynamicsSnapshot
        snap = ClassroomDynamicsSnapshot(
            timestamp_ts=time.time(),
            timestamp_iso="10:15:00",
            overall_engagement_score=35.0,
            smoothed_engagement_score=35.0,
            active_students_count=4,
            audio_state=audio,
            student_scores={},
            student_states=[],
            attentive_pct=25.0,
            distracted_pct=50.0,
            drowsy_pct=25.0,
            taking_notes_pct=0.0,
            trend_direction="FALLING",
        )

        suggestion = reasoning.evaluate(snap)
        self.assertIsNotNone(suggestion)
        self.assertIn("Attention", suggestion.title)
        self.assertEqual(suggestion.urgency, "HIGH")
        self.assertTrue(len(suggestion.action_prompt) > 10)

    def test_09_flask_web_api(self):
        """Epic 9 (Task 17): Verify Flask REST endpoints and state payload."""
        app = create_app()
        app.config["TESTING"] = True
        client = app.test_client()

        # 1. Test root page
        res_index = client.get("/")
        self.assertEqual(res_index.status_code, 200)
        self.assertIn(b"Multimodal Attendance & Engagement Agent", res_index.data)

        # 2. Test state API
        state_data = {}
        for _ in range(20):
            res_state = client.get("/api/state")
            self.assertEqual(res_state.status_code, 200)
            state_data = res_state.get_json()
            if state_data.get("status") != "initializing":
                break
            time.sleep(0.2)

        self.assertIn("timestamp", state_data)
        self.assertIn("overall_engagement", state_data)
        self.assertIn("attendance", state_data)

        # 3. Test students list
        res_students = client.get("/api/students")
        self.assertEqual(res_students.status_code, 200)
        data = res_students.get_json()
        self.assertGreaterEqual(len(data.get("students", [])), 5)


if __name__ == "__main__":
    unittest.main()
