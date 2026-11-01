"""
Classroom Agent Unified Multimodal Intelligence Pipeline.
Orchestrates Computer Vision, Audio Signal Analysis, Sustained Presence Attendance,
Multimodal Fusion, and Pedagogical Decision Support.
"""

import time
import threading
from typing import Dict, List, Optional, Any
import numpy as np
import cv2

from src.config import get_config
from src.enrollment.student_registry import StudentRegistry
from src.vision.face_detector import FaceDetector, DetectedFace
from src.vision.face_tracker import FaceTracker
from src.vision.visual_engagement import VisualEngagementAnalyzer
from src.audio.audio_analyzer import AudioAnalyzer
from src.attendance.attendance_engine import AttendanceEngine
from src.fusion.multimodal_fusion import MultimodalFusionEngine, ClassroomDynamicsSnapshot
from src.agent.reasoning_engine import AgentReasoningEngine


class ClassroomAgent:
    """Master agent coordinating all multimodal intelligence subsystems."""

    def __init__(
        self,
        video_source: Any = 0,
        enable_microphone: bool = True,
        use_simulator: bool = False,
    ):
        self.config = get_config()
        self.registry = StudentRegistry()
        self.registry.seed_sample_students()

        self.detector = FaceDetector()
        self.tracker = FaceTracker()
        self.visual_analyzer = VisualEngagementAnalyzer()
        self.audio_analyzer = AudioAnalyzer(use_microphone=enable_microphone)
        self.attendance_engine = AttendanceEngine(self.registry)
        self.fusion_engine = MultimodalFusionEngine()
        self.reasoning_engine = AgentReasoningEngine()

        self.video_source = video_source
        self.use_simulator = use_simulator
        self.cap: Optional[cv2.VideoCapture] = None

        self.is_running = False
        self.pipeline_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # Cached state for dashboard
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_snapshot: Optional[ClassroomDynamicsSnapshot] = None
        self.fps = 0.0

        # Simulation state
        self.sim_tick = 0

    def start(self):
        """Starts video capture, audio listening, and processing loop."""
        if self.is_running:
            return

        self.is_running = True
        self.audio_analyzer.start()

        if not self.use_simulator:
            try:
                self.cap = cv2.VideoCapture(self.video_source)
                if not self.cap.isOpened():
                    print(f"[ClassroomAgent] Could not open video source {self.video_source}. Enabling simulator.")
                    self.use_simulator = True
            except Exception as e:
                print(f"[ClassroomAgent] Video init error: {e}. Enabling simulator.")
                self.use_simulator = True

        self.pipeline_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.pipeline_thread.start()
        print(f"[ClassroomAgent] Pipeline started (Simulator={self.use_simulator}).")

    def _generate_simulated_frame(self) -> np.ndarray:
        """Generates realistic synthetic multi-student classroom frame for testing."""
        self.sim_tick += 1
        t = self.sim_tick * 0.1
        w, h = 960, 540
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Classroom background gradient
        for y in range(h):
            r = int(24 + (y / h) * 18)
            g = int(28 + (y / h) * 20)
            b = int(38 + (y / h) * 25)
            frame[y, :] = (b, g, r)

        # Draw classroom chalkboard / whiteboard in background
        cv2.rectangle(frame, (180, 20), (780, 160), (35, 45, 55), -1)
        cv2.rectangle(frame, (180, 20), (780, 160), (70, 85, 105), 3)
        cv2.putText(frame, "AI & Multimodal Intelligence - Lecture 04", (220, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 220, 240), 2)
        cv2.putText(frame, "Today: Behavioral Dynamics & Cognitive State Estimation", (220, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 170, 200), 1)

        # 4 Student desks and avatars
        student_configs = [
            ("STU-101", "Aarav Sharma", 150, 320, 0.90, "FORWARD"),
            ("STU-102", "Priya Patel", 370, 310, 0.85, "FORWARD"),
            ("STU-103", "Rohan Mehta", 590, 325, 0.50 if (self.sim_tick % 200 > 100) else 0.80, "DOWN" if (self.sim_tick % 200 > 100) else "FORWARD"),
            ("STU-104", "Ananya Rao", 810, 315, 0.35 if (self.sim_tick % 240 > 140) else 0.88, "AWAY" if (self.sim_tick % 240 > 140) else "FORWARD"),
        ]

        for s_id, name, cx, cy, att_factor, gaze_state in student_configs:
            # Subtle natural swaying motion
            sway_x = int(np.sin(t + cx) * 4)
            sway_y = int(np.cos(t * 0.7 + cy) * 2)
            face_cx = cx + sway_x
            face_cy = cy + sway_y
            face_size = 46

            # Shoulders / body
            cv2.ellipse(frame, (face_cx, face_cy + 85), (55, 40), 0, 0, 180, (50, 65, 80), -1)
            # Head / face
            cv2.circle(frame, (face_cx, face_cy), face_size, (190, 210, 225), -1)
            cv2.circle(frame, (face_cx, face_cy), face_size, (140, 160, 180), 2)

            # Hair
            cv2.ellipse(frame, (face_cx, face_cy - 18), (face_size, 25), 0, 180, 360, (40, 45, 55), -1)

            # Eyes based on gaze state
            eye_off_x = 0
            eye_off_y = 0
            if gaze_state == "DOWN":
                eye_off_y = 6
            elif gaze_state == "AWAY":
                eye_off_x = 10

            left_eye = (face_cx - 15 + eye_off_x, face_cy - 4 + eye_off_y)
            right_eye = (face_cx + 15 + eye_off_x, face_cy - 4 + eye_off_y)

            if att_factor < 0.40 and gaze_state == "DOWN":
                # Closed eyes (drowsy)
                cv2.line(frame, (left_eye[0] - 5, left_eye[1]), (left_eye[0] + 5, left_eye[1]), (50, 50, 60), 2)
                cv2.line(frame, (right_eye[0] - 5, right_eye[1]), (right_eye[0] + 5, right_eye[1]), (50, 50, 60), 2)
            else:
                cv2.circle(frame, left_eye, 4, (50, 50, 60), -1)
                cv2.circle(frame, right_eye, 4, (50, 50, 60), -1)

            # Mouth
            if att_factor > 0.70:
                cv2.ellipse(frame, (face_cx + eye_off_x, face_cy + 22), (10, 5), 0, 0, 180, (60, 70, 90), 2)
            else:
                cv2.line(frame, (face_cx - 8, face_cy + 22), (face_cx + 8, face_cy + 22), (60, 70, 90), 2)

            # Desk
            cv2.rectangle(frame, (cx - 75, cy + 90), (cx + 75, cy + 130), (50, 60, 75), -1)
            cv2.rectangle(frame, (cx - 75, cy + 90), (cx + 75, cy + 130), (80, 95, 115), 2)
            cv2.rectangle(frame, (cx - 30, cy + 95), (cx + 30, cy + 115), (100, 120, 140), -1)  # Notebook/laptop

        return frame

    def _run_loop(self):
        last_time = time.time()
        fps_counter = 0
        fps_start = time.time()

        while self.is_running:
            current_time = time.time()
            dt = max(0.001, current_time - last_time)
            last_time = current_time

            # 1. Grab Frame
            if self.use_simulator:
                raw_frame = self._generate_simulated_frame()
                time.sleep(0.033)  # ~30 FPS
            else:
                ret, raw_frame = self.cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

            # 2. Face Detection & Feature Extraction
            detected_faces = self.detector.detect_faces(raw_frame)

            # 3. Multi-target Tracking
            active_tracks = self.tracker.update(detected_faces, current_time)

            # 4. Identity Matching & Sustained Presence Registration
            for track in active_tracks:
                if track.student_id is None and track.face_crop is not None:
                    matched_id, conf, name = self.registry.match_face(track.face_crop, threshold=0.55)
                    if matched_id:
                        track.student_id = matched_id
                        track.student_name = name
                        track.match_confidence = conf
                    else:
                        # Fallback for simulator/unregistered targets
                        if self.use_simulator:
                            sim_map = {1: "STU-101", 2: "STU-102", 3: "STU-103", 4: "STU-104"}
                            if track.track_id in sim_map:
                                sid = sim_map[track.track_id]
                                p = self.registry.get(sid)
                                track.student_id = sid
                                track.student_name = p.name if p else f"Student {sid}"
                        else:
                            track.student_name = f"Guest #{track.track_id}"

            # 5. Visual Engagement Analysis
            visual_states = self.visual_analyzer.analyze(
                active_tracks, raw_frame.shape, current_time
            )

            # 6. Sustained Presence Attendance Update
            for v_state in visual_states:
                if v_state.student_id:
                    self.attendance_engine.update_presence(
                        student_id=v_state.student_id,
                        current_time_ts=current_time,
                        dt_seconds=dt,
                        current_engagement=v_state.visual_score,
                    )

            # 7. Audio State Ingestion
            audio_state = self.audio_analyzer.get_state()

            # 8. Multimodal Fusion
            snapshot = self.fusion_engine.fuse(visual_states, audio_state, current_time)

            # 9. AI Agent Pedagogical Reasoning
            self.reasoning_engine.evaluate(snapshot)

            # 10. Frame Annotation
            annotated_frame = self._render_annotations(raw_frame, visual_states, snapshot)

            # Update FPS
            fps_counter += 1
            if (current_time - fps_start) >= 1.0:
                self.fps = round(fps_counter / (current_time - fps_start), 1)
                fps_counter = 0
                fps_start = current_time

            with self.lock:
                self.latest_frame = raw_frame
                self.latest_annotated_frame = annotated_frame
                self.latest_snapshot = snapshot

    def _render_annotations(
        self,
        frame: np.ndarray,
        visual_states: List[Any],
        snapshot: ClassroomDynamicsSnapshot,
    ) -> np.ndarray:
        """Renders modern, aesthetic HUD overlays, bounding boxes, and gaze vectors."""
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # 1. Header HUD Overlay Bar
        cv2.rectangle(canvas, (0, 0), (w, 42), (18, 22, 30), -1)
        cv2.line(canvas, (0, 42), (w, 42), (40, 50, 70), 1)

        # Classroom Score Badge
        score_val = snapshot.smoothed_engagement_score
        score_col = (50, 205, 50) if score_val >= 70 else ((0, 165, 255) if score_val >= 50 else (60, 60, 230))
        cv2.putText(canvas, "CLASSROOM ENGAGEMENT:", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 190, 205), 1)
        cv2.putText(canvas, f"{score_val}%", (245, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, score_col, 2)

        # Acoustic Env Badge
        env_str = snapshot.audio_state.acoustic_environment.replace("_", " ")
        cv2.putText(canvas, f"ACOUSTIC: {env_str}", (340, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (140, 210, 255), 1)

        # Students count & FPS
        cv2.putText(canvas, f"ACTIVE: {snapshot.active_students_count}", (w - 220, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 220, 230), 1)
        cv2.putText(canvas, f"{self.fps} FPS", (w - 85, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (120, 220, 140), 1)

        # 2. Draw per-student tracked bounding boxes and state tags
        for st in visual_states:
            # Find bounding box from tracker
            track = self.tracker.tracks.get(st.track_id)
            if not track:
                continue

            x, y, bw, bh = track.bbox

            # Color coding based on status
            if st.drowsiness_flag:
                box_color = (40, 40, 230)      # Red
            elif st.status_label == "Attentive":
                box_color = (50, 205, 50)     # Green
            elif st.status_label == "Taking Notes":
                box_color = (240, 160, 40)    # Blue / Cyan
            else:
                box_color = (0, 165, 255)     # Amber / Orange

            # Elegant corner brackets instead of bulky solid rectangles
            corner_len = min(int(bw * 0.25), 18)
            t_thick = 2
            # Top-left
            cv2.line(canvas, (x, y), (x + corner_len, y), box_color, t_thick)
            cv2.line(canvas, (x, y), (x, y + corner_len), box_color, t_thick)
            # Top-right
            cv2.line(canvas, (x + bw, y), (x + bw - corner_len, y), box_color, t_thick)
            cv2.line(canvas, (x + bw, y), (x + bw, y + corner_len), box_color, t_thick)
            # Bottom-left
            cv2.line(canvas, (x, y + bh), (x + corner_len, y + bh), box_color, t_thick)
            cv2.line(canvas, (x, y + bh), (x, y + bh - corner_len), box_color, t_thick)
            # Bottom-right
            cv2.line(canvas, (x + bw, y + bh), (x + bw - corner_len, y + bh), box_color, t_thick)
            cv2.line(canvas, (x + bw, y + bh), (x + bw, y + bh - corner_len), box_color, t_thick)

            # Name & Engagement Badge Tag
            label_text = f"{st.student_name} | {st.visual_score}%"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tag_y = max(45, y - 6)
            cv2.rectangle(canvas, (x, tag_y - th - 6), (x + tw + 10, tag_y + 2), (18, 22, 30), -1)
            cv2.rectangle(canvas, (x, tag_y - th - 6), (x + tw + 10, tag_y + 2), box_color, 1)
            cv2.putText(canvas, label_text, (x + 5, tag_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 245, 250), 1)

            # Sub-badge with Head Pose & Gaze status
            sub_text = f"{st.status_label} (Y:{int(st.head_pose.yaw)} P:{int(st.head_pose.pitch)})"
            cv2.putText(canvas, sub_text, (x, y + bh + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, box_color, 1)

        return canvas

    def get_dashboard_payload(self) -> Dict[str, Any]:
        """Returns full JSON state for live web dashboard."""
        with self.lock:
            snap = self.latest_snapshot
            if snap is None:
                return {"status": "initializing"}

            att_summary = self.attendance_engine.get_summary()
            suggestions = self.reasoning_engine.get_latest_suggestions()

            return {
                "timestamp": snap.timestamp_iso,
                "overall_engagement": snap.smoothed_engagement_score,
                "raw_engagement": snap.overall_engagement_score,
                "active_students": snap.active_students_count,
                "attentive_pct": snap.attentive_pct,
                "notes_pct": snap.taking_notes_pct,
                "distracted_pct": snap.distracted_pct,
                "drowsy_pct": snap.drowsy_pct,
                "trend": snap.trend_direction,
                "audio": {
                    "rms": snap.audio_state.rms_energy,
                    "vad_active": snap.audio_state.is_speech_active,
                    "environment": snap.audio_state.acoustic_environment,
                    "monologue_sec": snap.audio_state.speech_duration_sec,
                    "chatter_index": snap.audio_state.chatter_index,
                    "audio_factor": snap.audio_state.audio_engagement_factor,
                },
                "attendance": {
                    "session_id": att_summary.session_id,
                    "session_start": att_summary.session_start_iso,
                    "enrolled_total": att_summary.total_enrolled,
                    "present_count": att_summary.present_count,
                    "late_count": att_summary.late_count,
                    "partial_count": att_summary.partial_count,
                    "absent_count": att_summary.absent_count,
                    "attendance_rate": att_summary.overall_attendance_rate,
                    "records": [r.to_dict() for r in att_summary.records],
                },
                "suggestions": suggestions,
                "timeline": [
                    {
                        "index": m.interval_index,
                        "time": m.timestamp_iso,
                        "score": m.aggregate_engagement_score,
                        "attentive": m.attentive_pct,
                        "distracted": m.distracted_pct,
                        "drowsy": m.drowsy_pct,
                        "env": m.acoustic_environment,
                        "drop": m.is_attention_drop,
                        "peak": m.is_interaction_peak,
                    }
                    for m in snap.interval_history[-30:]
                ],
                "students": [
                    {
                        "track_id": s.track_id,
                        "student_id": s.student_id or "UNREGISTERED",
                        "name": s.student_name,
                        "score": s.visual_score,
                        "status": s.status_label,
                        "yaw": s.head_pose.yaw,
                        "pitch": s.head_pose.pitch,
                        "stillness_sec": s.stillness_seconds,
                        "ear": s.eye_aspect_ratio,
                    }
                    for s in snap.student_states
                ],
                "fps": self.fps,
            }

    def get_annotated_jpeg(self) -> Optional[bytes]:
        """Encodes latest annotated frame to JPEG for live MJPEG HTTP streaming."""
        with self.lock:
            if self.latest_annotated_frame is None:
                return None
            ret, jpeg = cv2.imencode(".jpg", self.latest_annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                return jpeg.tobytes()
            return None

    def stop(self):
        self.is_running = False
        if self.audio_analyzer:
            self.audio_analyzer.stop()
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        print("[ClassroomAgent] Stopped.")
