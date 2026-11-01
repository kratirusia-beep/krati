"""
Flask Web Application & API Gateway for Multimodal Attendance & Engagement Agent.
Epic 9: Unified Interactive Dashboard & Control Center (Task 17)
"""

import os
import time
import base64
from pathlib import Path
from flask import Flask, render_template, Response, jsonify, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
from typing import Optional

from src.pipeline.classroom_agent import ClassroomAgent
from src.config import get_config

# Global agent singleton
GLOBAL_AGENT: Optional[ClassroomAgent] = None


def get_agent() -> ClassroomAgent:
    global GLOBAL_AGENT
    if GLOBAL_AGENT is None:
        # Default start in simulator mode for instant offline usability, can toggle to webcam
        GLOBAL_AGENT = ClassroomAgent(video_source=0, enable_microphone=True, use_simulator=True)
        GLOBAL_AGENT.start()
    return GLOBAL_AGENT


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parent / "templates"),
        static_folder=str(Path(__file__).resolve().parent / "static"),
    )
    CORS(app)

    @app.route("/")
    def index():
        return render_template("index.html")

    def gen_frames(agent: ClassroomAgent):
        while True:
            frame_bytes = agent.get_annotated_jpeg()
            if frame_bytes is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.04)

    @app.route("/video_feed")
    def video_feed():
        agent = get_agent()
        return Response(gen_frames(agent), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/state")
    def api_state():
        agent = get_agent()
        return jsonify(agent.get_dashboard_payload())

    @app.route("/api/mode", methods=["POST"])
    def api_mode():
        agent = get_agent()
        data = request.get_json() or {}
        use_sim = data.get("use_simulator", True)

        agent.stop()
        global GLOBAL_AGENT
        GLOBAL_AGENT = ClassroomAgent(video_source=0, enable_microphone=True, use_simulator=use_sim)
        GLOBAL_AGENT.start()
        return jsonify({"success": True, "mode": "Simulator" if use_sim else "Live Camera"})

    @app.route("/api/enroll", methods=["POST"])
    def api_enroll():
        agent = get_agent()
        data = request.get_json() or {}
        s_id = data.get("student_id", "").strip()
        name = data.get("name", "").strip()
        roll = data.get("roll_no", "").strip()
        dept = data.get("department", "Computer Science").strip()
        photo_b64 = data.get("photo_base64", "")

        if not s_id or not name:
            return jsonify({"success": False, "error": "Student ID and Name are required."}), 400

        # Decode face image if provided
        face_img = None
        if photo_b64 and "," in photo_b64:
            try:
                raw_bytes = base64.b64decode(photo_b64.split(",")[1])
                nparr = np.frombuffer(raw_bytes, np.uint8)
                face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"[Enroll API] Image decode error: {e}")

        if face_img is None:
            # Fallback synthetic profile image
            face_img = np.random.randint(60, 210, (64, 64, 3), dtype=np.uint8)
            cv2.circle(face_img, (32, 32), 20, (200, 200, 200), -1)

        profile = agent.registry.enroll(s_id, name, roll, dept, face_img, photo_base64=photo_b64)
        agent.attendance_engine.refresh_enrolled_students()

        return jsonify({"success": True, "student": profile.to_dict()})

    @app.route("/api/students")
    def api_students():
        agent = get_agent()
        return jsonify({"students": agent.registry.list_all()})

    @app.route("/api/export_attendance")
    def api_export_attendance():
        agent = get_agent()
        file_path = agent.attendance_engine.export_csv()
        if os.path.exists(file_path):
            return send_file(
                file_path,
                mimetype="text/csv",
                as_attachment=True,
                download_name=os.path.basename(file_path)
            )
        return jsonify({"error": "Failed to generate report"}), 500

    @app.route("/api/reset_session", methods=["POST"])
    def api_reset_session():
        agent = get_agent()
        agent.attendance_engine = AttendanceEngine(agent.registry)
        agent.fusion_engine = MultimodalFusionEngine()
        return jsonify({"success": True, "session_id": agent.attendance_engine.session_id})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
