"""
=============================================================================
Multimodal Attendance & Engagement Agent - Configuration Management
Epic 1: Project Architecture & Environment Initialization
Task 1: System Configuration, Constants & Parameter Tuning
=============================================================================
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "classroom.db"
ENROLLMENTS_DIR = DATA_DIR / "enrollments"
EXPORTS_DIR = DATA_DIR / "exports"
MODELS_DIR = DATA_DIR / "models"

# Ensure directories exist
for folder in [DATA_DIR, ENROLLMENTS_DIR, EXPORTS_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Application Settings
APP_NAME = "Multimodal Attendance & Engagement Agent"
VERSION = "1.0.0"
DEBUG = False
OFFLINE_MODE = True
PRIVACY_PRESERVE_RAW_FRAMES = False  # If True, raw video is never stored to disk

# Video Processing Parameters
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 15
FACE_DETECTION_SCALE_FACTOR = 1.15
FACE_DETECTION_MIN_NEIGHBORS = 4
FACE_MIN_SIZE = (45, 45)

# Sustained Presence & Attendance Parameters (Scenario 1)
# Attendance is marked only if a student is detected for sustained duration
SUSTAINED_PRESENCE_SECONDS_DEMO = 8      # Fast demonstration mode
SUSTAINED_PRESENCE_SECONDS_PROD = 45     # Production classroom mode
SUSTAINED_PRESENCE_THRESHOLD = SUSTAINED_PRESENCE_SECONDS_DEMO
MAX_DISAPPEARED_FRAMES = 45             # Tolerance for temporary occlusion (~3s at 15 FPS)
FACE_RECOGNITION_SIMILARITY_THRESHOLD = 0.62  # Cosine similarity threshold for feature embeddings

# Visual Engagement Indicators (Scenario 2)
HEAD_POSE_YAW_THRESHOLD = 28.0           # Degrees away from center (left/right)
HEAD_POSE_PITCH_THRESHOLD = 22.0         # Degrees away from center (up/down)
MOTION_INACTIVITY_THRESHOLD = 0.015      # Motion delta below which student is stationary
MOTION_RESTLESSNESS_THRESHOLD = 0.28     # Motion delta above which student is fidgeting
DROWSINESS_STILLNESS_SECONDS = 12.0      # Duration of stillness + downward tilt -> Drowsiness warning

# Audio Signal Processing (Scenario 2)
AUDIO_SAMPLE_RATE = 16000
AUDIO_BLOCK_SIZE = 1024
AUDIO_VAD_ENERGY_THRESHOLD = 0.018       # RMS energy threshold for voice detection
AUDIO_SILENCE_DECAY = 0.95
AUDIO_HIGH_NOISE_THRESHOLD = 0.12        # Classroom acoustic disruption threshold

# Multimodal Fusion Weights (Total = 1.0)
WEIGHT_HEAD_POSE = 0.40                  # Visual orientation toward screen/instructor
WEIGHT_MOVEMENT = 0.20                   # Micro-activity vs inactivity/slump
WEIGHT_EYE_ALERTNESS = 0.15              # Openness and gaze stability
WEIGHT_AUDIO_INTERACTION = 0.25          # Verbal participation and acoustic responsiveness

# Temporal Aggregation & Analytics
INTERVAL_WINDOW_SECONDS = 15             # Aggregation time window
ATTENTION_DROP_DELTA_THRESHOLD = 22.0    # Drop in engagement percentage triggering alert
LOW_ENGAGEMENT_ALERT_LEVEL = 45.0        # Engagement percentage below which warning fires
HIGH_ENGAGEMENT_LEVEL = 75.0             # Above this is active engagement

# AI Teaching Suggestion Parameters (Scenario 3)
SUGGESTION_COOLDOWN_SECONDS = 30         # Minimum seconds before firing same category suggestion
MAX_RECENT_SUGGESTIONS = 20

# Web Dashboard Server
WEB_HOST = "127.0.0.1"
WEB_PORT = 5000
