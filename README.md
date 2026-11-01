# 🎓 Multimodal Attendance & Engagement Agent

> **AI-Powered, Privacy-Preserving Classroom Intelligence System**  
> An edge-AI solution integrating **Computer Vision**, **Audio Signal Processing**, and **Agent-Based Pedagogical Reasoning** to automatically track sustained attendance, analyze real-time learner engagement dynamics, and provide actionable teaching suggestions. Designed to run **100% offline** on local hardware.

---

## 🌟 Key Capabilities & Scenarios

### 📸 Scenario 1: Automated Sustained Presence Attendance Tracking
- **Multi-Target Face Detection & Tracking**: Tracks multiple learners concurrently using OpenCV Haar/YuNet cascade filters and spatial IoU + centroid tracking.
- **Sustained Presence Temporal Verification**: Eliminates single-frame false positives caused by individuals walking past the camera. Attendance is marked only after sustained presence exceeds temporal thresholds (e.g. 15s in-seat duration).
- **Verifiable Timestamped Logging**: Automatically records *First Seen*, *Last Seen*, *Active Duration (mins)*, *Average Engagement Score (%)*, and classifies status as **PRESENT**, **LATE**, **PARTIAL**, or **ABSENT**.
- **1-Click CSV Export**: Instant export to standardized CSV reports for academic institutional records.

### 🧠 Scenario 2: Multimodal Engagement & Session Dynamics Analysis
- **3D Head Pose & Gaze Orientation**: Computes Euler angles (Yaw, Pitch, Roll) using 3D perspective-n-point (`cv2.solvePnP`) to classify attention (*Looking at Board*, *Taking Notes/Reading*, *Turned Away/Distracted*).
- **Behavioral & Inactivity Cues**: Evaluates Eye Aspect Ratio (EAR) for drowsiness/fatigue slumps and tracks prolonged physical stillness (>45s).
- **Classroom Acoustic Signal Processing**: Non-blocking microphone ingestion extracting RMS Energy, Zero-Crossing Rate (ZCR), Voice Activity Detection (VAD), and acoustic environment classification (*Instructor Monologue*, *Classroom Discussion*, *Side-Chatter*, *Silent Work*).
- **Multimodal Signal Fusion**: Time-synchronized fusion algorithm combining visual attention, motion vitality, and acoustic environment into an instantaneous and smoothed **Classroom Engagement Index (0–100%)**.
- **Temporal Timeline Analytics**: Detects session pacing issues, attention dips, and interaction peaks across 5-second interval time series.

### 💡 Scenario 3: AI-Driven Pedagogical Decision Support
- **Rule-Based & Heuristic Reasoning Core**: Evaluates continuous classroom state patterns to provide real-time, context-aware teaching interventions.
- **Evidence-Based Pedagogical Categories**:
  - **Interaction Prompts**: Suggests launching a 2-minute Think-Pair-Share or quick concept poll when classroom attention drops.
  - **Pacing Alerts**: Flags extended teacher monologues (>10 mins) without student participation.
  - **Fatigue / Drowsiness Clusters**: Recommends a 60-second kinesthetic stretch or targeted Socratic check-in.
  - **Classroom Management**: Detects elevated side-chatter during lecture segments and suggests redirection phrases.
- **Explainable Decision Support**: Displays underlying multimodal triggers (e.g., *Engagement: 38%*, *Stillness: 52s*, *Acoustic: SIDE_CHATTER*) while serving strictly as an assistive tool for educators.

---

## 🔒 100% Offline & Privacy-Preserving Architecture

- **Zero Cloud Transmission**: All inference, embedding extraction, and reasoning run entirely on the local CPU/edge device without external API calls or internet dependencies.
- **Biometric SHA-256 Hashing**: Face embeddings are anonymized and hashed using SHA-256 in local storage.
- **Local Data Retention**: Attendance records and session logs are stored strictly within the local workspace directory.

---

## 📊 Project Stats & Epics Mapping

| Epic | Tasks | Description | Module |
|---|---|---|---|
| **Epic 1: Foundation & Privacy** | Task 1 & 2 | Central YAML config manager, privacy hashing, and storage architecture | `src/config.py` |
| **Epic 2: Student Enrollment** | Task 3 & 4 | Offline identity registry, 128-dim feature extraction, and cosine matching | `src/enrollment/` |
| **Epic 3: Video Stream & Tracking** | Task 5 & 6 | Video feed ingestion (webcam/simulator) and multi-target spatial tracking | `src/vision/` |
| **Epic 4: Attendance Engine** | Task 7 & 8 | Sustained presence temporal filter, status classification, CSV export | `src/attendance/` |
| **Epic 5: Visual Engagement** | Task 9 & 10 | 3D head pose estimation, gaze classification, EAR, and stillness monitoring | `src/vision/` |
| **Epic 6: Audio Signal Processing** | Task 11 & 12 | RMS energy, zero-crossing rate, VAD, and acoustic context classification | `src/audio/` |
| **Epic 7: Multimodal Signal Fusion** | Task 13 & 14 | Cross-modal fusion algorithm, interval metrics, and dynamics timeline | `src/fusion/` |
| **Epic 8: AI Agent Reasoning** | Task 15 & 16 | Temporal rule evaluator, pedagogical catalog, and explainable suggestions | `src/agent/` |
| **Epic 9: Unified Web Dashboard** | Task 17 | Glassmorphic real-time UI, video stream HUD, Chart.js timeline, enrollment modal | `src/web/` |

---

## 🛠️ System Requirements

### Hardware Requirements
- **Processor**: Intel Core i5 (8th Gen or above) / AMD Ryzen 5 or equivalent
- **RAM**: Minimum 8 GB (Recommended: 16 GB for multitasking)
- **Storage**: 256 GB SSD (or 500 GB HDD minimum)
- **Camera & Microphone**: Standard integrated webcam and microphone (optional; includes built-in interactive Classroom Simulator)

### Software Requirements
- **Operating System**: Windows 10/11, macOS (Monterey+), or Linux (Ubuntu 20.04+)
- **Python**: Version 3.8 or above (Tested on Python 3.13)
- **Browser**: Modern Chrome, Firefox, or Edge

---

## 🚀 Quickstart Guide

### 1. Installation
Clone or navigate to the project directory and install requirements:
```bash
pip install -r requirements.txt
```

### 2. Launching the Web Intelligence Dashboard
Start the real-time Glassmorphic Educator Control Center:
```bash
python scripts/run_agent.py --mode web --port 5000
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

- **Classroom Simulator Mode**: Enabled by default for immediate out-of-the-box evaluation without needing a physical multi-student classroom.
- **Hardware Webcam Mode**: Click **"Switch to Live Camera"** in the top navigation bar to switch to your connected USB/integrated webcam in real time.

### 3. Running Headless / CLI Monitor Mode
To run the agent directly in terminal for headless deployments:
```bash
python scripts/run_agent.py --mode cli
```

### 4. Enrolling New Students
- **Via Web Dashboard**: Click **"+ Enroll Student"** in the top navigation bar, enter ID, Name, Roll No, Department, and optionally upload a photo.
- **Via CLI**:
```bash
python scripts/enroll_student.py --id "STU-107" --name "Diya Sen" --roll "CS2107" --dept "Computer Science"
```

### 5. Generating Sample Attendance Logs
To generate test datasets and export sample CSV attendance logs:
```bash
python scripts/generate_sample_data.py
```
Outputs verifiable CSV logs directly into `data/attendance_logs/`.

---

## 🧪 Automated Test Suite

Run the comprehensive unit and integration test suite covering all 9 Epics:
```bash
python tests/test_system.py
```
*Result: 9/9 tests pass in ~1.0s.*

---

## 📂 Project Architecture

```
krati/
├── README.md                           # Comprehensive documentation
├── requirements.txt                    # Project dependencies
├── config/
│   └── settings.yaml                   # Central threshold & system configuration
├── data/
│   ├── students/registry.json          # Offline student identity registry
│   └── attendance_logs/                # Exported CSV attendance reports
├── src/
│   ├── config.py                       # Configuration manager & singleton loader
│   ├── enrollment/
│   │   ├── __init__.py
│   │   └── student_registry.py         # Offline embeddings & cosine face matcher
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── face_detector.py            # Offline face & facial landmark detector
│   │   ├── face_tracker.py             # Spatial IoU & centroid multi-target tracker
│   │   ├── head_pose.py                # 3D solvePnP Euler angle & gaze estimator
│   │   └── visual_engagement.py        # Visual attention, EAR, and stillness analyzer
│   ├── audio/
│   │   ├── __init__.py
│   │   └── audio_analyzer.py           # RMS, VAD, and acoustic classifier
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── multimodal_fusion.py        # Time-synchronized cross-modal fusion engine
│   ├── attendance/
│   │   ├── __init__.py
│   │   └── attendance_engine.py        # Sustained presence tracker & CSV exporter
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── teaching_advisor.py         # Pedagogical heuristics & intervention catalog
│   │   └── reasoning_engine.py         # Continuous temporal pattern reasoning core
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── classroom_agent.py          # Master orchestrator & simulator generator
│   └── web/
│       ├── __init__.py
│       ├── app.py                      # Flask backend API & MJPEG streaming
│       ├── static/
│       │   ├── css/style.css           # Modern Dark Glassmorphic styling
│       │   └── js/app.js               # Reactive Chart.js & state controller
│       └── templates/
│           └── index.html              # Educator Control Center dashboard
├── scripts/
│   ├── run_agent.py                    # Master application launcher (Web / CLI)
│   ├── enroll_student.py               # CLI student enrollment tool
│   └── generate_sample_data.py         # Sample data & CSV generator
└── tests/
    └── test_system.py                  # Automated test suite (9 Epics / 17 Tasks)
```

---

## 🎓 Pedagogical Design Principles
The agent is designed as an assistive decision-support companion rather than an intrusive monitor. Suggestions respect cognitive load theory, natural attention cycles, and active learning dynamics:
1. **Passive listening attention curves** show predictable drops after 10–15 minutes.
2. **Kinesthetic resets** (60s movement) stimulate cerebral blood flow during afternoon fatigue slumps.
3. **Sustained presence filtering** ensures fair attendance logs by dismissing transient camera walk-bys.
4. **Offline processing** guarantees strict student biometric privacy compliance.
