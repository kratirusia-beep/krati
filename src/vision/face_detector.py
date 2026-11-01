"""
Offline Multi-Face Detection and Facial Feature Localization.
Epic 3: Video Stream & Face Detection / Tracking (Task 5 & 6)
Operates 100% offline using OpenCV Haar Cascades with adaptive histogram equalization.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2

from src.config import get_config


@dataclass
class DetectedFace:
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    face_crop: np.ndarray
    landmarks_2d: Optional[np.ndarray] = None  # 2D points: [nose, chin, left_eye, right_eye, left_mouth, right_mouth]
    eyes_detected: int = 0
    eye_aspect_ratio: float = 0.30


class FaceDetector:
    """Robust offline face detector using multi-stage OpenCV cascade classifiers."""

    def __init__(self):
        self.config = get_config()
        haar_dir = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(haar_dir + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(haar_dir + "haarcascade_eye.xml")
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load OpenCV haarcascade_frontalface_default.xml")

    def detect_faces(self, frame: np.ndarray) -> List[DetectedFace]:
        """Detects faces in a BGR frame and estimates 2D key facial landmark points."""
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for robustness in varying lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        min_w, min_h = self.config.get("vision.min_face_size", [40, 40])
        faces = self.face_cascade.detectMultiScale(
            enhanced_gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(min_w, min_h),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        detected_list: List[DetectedFace] = []
        for (x, y, w, h) in faces:
            # Boundary guards
            x_min = max(0, x)
            y_min = max(0, y)
            x_max = min(frame.shape[1], x + w)
            y_max = min(frame.shape[0], y + h)

            face_roi_bgr = frame[y_min:y_max, x_min:x_max]
            face_roi_gray = enhanced_gray[y_min:y_max, x_min:x_max]

            # Eye detection inside upper half of face ROI
            roi_h, roi_w = face_roi_gray.shape
            upper_half = face_roi_gray[0:int(roi_h * 0.6), :]
            eyes = self.eye_cascade.detectMultiScale(
                upper_half,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(int(roi_w * 0.15), int(roi_h * 0.15))
            )

            # Heuristic landmark reconstruction for 3D PnP head pose
            # [nose, chin, left_eye, right_eye, left_mouth, right_mouth]
            nose_pt = (x + w * 0.50, y + h * 0.55)
            chin_pt = (x + w * 0.50, y + h * 0.95)

            if len(eyes) >= 2:
                # Sort eyes by X coordinate (left to right)
                sorted_eyes = sorted(eyes, key=lambda e: e[0])
                left_eye_rect = sorted_eyes[0]
                right_eye_rect = sorted_eyes[-1]
                left_eye_pt = (x + left_eye_rect[0] + left_eye_rect[2]*0.5, y + left_eye_rect[1] + left_eye_rect[3]*0.5)
                right_eye_pt = (x + right_eye_rect[0] + right_eye_rect[2]*0.5, y + right_eye_rect[1] + right_eye_rect[3]*0.5)
                ear = 0.32  # Alert
            else:
                left_eye_pt = (x + w * 0.30, y + h * 0.35)
                right_eye_pt = (x + w * 0.70, y + h * 0.35)
                ear = 0.22 if len(eyes) == 0 else 0.28

            left_mouth_pt = (x + w * 0.32, y + h * 0.78)
            right_mouth_pt = (x + w * 0.68, y + h * 0.78)

            landmarks = np.array([
                nose_pt,
                chin_pt,
                left_eye_pt,
                right_eye_pt,
                left_mouth_pt,
                right_mouth_pt
            ], dtype=np.float32)

            detected_list.append(
                DetectedFace(
                    bbox=(x, y, w, h),
                    confidence=0.88,
                    face_crop=face_roi_bgr,
                    landmarks_2d=landmarks,
                    eyes_detected=len(eyes),
                    eye_aspect_ratio=ear
                )
            )

        return detected_list
