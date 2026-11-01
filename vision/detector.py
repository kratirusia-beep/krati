"""
=============================================================================
Multimodal Attendance & Engagement Agent - Face Detector
Epic 3: Sustained Presence & Computer Vision Attendance Pipeline
Task 4: Multi-Face Detection & Preprocessing Pipeline
=============================================================================
"""

import cv2
import numpy as np
from typing import List, Tuple
from config import (
    FACE_DETECTION_SCALE_FACTOR,
    FACE_DETECTION_MIN_NEIGHBORS,
    FACE_MIN_SIZE,
)
from core.logger import logger


class FaceDetector:
    """Robust, offline multi-face detector using OpenCV cascades with CLAHE lighting normalization."""

    def __init__(self):
        # Load frontal and profile cascades
        frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"

        self.frontal_cascade = cv2.CascadeClassifier(frontal_path)
        self.profile_cascade = cv2.CascadeClassifier(profile_path)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        if self.frontal_cascade.empty():
            logger.error("Failed to load frontal face cascade.")
        else:
            logger.info("FaceDetector initialized successfully with OpenCV cascades.")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Converts to grayscale and applies CLAHE for lighting invariance."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        # Apply Contrast Limited Adaptive Histogram Equalization
        return self.clahe.apply(gray)

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detects all faces in the frame.
        
        Returns:
            List of bounding boxes [(x, y, w, h), ...]
        """
        gray = self.preprocess(frame)
        faces = self.frontal_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_DETECTION_SCALE_FACTOR,
            minNeighbors=FACE_DETECTION_MIN_NEIGHBORS,
            minSize=FACE_MIN_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        detected_boxes = []
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                detected_boxes.append((int(x), int(y), int(w), int(h)))

        # If few or no faces found, check profile for students looking sideways
        if len(detected_boxes) == 0:
            profiles = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION_SCALE_FACTOR,
                minNeighbors=FACE_DETECTION_MIN_NEIGHBORS,
                minSize=FACE_MIN_SIZE,
            )
            for (x, y, w, h) in profiles:
                detected_boxes.append((int(x), int(y), int(w), int(h)))

        return detected_boxes
