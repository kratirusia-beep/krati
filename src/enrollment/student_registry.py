"""
Student Enrollment & Identity Registry.
Epic 2: Student Enrollment & Identity Management (Task 3 & 4)
Privacy-preserving offline biometric registration and cosine-similarity matching.
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

from src.config import get_config


class StudentProfile:
    def __init__(
        self,
        student_id: str,
        name: str,
        roll_no: str,
        department: str,
        embedding: Optional[List[float]] = None,
        embedding_hash: Optional[str] = None,
        enrolled_at: Optional[str] = None,
        is_anonymized: bool = False,
        photo_base64: Optional[str] = None,
    ):
        self.student_id = student_id
        self.name = name
        self.roll_no = roll_no
        self.department = department
        self.embedding = embedding or []
        self.embedding_hash = embedding_hash or self._compute_hash(self.embedding)
        self.enrolled_at = enrolled_at or datetime.now().isoformat()
        self.is_anonymized = is_anonymized
        self.photo_base64 = photo_base64

    @staticmethod
    def _compute_hash(embedding: List[float]) -> str:
        if not embedding:
            return ""
        emb_bytes = np.array(embedding, dtype=np.float32).tobytes()
        return hashlib.sha256(emb_bytes).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "student_id": self.student_id,
            "name": self.name,
            "roll_no": self.roll_no,
            "department": self.department,
            "embedding": self.embedding,
            "embedding_hash": self.embedding_hash,
            "enrolled_at": self.enrolled_at,
            "is_anonymized": self.is_anonymized,
            "photo_base64": self.photo_base64,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StudentProfile":
        return cls(
            student_id=data["student_id"],
            name=data["name"],
            roll_no=data.get("roll_no", ""),
            department=data.get("department", "General"),
            embedding=data.get("embedding", []),
            embedding_hash=data.get("embedding_hash"),
            enrolled_at=data.get("enrolled_at"),
            is_anonymized=data.get("is_anonymized", False),
            photo_base64=data.get("photo_base64"),
        )


class StudentRegistry:
    """Manages student profiles, computes offline embeddings, and matches faces."""

    def __init__(self, storage_path: Optional[str] = None):
        self.config = get_config()
        if storage_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.storage_path = str(base_dir / "data" / "students" / "registry.json")
        else:
            self.storage_path = storage_path

        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        self.students: Dict[str, StudentProfile] = {}
        self.load()

    @staticmethod
    def extract_embedding(face_crop: np.ndarray) -> np.ndarray:
        """
        Computes a deterministic, normalized 128-dimensional offline face feature embedding.
        Works entirely offline using multi-scale gradient and block-frequency descriptors.
        """
        if face_crop is None or face_crop.size == 0:
            return np.zeros(128, dtype=np.float32)

        # 1. Normalize size & grayscale
        if len(face_crop.shape) == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop.copy()

        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        equalized = cv2.equalizeHist(resized)

        # 2. Extract multi-block spatial gradient histograms (HOG-like)
        # 4x4 blocks -> 16 regions, each with 8 gradient orientation bins = 128 dimensions!
        sobel_x = cv2.Sobel(equalized, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(equalized, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(sobel_x, sobel_y, angleInDegrees=True)

        feature_vector = []
        block_h, block_w = 16, 16
        for r in range(4):
            for c in range(4):
                block_mag = magnitude[r*block_h:(r+1)*block_h, c*block_w:(c+1)*block_w]
                block_ang = angle[r*block_h:(r+1)*block_h, c*block_w:(c+1)*block_w]
                # 8 orientation bins
                hist, _ = np.histogram(block_ang, bins=8, range=(0, 360), weights=block_mag)
                feature_vector.extend(hist)

        emb = np.array(feature_vector, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb /= norm
        return emb

    def enroll(
        self,
        student_id: str,
        name: str,
        roll_no: str,
        department: str,
        face_img: np.ndarray,
        is_anonymized: bool = False,
        photo_base64: Optional[str] = None,
    ) -> StudentProfile:
        """Registers a new student profile and persists it to local storage."""
        embedding = self.extract_embedding(face_img).tolist()
        profile = StudentProfile(
            student_id=student_id,
            name=name,
            roll_no=roll_no,
            department=department,
            embedding=embedding,
            is_anonymized=is_anonymized,
            photo_base64=photo_base64,
        )
        self.students[student_id] = profile
        self.save()
        return profile

    def match_face(self, face_crop: np.ndarray, threshold: float = 0.65) -> Tuple[Optional[str], float, str]:
        """
        Finds the closest enrolled student matching the provided face crop using cosine similarity.
        Returns: (student_id, similarity_score, student_name)
        """
        if not self.students or face_crop is None or face_crop.size == 0:
            return None, 0.0, "Unknown"

        target_emb = self.extract_embedding(face_crop)
        best_id = None
        best_sim = -1.0
        best_name = "Unknown"

        for s_id, profile in self.students.items():
            if not profile.embedding:
                continue
            ref_emb = np.array(profile.embedding, dtype=np.float32)
            sim = float(np.dot(target_emb, ref_emb))
            if sim > best_sim:
                best_sim = sim
                best_id = s_id
                best_name = profile.name

        if best_sim >= threshold and best_id is not None:
            return best_id, best_sim, best_name
        return None, max(best_sim, 0.0), "Unknown"

    def list_all(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self.students.values()]

    def get(self, student_id: str) -> Optional[StudentProfile]:
        return self.students.get(student_id)

    def delete(self, student_id: str) -> bool:
        if student_id in self.students:
            del self.students[student_id]
            self.save()
            return True
        return False

    def save(self):
        data = {s_id: p.to_dict() for s_id, p in self.students.items()}
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self):
        if not os.path.exists(self.storage_path):
            self.students = {}
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.students = {s_id: StudentProfile.from_dict(item) for s_id, item in data.items()}
        except Exception as e:
            print(f"[StudentRegistry] Load failed: {e}. Initializing empty.")
            self.students = {}

    def seed_sample_students(self):
        """Pre-populates a realistic classroom roster for immediate offline demonstrations."""
        if len(self.students) >= 5:
            return

        samples = [
            ("STU-101", "Aarav Sharma", "CS2101", "Computer Science"),
            ("STU-102", "Priya Patel", "CS2102", "Computer Science"),
            ("STU-103", "Rohan Mehta", "AI2103", "Data Science & AI"),
            ("STU-104", "Ananya Rao", "AI2104", "Data Science & AI"),
            ("STU-105", "Vikram Singh", "EC2105", "Electronics"),
            ("STU-106", "Sneha Nair", "CS2106", "Computer Science"),
        ]

        # Generate synthetic distinct face representations for each sample student
        np.random.seed(42)
        for s_id, name, roll, dept in samples:
            dummy_face = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
            # Add facial synthetic gradients
            cv2.circle(dummy_face, (24, 24), 8, (220, 220, 220), -1)
            cv2.circle(dummy_face, (40, 24), 8, (220, 220, 220), -1)
            cv2.ellipse(dummy_face, (32, 45), (12, 6), 0, 0, 180, (180, 180, 180), 2)
            self.enroll(s_id, name, roll, dept, dummy_face)
        print(f"[StudentRegistry] Seeded {len(samples)} sample students into registry.")
