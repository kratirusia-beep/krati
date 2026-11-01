"""
=============================================================================
Multimodal Attendance & Engagement Agent - Face Feature Encoder
Epic 2: Student Enrollment & Biometric Identity Management
Task 3: Offline Face Feature Extraction & Privacy-Preserving Cosine Matcher
=============================================================================
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from config import FACE_RECOGNITION_SIMILARITY_THRESHOLD
from core.models import Student
from core.logger import logger


class FaceFeatureEncoder:
    """Offline, deterministic 128-dimensional spatial-gradient biometric feature extractor.
    Preserves privacy by storing only normalized mathematical vector embeddings.
    """

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.target_size = (100, 100)

    def extract_embedding(self, face_bgr: np.ndarray) -> Optional[List[float]]:
        """Extracts a normalized 128-D feature vector from a cropped face image.
        
        Uses multi-region spatial gradient projections, intensity moments,
        and localized texture descriptors to form an identity-distinctive signature.
        """
        if face_bgr is None or face_bgr.size == 0:
            return None

        # Resize to standard canonical face size
        resized = cv2.resize(face_bgr, self.target_size)
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized

        # Apply histogram equalization
        eq = cv2.equalizeHist(gray)

        # 1. Multi-cell spatial block histograms (4x4 grid = 16 blocks)
        h, w = eq.shape
        block_h, block_w = h // 4, w // 4
        spatial_features = []

        for i in range(4):
            for j in range(4):
                cell = eq[i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w]
                # Mean and std dev of cell
                mean_val = float(np.mean(cell)) / 255.0
                std_val = float(np.std(cell)) / 255.0

                # Sobel gradients in horizontal and vertical directions
                gx = cv2.Sobel(cell, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(cell, cv2.CV_32F, 0, 1, ksize=3)
                mag = np.sqrt(gx**2 + gy**2)
                mean_mag = float(np.mean(mag)) / 255.0
                mean_gx = float(np.mean(np.abs(gx))) / 255.0
                mean_gy = float(np.mean(np.abs(gy))) / 255.0

                spatial_features.extend([mean_val, std_val, mean_mag, mean_gx, mean_gy])

        # 16 blocks * 5 features = 80 features

        # 2. Key facial landmark region projections (Eye band, Nose band, Mouth band)
        # Eye region (top 20% to 50%)
        eye_band = eq[int(h * 0.2) : int(h * 0.5), :]
        eye_proj_h = np.mean(eye_band, axis=0)  # 100 values -> downsample to 16
        eye_feat = cv2.resize(eye_proj_h.reshape(1, -1), (16, 1)).flatten() / 255.0

        # Nose & Mouth region (bottom 50% to 85%)
        mouth_band = eq[int(h * 0.5) : int(h * 0.85), :]
        mouth_proj_h = np.mean(mouth_band, axis=0)
        mouth_feat = cv2.resize(mouth_proj_h.reshape(1, -1), (16, 1)).flatten() / 255.0

        # Central vertical symmetry projection
        vert_proj = np.mean(eq, axis=1)
        vert_feat = cv2.resize(vert_proj.reshape(1, -1), (16, 1)).flatten() / 255.0

        # Total: 80 + 16 + 16 + 16 = 128 features!
        raw_vector = np.concatenate([
            np.array(spatial_features, dtype=np.float32),
            eye_feat.astype(np.float32),
            mouth_feat.astype(np.float32),
            vert_feat.astype(np.float32),
        ])

        # L2 Normalization for Cosine Distance matching
        norm = np.linalg.norm(raw_vector)
        if norm > 1e-6:
            normalized_vector = raw_vector / norm
        else:
            normalized_vector = raw_vector

        return normalized_vector.tolist()

    @staticmethod
    def compute_similarity(emb1: List[float], emb2: List[float]) -> float:
        """Computes Cosine Similarity between two face embeddings (range 0.0 to 1.0)."""
        if not emb1 or not emb2 or len(emb1) != len(emb2):
            return 0.0
        v1 = np.array(emb1, dtype=np.float32)
        v2 = np.array(emb2, dtype=np.float32)
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        sim = float(dot / (norm1 * norm2))
        return max(0.0, min(1.0, sim))

    def identify_face(
        self,
        face_embedding: List[float],
        enrolled_students: List[Student],
        threshold: float = FACE_RECOGNITION_SIMILARITY_THRESHOLD,
    ) -> Tuple[Optional[Student], float]:
        """Matches a face embedding against enrolled students in the database.
        
        Returns:
            Tuple of (Matched Student or None, similarity score)
        """
        if not face_embedding or not enrolled_students:
            return None, 0.0

        best_student = None
        best_similarity = 0.0

        for student in enrolled_students:
            if not student.face_embedding:
                continue
            sim = self.compute_similarity(face_embedding, student.face_embedding)
            if sim > best_similarity:
                best_similarity = sim
                best_student = student

        if best_similarity >= threshold:
            return best_student, best_similarity
        return None, best_similarity
