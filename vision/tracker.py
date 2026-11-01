"""
=============================================================================
Multimodal Attendance & Engagement Agent - Multi-Person Centroid & IoU Tracker
Epic 3: Sustained Presence & Computer Vision Attendance Pipeline
Task 4: Continuous Visual Tracking & Track State Management
=============================================================================
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from config import MAX_DISAPPEARED_FRAMES


class TrackedPerson:
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int]):
        self.track_id = track_id
        self.bbox = bbox  # (x, y, w, h)
        self.centroid = self._calculate_centroid(bbox)
        self.disappeared = 0
        self.student_id: Optional[str] = None
        self.student_name: str = f"Person_{track_id}"
        self.recognition_confidence: float = 0.0

        # Temporal presence metrics (Scenario 1)
        self.first_seen_time = time.time()
        self.last_seen_time = time.time()
        self.total_present_seconds = 0.0
        self.sustained_presence_seconds = 0.0
        self.is_marked_present = False

        # Visual engagement buffer (Scenario 2)
        self.prev_face_gray: Optional[np.ndarray] = None
        self.motion_delta = 0.0
        self.inactivity_seconds = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.visual_attention_score = 100.0
        self.engagement_score = 75.0
        self.status_label = "FOCUSED"

    @staticmethod
    def _calculate_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x, y, w, h = bbox
        return int(x + w / 2.0), int(y + h / 2.0)

    def update_bbox(self, bbox: Tuple[int, int, int, int], elapsed_step: float):
        self.bbox = bbox
        self.centroid = self._calculate_centroid(bbox)
        self.disappeared = 0
        now = time.time()
        self.last_seen_time = now
        self.total_present_seconds += elapsed_step
        self.sustained_presence_seconds += elapsed_step

    def mark_missing(self, elapsed_step: float):
        self.disappeared += 1
        # When missing, sustained presence gently decays or holds
        self.sustained_presence_seconds = max(0.0, self.sustained_presence_seconds - (elapsed_step * 0.5))


class MultiPersonTracker:
    """Robust centroid and IoU based multi-target tracking across classroom video frames."""

    def __init__(self, max_disappeared: int = MAX_DISAPPEARED_FRAMES):
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedPerson] = {}
        self.max_disappeared = max_disappeared
        self.last_update_time = time.time()

    @staticmethod
    def _compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def update(self, rects: List[Tuple[int, int, int, int]]) -> Dict[int, TrackedPerson]:
        """Updates tracks with newly detected bounding boxes.
        
        Returns:
            Dictionary of active {track_id: TrackedPerson}
        """
        now = time.time()
        elapsed_step = max(0.01, min(1.0, now - self.last_update_time))
        self.last_update_time = now

        # Case 1: No detections this frame
        if len(rects) == 0:
            tracks_to_delete = []
            for track_id, person in self.tracks.items():
                person.mark_missing(elapsed_step)
                if person.disappeared > self.max_disappeared:
                    tracks_to_delete.append(track_id)
            for track_id in tracks_to_delete:
                del self.tracks[track_id]
            return self.tracks

        # Case 2: No existing tracks, register all detections
        if len(self.tracks) == 0:
            for rect in rects:
                self.register(rect)
            return self.tracks

        # Case 3: Match existing tracks to detections using IoU + Centroid Distance
        track_ids = list(self.tracks.keys())
        existing_boxes = [self.tracks[tid].bbox for tid in track_ids]

        # Compute cost matrix (negative IoU + normalized distance)
        matched_detections = set()
        matched_tracks = set()

        # Step A: High overlap IoU matching
        for t_idx, tid in enumerate(track_ids):
            best_iou = 0.0
            best_d_idx = -1
            for d_idx, rect in enumerate(rects):
                if d_idx in matched_detections:
                    continue
                iou = self._compute_iou(existing_boxes[t_idx], rect)
                if iou > best_iou:
                    best_iou = iou
                    best_d_idx = d_idx

            if best_iou >= 0.25 and best_d_idx != -1:
                self.tracks[tid].update_bbox(rects[best_d_idx], elapsed_step)
                matched_detections.add(best_d_idx)
                matched_tracks.add(tid)

        # Step B: Distance matching for remaining
        unmatched_tracks = [tid for tid in track_ids if tid not in matched_tracks]
        unmatched_rects = [d_idx for d_idx in range(len(rects)) if d_idx not in matched_detections]

        for tid in unmatched_tracks:
            p = self.tracks[tid]
            best_dist = float("inf")
            best_d_idx = -1
            for d_idx in unmatched_rects:
                if d_idx in matched_detections:
                    continue
                rx, ry, rw, rh = rects[d_idx]
                dcx, dcy = int(rx + rw / 2.0), int(ry + rh / 2.0)
                dist = np.hypot(p.centroid[0] - dcx, p.centroid[1] - dcy)
                if dist < best_dist and dist < 120.0:  # reasonable motion threshold
                    best_dist = dist
                    best_d_idx = d_idx

            if best_d_idx != -1:
                self.tracks[tid].update_bbox(rects[best_d_idx], elapsed_step)
                matched_detections.add(best_d_idx)
                matched_tracks.add(tid)
            else:
                p.mark_missing(elapsed_step)

        # Step C: Register remaining new detections
        for d_idx in range(len(rects)):
            if d_idx not in matched_detections:
                self.register(rects[d_idx])

        # Step D: Prune stale tracks
        stale_tracks = [tid for tid, p in self.tracks.items() if p.disappeared > self.max_disappeared]
        for tid in stale_tracks:
            del self.tracks[tid]

        return self.tracks

    def register(self, bbox: Tuple[int, int, int, int]):
        person = TrackedPerson(self.next_track_id, bbox)
        self.tracks[self.next_track_id] = person
        self.next_track_id += 1
