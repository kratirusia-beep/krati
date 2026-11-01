"""
Multi-target Face Tracking across consecutive video frames.
Epic 3: Video Stream & Face Tracking (Task 5 & 6)
Maintains stable tracklets, calculates motion vectors, and tracks presence continuity.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

from .face_detector import DetectedFace


@dataclass
class TrackedTarget:
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    centroid: Tuple[float, float]
    consecutive_hits: int = 1
    missing_frames: int = 0
    total_frames_seen: int = 1
    first_seen_timestamp: float = 0.0
    last_seen_timestamp: float = 0.0
    student_id: Optional[str] = None
    student_name: str = "Unknown"
    match_confidence: float = 0.0
    motion_energy: float = 0.0
    face_crop: Optional[np.ndarray] = None
    landmarks_2d: Optional[np.ndarray] = None
    eye_aspect_ratio: float = 0.30
    recent_centroids: List[Tuple[float, float]] = field(default_factory=list)


class FaceTracker:
    """Multi-target spatial IoU + Centroid tracker."""

    def __init__(self, max_missing_frames: int = 25, min_iou_dist: float = 0.25):
        self.max_missing = max_missing_frames
        self.min_iou = min_iou_dist
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedTarget] = {}

    @staticmethod
    def _calc_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        denom = float(boxAArea + boxBArea - interArea)
        return interArea / denom if denom > 0 else 0.0

    @staticmethod
    def _centroid(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        return (box[0] + box[2] * 0.5, box[1] + box[3] * 0.5)

    def update(self, detected_faces: List[DetectedFace], current_time: float) -> List[TrackedTarget]:
        """Matches detections to active tracks, updates states, and initiates new tracks."""
        # 1. Compute IoU cost matrix
        track_ids = list(self.tracks.keys())
        active_tracks = [self.tracks[tid] for tid in track_ids]

        matched_tracks = set()
        matched_detections = set()

        if active_tracks and detected_faces:
            # Greedy matching based on IoU and centroid distance
            scores = np.zeros((len(active_tracks), len(detected_faces)))
            for t_idx, track in enumerate(active_tracks):
                for d_idx, det in enumerate(detected_faces):
                    iou = self._calc_iou(track.bbox, det.bbox)
                    c_det = self._centroid(det.bbox)
                    dist = np.hypot(track.centroid[0] - c_det[0], track.centroid[1] - c_det[1])
                    # Combined proximity metric
                    scores[t_idx, d_idx] = iou - (dist / 1000.0)

            # Sort candidate pairs descending
            candidate_pairs = []
            for t_idx in range(len(active_tracks)):
                for d_idx in range(len(detected_faces)):
                    candidate_pairs.append((scores[t_idx, d_idx], t_idx, d_idx))
            candidate_pairs.sort(key=lambda x: x[0], reverse=True)

            for score, t_idx, d_idx in candidate_pairs:
                if t_idx in matched_tracks or d_idx in matched_detections:
                    continue
                if score > -0.5:  # Valid spatial association
                    matched_tracks.add(t_idx)
                    matched_detections.add(d_idx)
                    tid = track_ids[t_idx]
                    track = self.tracks[tid]
                    det = detected_faces[d_idx]

                    # Compute motion displacement
                    new_centroid = self._centroid(det.bbox)
                    disp = np.hypot(new_centroid[0] - track.centroid[0], new_centroid[1] - track.centroid[1])
                    norm_motion = min(disp / max(det.bbox[2], 1), 1.0)

                    # Update track state
                    track.bbox = det.bbox
                    track.centroid = new_centroid
                    track.consecutive_hits += 1
                    track.missing_frames = 0
                    track.total_frames_seen += 1
                    track.last_seen_timestamp = current_time
                    track.face_crop = det.face_crop
                    track.landmarks_2d = det.landmarks_2d
                    track.eye_aspect_ratio = det.eye_aspect_ratio
                    track.motion_energy = 0.7 * track.motion_energy + 0.3 * norm_motion
                    track.recent_centroids.append(new_centroid)
                    if len(track.recent_centroids) > 20:
                        track.recent_centroids.pop(0)

        # 2. Handle unmatched active tracks (missing frames)
        for t_idx, tid in enumerate(track_ids):
            if t_idx not in matched_tracks:
                track = self.tracks[tid]
                track.missing_frames += 1
                track.consecutive_hits = 0

        # Purge stale tracks
        stale_ids = [tid for tid, track in self.tracks.items() if track.missing_frames > self.max_missing]
        for tid in stale_ids:
            del self.tracks[tid]

        # 3. Create new tracks for unmatched detections
        for d_idx, det in enumerate(detected_faces):
            if d_idx not in matched_detections:
                c = self._centroid(det.bbox)
                new_track = TrackedTarget(
                    track_id=self.next_track_id,
                    bbox=det.bbox,
                    centroid=c,
                    consecutive_hits=1,
                    missing_frames=0,
                    total_frames_seen=1,
                    first_seen_timestamp=current_time,
                    last_seen_timestamp=current_time,
                    face_crop=det.face_crop,
                    landmarks_2d=det.landmarks_2d,
                    eye_aspect_ratio=det.eye_aspect_ratio,
                    recent_centroids=[c],
                )
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1

        # Return active tracks with at least 1 recent hit
        return [t for t in self.tracks.values() if t.missing_frames == 0]
