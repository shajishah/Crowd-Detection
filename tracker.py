"""
Multi-Object Tracker with persistent ID assignment and trajectory tracking
"""
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import cv2

@dataclass
class TrackedPerson:
    """Represents a tracked person with persistent ID"""
    track_id: int
    centroid: tuple  # (x, y)
    bbox: tuple  # (x1, y1, x2, y2)
    trajectory: deque = field(default_factory=lambda: deque(maxlen=30))  # Last 30 frames
    confidence: float = 1.0
    frames_since_seen: int = 0
    appearance_feature: Optional[np.ndarray] = None
    pose_keypoints: Optional[np.ndarray] = None
    
    def update(self, centroid, bbox, confidence=1.0):
        """Update person's current position"""
        self.centroid = centroid
        self.bbox = bbox
        self.confidence = confidence
        self.trajectory.append(centroid)
        self.frames_since_seen = 0
    
    def get_velocity(self) -> Optional[tuple]:
        """Calculate velocity vector from recent trajectory"""
        if len(self.trajectory) < 2:
            return None
        prev = self.trajectory[-2]
        curr = self.trajectory[-1]
        return (curr[0] - prev[0], curr[1] - prev[1])
    
    def get_trajectory_array(self) -> np.ndarray:
        """Return trajectory as numpy array"""
        return np.array(list(self.trajectory))


class MultiObjectTracker:
    """
    Simple centroid tracking with ID assignment
    Tracks persons across frames using centroid distance matching
    """
    
    def __init__(self, max_distance=50, max_frames_to_skip=5):
        """
        Args:
            max_distance: Maximum distance to match centroids across frames
            max_frames_to_skip: Number of frames before dropping a track
        """
        self.max_distance = max_distance
        self.max_frames_to_skip = max_frames_to_skip
        self.tracked_persons: Dict[int, TrackedPerson] = {}
        self.next_id = 0
    
    def distance(self, pt1: tuple, pt2: tuple) -> float:
        """Euclidean distance between two points"""
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
    
    def update(self, detections: List[tuple]) -> List[TrackedPerson]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of (x1, y1, x2, y2) bounding boxes
            
        Returns:
            List of TrackedPerson objects with updated tracks
        """
        # Calculate centroids from detections
        centroids = []
        for x1, y1, x2, y2 in detections:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            centroids.append((cx, cy))
        
        # Match detections to existing tracks
        if len(self.tracked_persons) == 0:
            # First frame or no active tracks
            for i, (centroid, bbox) in enumerate(zip(centroids, detections)):
                self.tracked_persons[self.next_id] = TrackedPerson(
                    track_id=self.next_id,
                    centroid=centroid,
                    bbox=bbox
                )
                self.next_id += 1
        else:
            # Match detections to tracks
            matched, unmatched_dets, unmatched_tracks = self._match_detections(
                centroids, list(self.tracked_persons.values())
            )
            
            # Update matched tracks
            for track_idx, det_idx in matched:
                track = list(self.tracked_persons.values())[track_idx]
                track.update(centroids[det_idx], detections[det_idx])
            
            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                self.tracked_persons[self.next_id] = TrackedPerson(
                    track_id=self.next_id,
                    centroid=centroids[det_idx],
                    bbox=detections[det_idx]
                )
                self.next_id += 1
            
            # Mark unmatched tracks as not seen
            for track_idx in unmatched_tracks:
                track = list(self.tracked_persons.values())[track_idx]
                track.frames_since_seen += 1
            
            # Remove tracks that haven't been seen in too long
            self.tracked_persons = {
                tid: track for tid, track in self.tracked_persons.items()
                if track.frames_since_seen <= self.max_frames_to_skip
            }
        
        return list(self.tracked_persons.values())
    
    def _match_detections(self, centroids, tracks):
        """
        Match detections to tracks using Hungarian algorithm (simplified greedy)
        
        Returns:
            (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if len(centroids) == 0 or len(tracks) == 0:
            return [], list(range(len(centroids))), list(range(len(tracks)))
        
        # Build distance matrix
        dist_matrix = np.zeros((len(tracks), len(centroids)))
        for i, track in enumerate(tracks):
            for j, centroid in enumerate(centroids):
                dist_matrix[i, j] = self.distance(track.centroid, centroid)
        
        # Greedy matching
        matched = []
        used_dets = set()
        used_tracks = set()
        
        # Sort by minimum distance
        distances = []
        for i in range(len(tracks)):
            for j in range(len(centroids)):
                distances.append((dist_matrix[i, j], i, j))
        distances.sort()
        
        for dist, track_idx, det_idx in distances:
            if dist <= self.max_distance and track_idx not in used_tracks and det_idx not in used_dets:
                matched.append((track_idx, det_idx))
                used_tracks.add(track_idx)
                used_dets.add(det_idx)
        
        unmatched_dets = [i for i in range(len(centroids)) if i not in used_dets]
        unmatched_tracks = [i for i in range(len(tracks)) if i not in used_tracks]
        
        return matched, unmatched_dets, unmatched_tracks
    
    def get_active_tracks(self) -> List[TrackedPerson]:
        """Get all currently active tracks"""
        return list(self.tracked_persons.values())
    
    def clear(self):
        """Reset tracker"""
        self.tracked_persons.clear()
        self.next_id = 0
