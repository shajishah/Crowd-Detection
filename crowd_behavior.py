"""
Crowd Behavior Modeling and Analysis
Tracks dominant motion patterns and establishes normal behavior baselines
"""
import numpy as np
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import cv2

@dataclass
class CrowdFlowMetrics:
    """Metrics for crowd behavior"""
    dominant_direction: float  # Dominant flow angle in degrees
    average_speed: float  # Average movement speed
    flow_magnitude: float  # Strength of dominant flow (0-1)
    density: int  # Number of persons detected
    activity_level: float  # Overall activity level (0-1)


class CrowdBehaviorAnalyzer:
    """
    Analyzes collective crowd behavior and establishes baselines
    """
    
    def __init__(self, history_length: int = 60):
        """
        Args:
            history_length: Number of frames to maintain history for baseline calculation
        """
        self.history_length = history_length
        self.flow_history = deque(maxlen=history_length)  # Direction vectors
        self.speed_history = deque(maxlen=history_length)  # Movement speeds
        self.density_history = deque(maxlen=history_length)  # Crowd density
        
        # Baseline (normal behavior)
        self.baseline_direction = None
        self.baseline_speed = None
        self.baseline_density = None
        self.is_baseline_set = False
    
    def analyze_frame(self, tracked_persons, frame_shape: Tuple[int, int]) -> CrowdFlowMetrics:
        """
        Analyze crowd behavior in current frame
        
        Args:
            tracked_persons: List of TrackedPerson objects
            frame_shape: (height, width) of frame
            
        Returns:
            CrowdFlowMetrics with current crowd analysis
        """
        # Extract motion vectors
        motion_vectors = []
        speeds = []
        
        for person in tracked_persons:
            velocity = person.get_velocity()
            if velocity is not None:
                motion_vectors.append(velocity)
                speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
                speeds.append(speed)
        
        # Calculate metrics
        dominant_direction = self._calculate_dominant_direction(motion_vectors)
        average_speed = np.mean(speeds) if speeds else 0.0
        flow_magnitude = self._calculate_flow_magnitude(motion_vectors)
        density = len(tracked_persons)
        activity_level = min(1.0, average_speed / 20.0)  # Normalize to 0-1
        
        metrics = CrowdFlowMetrics(
            dominant_direction=dominant_direction,
            average_speed=average_speed,
            flow_magnitude=flow_magnitude,
            density=density,
            activity_level=activity_level
        )
        
        # Store in history
        self.flow_history.append(dominant_direction)
        self.speed_history.append(average_speed)
        self.density_history.append(density)
        
        # Update baseline if enough history
        if len(self.flow_history) >= self.history_length * 0.5 and not self.is_baseline_set:
            self._set_baseline()
        
        return metrics
    
    def _calculate_dominant_direction(self, vectors: List[Tuple[float, float]]) -> float:
        """
        Calculate dominant crowd flow direction using circular mean
        
        Returns: Angle in degrees (0-360)
        """
        if not vectors:
            return 0.0
        
        vectors = np.array(vectors)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0]) * 180 / np.pi
        angles = (angles + 360) % 360
        
        # Circular mean
        sin_sum = np.sum(np.sin(angles * np.pi / 180))
        cos_sum = np.sum(np.cos(angles * np.pi / 180))
        dominant = np.arctan2(sin_sum, cos_sum) * 180 / np.pi
        return float((dominant + 360) % 360)
    
    def _calculate_flow_magnitude(self, vectors: List[Tuple[float, float]]) -> float:
        """
        Calculate flow consistency (how uniform is crowd motion)
        
        Returns: Value between 0 (random) and 1 (highly organized)
        """
        if len(vectors) < 2:
            return 0.0
        
        vectors = np.array(vectors)
        norms = np.linalg.norm(vectors, axis=1)
        
        if np.sum(norms) == 0:
            return 0.0
        
        # Normalize vectors
        normalized = vectors / (norms[:, np.newaxis] + 1e-6)
        
        # Mean resultant length
        mean_vector = np.mean(normalized, axis=0)
        magnitude = np.linalg.norm(mean_vector)
        
        return float(np.clip(magnitude, 0, 1))
    
    def _set_baseline(self):
        """Set baseline for normal crowd behavior"""
        self.baseline_direction = np.mean(list(self.flow_history))
        self.baseline_speed = np.mean(list(self.speed_history))
        self.baseline_density = np.mean(list(self.density_history))
        self.is_baseline_set = True
    
    def get_baseline(self) -> Optional[Dict]:
        """Get current baseline metrics"""
        if not self.is_baseline_set:
            return None
        
        return {
            'direction': self.baseline_direction,
            'speed': self.baseline_speed,
            'density': self.baseline_density
        }
    
    def is_against_crowd_flow(self, person_velocity: Tuple[float, float], 
                              dominant_direction: float, 
                              threshold_angle: float = 90.0) -> bool:
        """
        Check if person is moving against dominant crowd flow
        
        Args:
            person_velocity: (vx, vy) velocity vector
            dominant_direction: Dominant flow angle in degrees
            threshold_angle: Angle threshold in degrees
            
        Returns:
            True if person moves against flow
        """
        person_angle = np.arctan2(person_velocity[1], person_velocity[0]) * 180 / np.pi
        person_angle = (person_angle + 360) % 360
        
        # Calculate angular difference
        angle_diff = abs(person_angle - dominant_direction)
        angle_diff = min(angle_diff, 360 - angle_diff)
        
        return angle_diff > threshold_angle
    
    def visualize_flow(self, frame: np.ndarray, metrics: CrowdFlowMetrics) -> np.ndarray:
        """
        Visualize crowd flow on frame
        
        Args:
            frame: Input frame
            metrics: CrowdFlowMetrics
            
        Returns:
            Frame with flow visualization
        """
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        
        # Draw flow direction arrow
        arrow_length = 50
        angle_rad = metrics.dominant_direction * np.pi / 180
        end_x = int(center[0] + arrow_length * np.cos(angle_rad))
        end_y = int(center[1] + arrow_length * np.sin(angle_rad))
        
        # Color based on flow magnitude (red=weak, green=strong)
        color_value = int(255 * metrics.flow_magnitude)
        color = (255 - color_value, color_value, 0)
        
        cv2.arrowedLine(frame, center, (end_x, end_y), color, 3, tipLength=0.3)
        
        # Add metrics text
        info_text = [
            f"Direction: {metrics.dominant_direction:.1f}°",
            f"Speed: {metrics.average_speed:.2f} px/frame",
            f"Flow: {metrics.flow_magnitude:.2f}",
            f"Crowd: {metrics.density} persons",
            f"Activity: {metrics.activity_level:.2f}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            y_offset += 25
        
        return frame
