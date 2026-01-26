"""
Behavioral Outlier Detection
Identifies individuals with anomalous movement patterns
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum

class AnomalyType(Enum):
    """Types of behavioral anomalies"""
    AGAINST_FLOW = "Against Crowd Flow"
    ERRATIC_MOVEMENT = "Erratic/Abrupt Movement"
    STATIONARY_IN_FLOW = "Stationary in High-Flow Zone"
    RUNNING = "Running"
    RAISED_ARMS = "Unusual Gesture"
    CROUCHING = "Crouching"
    HIGH_CROWD_DENSITY = "High Crowd Density"
    CRITICAL_OVERCROWDING = "Critical Overcrowding"
    PANIC_FLOW = "Panic Flow"
    CROWD_CONGESTION = "Crowd Congestion"
    UNKNOWN = "Unknown"


@dataclass
class BehavioralAnomaly:
    """Detected behavioral anomaly"""
    track_id: int
    anomaly_type: AnomalyType
    confidence: float  # 0-1
    description: str
    centroid: Tuple[float, float]
    bbox: Tuple[float, float, float, float]


class BehavioralOutlierDetector:
    """
    Detects individuals with anomalous behavior
    Compares individual motion patterns against crowd baseline
    """
    
    def __init__(self):
        self.movement_history = {}  # track_id -> list of movements
        self.stationary_counter = {}  # track_id -> frames stationary
        self.baseline_speed = None
        self.baseline_direction = None
    
    def detect_anomalies(self, 
                        tracked_persons,
                        crowd_metrics,
                        pose_data: Dict = None) -> List[BehavioralAnomaly]:
        """
        Detect behavioral anomalies in tracked persons
        
        Args:
            tracked_persons: List of TrackedPerson objects
            crowd_metrics: CrowdFlowMetrics object
            pose_data: Dictionary mapping track_id to PoseAnalysis
            
        Returns:
            List of detected BehavioralAnomaly objects
        """
        anomalies = []
        
        for person in tracked_persons:
            person_anomalies = self._analyze_person(
                person, 
                crowd_metrics,
                pose_data.get(person.track_id) if pose_data else None
            )
            anomalies.extend(person_anomalies)
        
        return anomalies
    
    def _analyze_person(self, person, crowd_metrics, pose_analysis) -> List[BehavioralAnomaly]:
        """Analyze single person for anomalies"""
        anomalies = []
        
        velocity = person.get_velocity()
        if velocity is None:
            return anomalies
        
        speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
        
        # 1. Check if moving against crowd flow
        if crowd_metrics.flow_magnitude > 0.3:  # Only if crowd has clear flow
            if self._is_against_flow(person, crowd_metrics, velocity):
                anomalies.append(BehavioralAnomaly(
                    track_id=person.track_id,
                    anomaly_type=AnomalyType.AGAINST_FLOW,
                    confidence=0.7,
                    description="Moving against dominant crowd flow",
                    centroid=person.centroid,
                    bbox=person.bbox
                ))
        
        # 2. Check for erratic movement
        erratic_conf = self._detect_erratic_movement(person)
        if erratic_conf > 0.5:
            anomalies.append(BehavioralAnomaly(
                track_id=person.track_id,
                anomaly_type=AnomalyType.ERRATIC_MOVEMENT,
                confidence=erratic_conf,
                description="Erratic or abrupt movement patterns",
                centroid=person.centroid,
                bbox=person.bbox
            ))
        
        # 3. Check for stationary in high-flow zone
        if speed < 1.0 and crowd_metrics.activity_level > 0.5:
            self.stationary_counter[person.track_id] = self.stationary_counter.get(person.track_id, 0) + 1
            
            if self.stationary_counter[person.track_id] > 15:  # Stationary for 15+ frames
                anomalies.append(BehavioralAnomaly(
                    track_id=person.track_id,
                    anomaly_type=AnomalyType.STATIONARY_IN_FLOW,
                    confidence=min(0.9, 0.5 + (crowd_metrics.activity_level * 0.4)),
                    description="Stationary while crowd is moving",
                    centroid=person.centroid,
                    bbox=person.bbox
                ))
        else:
            self.stationary_counter[person.track_id] = 0
        
        # 4. Pose-based anomalies
        if pose_analysis:
            if pose_analysis.is_running:
                anomalies.append(BehavioralAnomaly(
                    track_id=person.track_id,
                    anomaly_type=AnomalyType.RUNNING,
                    confidence=0.8,
                    description="Person is running",
                    centroid=person.centroid,
                    bbox=person.bbox
                ))
            
            if pose_analysis.is_raised_arms and crowd_metrics.activity_level < 0.3:
                anomalies.append(BehavioralAnomaly(
                    track_id=person.track_id,
                    anomaly_type=AnomalyType.RAISED_ARMS,
                    confidence=pose_analysis.gesture_confidence,
                    description="Raised arms in calm environment",
                    centroid=person.centroid,
                    bbox=person.bbox
                ))
            
            if pose_analysis.posture_type == 'crouching' and crowd_metrics.activity_level < 0.2:
                anomalies.append(BehavioralAnomaly(
                    track_id=person.track_id,
                    anomaly_type=AnomalyType.CROUCHING,
                    confidence=0.7,
                    description="Crouching in calm environment",
                    centroid=person.centroid,
                    bbox=person.bbox
                ))
        
        return anomalies
    
    def _is_against_flow(self, person, crowd_metrics, velocity) -> bool:
        """Check if person moves against crowd flow"""
        person_angle = np.arctan2(velocity[1], velocity[0]) * 180 / np.pi
        person_angle = (person_angle + 360) % 360
        
        dominant = crowd_metrics.dominant_direction
        angle_diff = abs(person_angle - dominant)
        angle_diff = min(angle_diff, 360 - angle_diff)
        
        # Against flow if angle difference > 90 degrees
        return angle_diff > 90
    
    def _detect_erratic_movement(self, person) -> float:
        """
        Detect erratic movement by analyzing trajectory changes
        
        Returns: Confidence score 0-1
        """
        trajectory = person.get_trajectory_array()
        
        if len(trajectory) < 5:
            return 0.0
        
        # Calculate direction changes
        directions = []
        for i in range(1, len(trajectory)):
            prev = trajectory[i-1]
            curr = trajectory[i]
            angle = np.arctan2(curr[1] - prev[1], curr[0] - prev[0])
            directions.append(angle)
        
        if len(directions) < 2:
            return 0.0
        
        # Calculate angular changes
        directions = np.array(directions)
        angular_changes = np.abs(np.diff(directions))
        
        # Normalize to 0-π range
        angular_changes = np.minimum(angular_changes, np.pi - angular_changes)
        
        # High frequent changes = erratic
        mean_change = np.mean(angular_changes)
        std_change = np.std(angular_changes)
        
        # Erratic if high mean change or high variance
        erratic_score = min(1.0, (mean_change / np.pi) + (std_change / np.pi) * 0.5)
        
        return float(erratic_score)
    
    def cleanup_person(self, track_id: int):
        """Clean up tracking data for removed person"""
        self.movement_history.pop(track_id, None)
        self.stationary_counter.pop(track_id, None)
