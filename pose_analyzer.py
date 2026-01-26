"""
Pose Estimation and Movement Analysis using MediaPipe
Extracts skeletal keypoints and analyzes motion patterns
"""
import mediapipe as mp
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class PoseAnalysis:
    """Results from pose analysis"""
    keypoints: np.ndarray  # (33, 3) - x, y, confidence for each keypoint
    is_visible: bool  # Person is visible and pose detected
    posture_type: str  # 'standing', 'crouching', 'lying', 'unknown'
    movement_speed: float  # Pixels per frame
    movement_direction: float  # Angle in degrees (0-360)
    is_running: bool
    is_raised_arms: bool
    gesture_confidence: float


class PoseAnalyzer:
    """Analyzes human poses and movements"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=light, 1=full, 2=heavy
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Keypoint indices
        self.NOSE = 0
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_ANKLE = 27
        self.RIGHT_ANKLE = 28
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        
        # Pose history for motion analysis
        self.keypoint_history = {}  # track_id -> deque of keypoints
    
    def analyze_frame(self, frame: np.ndarray, offset: Tuple[int, int] = (0, 0)) -> Optional[PoseAnalysis]:
        """
        Analyze pose in a frame (or crop)
        
        Args:
            frame: RGB frame or crop from OpenCV
            offset: (x, y) offset of the crop in the original frame
            
        Returns:
            PoseAnalysis object or None if no pose detected
        """
        results = self.pose.process(frame)
        
        if not results.pose_landmarks:
            return None
        
        keypoints = np.zeros((33, 3))
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints[idx] = [landmark.x, landmark.y, landmark.z]
        
        # Denormalize to pixel coordinates (relative to crop)
        h, w = frame.shape[:2]
        keypoints[:, 0] *= w
        keypoints[:, 1] *= h
        
        # Add offset to convert to global coordinates
        keypoints[:, 0] += offset[0]
        keypoints[:, 1] += offset[1]
        
        # Analyze posture and movement
        posture = self._analyze_posture(keypoints)
        movement_speed = 0.0
        movement_direction = 0.0
        is_running = self._detect_running(keypoints)
        is_raised_arms = self._detect_raised_arms(keypoints)
        gesture_confidence = self._calculate_gesture_confidence(keypoints)
        
        return PoseAnalysis(
            keypoints=keypoints,
            is_visible=True,
            posture_type=posture,
            movement_speed=movement_speed,
            movement_direction=movement_direction,
            is_running=is_running,
            is_raised_arms=is_raised_arms,
            gesture_confidence=gesture_confidence
        )
    
    def update_trajectory(self, track_id: int, keypoints: np.ndarray):
        """Update movement trajectory for a tracked person"""
        if track_id not in self.keypoint_history:
            self.keypoint_history[track_id] = deque(maxlen=15)
        self.keypoint_history[track_id].append(keypoints.copy())
    
    def _analyze_posture(self, keypoints: np.ndarray) -> str:
        """
        Analyze body posture from keypoints
        
        Returns: 'standing', 'crouching', 'lying', 'unknown'
        """
        # Get key joint positions
        nose = keypoints[self.NOSE]
        left_hip = keypoints[self.LEFT_HIP]
        right_hip = keypoints[self.RIGHT_HIP]
        left_ankle = keypoints[self.LEFT_ANKLE]
        right_ankle = keypoints[self.RIGHT_ANKLE]
        
        hip_center = (left_hip + right_hip) / 2
        ankle_center = (left_ankle + right_ankle) / 2
        
        # Calculate vertical distances
        torso_length = np.linalg.norm(nose[:2] - hip_center[:2])
        leg_length = np.linalg.norm(hip_center[:2] - ankle_center[:2])
        
        if leg_length < torso_length * 0.3:
            return 'crouching'
        elif leg_length < torso_length * 0.8:
            return 'sitting'
        elif abs(nose[1] - ankle_center[1]) < 50:  # Close to ground
            return 'lying'
        else:
            return 'standing'
    
    def _detect_running(self, keypoints: np.ndarray) -> bool:
        """
        Detect if person is running based on joint positions
        Running: high leg lift, significant knee bend
        """
        left_hip = keypoints[self.LEFT_HIP]
        right_hip = keypoints[self.RIGHT_HIP]
        left_ankle = keypoints[self.LEFT_ANKLE]
        right_ankle = keypoints[self.RIGHT_ANKLE]
        left_knee = keypoints[10]
        right_knee = keypoints[9]
        
        # Knee lift height relative to hip
        left_knee_lift = max(0, left_hip[1] - left_knee[1])
        right_knee_lift = max(0, right_hip[1] - right_knee[1])
        
        # Running detected if knees are lifted high
        return (left_knee_lift > 30 or right_knee_lift > 30)
    
    def _detect_raised_arms(self, keypoints: np.ndarray) -> bool:
        """
        Detect if person has raised arms
        Raised when wrists are above shoulders
        """
        left_shoulder = keypoints[self.LEFT_SHOULDER]
        right_shoulder = keypoints[self.RIGHT_SHOULDER]
        left_wrist = keypoints[self.LEFT_WRIST]
        right_wrist = keypoints[self.RIGHT_WRIST]
        
        left_raised = left_wrist[1] < left_shoulder[1]
        right_raised = right_wrist[1] < right_shoulder[1]
        
        return left_raised and right_raised
    
    def _calculate_gesture_confidence(self, keypoints: np.ndarray) -> float:
        """Calculate confidence in detected gesture/posture"""
        confidences = keypoints[:, 2]  # z channel contains confidence
        return float(np.mean(confidences[confidences > 0]))
    
    def calculate_movement_speed(self, track_id: int) -> float:
        """Calculate speed from recent keypoint history"""
        if track_id not in self.keypoint_history or len(self.keypoint_history[track_id]) < 2:
            return 0.0
        
        history = self.keypoint_history[track_id]
        curr_nose = history[-1][self.NOSE][:2]
        prev_nose = history[-2][self.NOSE][:2]
        
        speed = np.linalg.norm(curr_nose - prev_nose)
        return float(speed)
    
    def calculate_movement_direction(self, track_id: int) -> float:
        """Calculate movement direction in degrees (0-360)"""
        if track_id not in self.keypoint_history or len(self.keypoint_history[track_id]) < 2:
            return 0.0
        
        history = self.keypoint_history[track_id]
        curr_nose = history[-1][self.NOSE][:2]
        prev_nose = history[-2][self.NOSE][:2]
        
        dx = curr_nose[0] - prev_nose[0]
        dy = curr_nose[1] - prev_nose[1]
        
        angle = np.arctan2(dy, dx) * 180 / np.pi
        return float((angle + 360) % 360)


from collections import deque
