"""
Appearance-based Anomaly Detection
Analyzes clothing attributes and detects statistical deviations
"""
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from enum import Enum

class ClothingType(Enum):
    """Clothing type classification"""
    LIGHT = "Light Clothing"
    NORMAL = "Normal Clothing"
    HEAVY = "Heavy Clothing"
    FORMAL = "Formal Wear"
    ATHLETIC = "Athletic Wear"
    UNKNOWN = "Unknown"


@dataclass
class AppearanceProfile:
    """Appearance characteristics of a person"""
    dominant_color: Tuple[int, int, int]  # BGR
    color_variance: float  # How varied the colors are
    brightness: float  # Average brightness 0-255
    clothing_type: ClothingType
    has_accessories: bool
    has_face_covering: bool
    is_outlier: bool = False
    outlier_confidence: float = 0.0


@dataclass
class AppearanceAnomaly:
    """Detected appearance anomaly"""
    track_id: int
    reason: str
    confidence: float
    profile: AppearanceProfile


class AppearanceAnalyzer:
    """
    Analyzes visual appearance and detects anomalies
    """
    
    def __init__(self, history_length: int = 100):
        self.history_length = history_length
        self.appearance_profiles = {}  # track_id -> list of AppearanceProfile
        self.crowd_baseline = None
        self.is_baseline_set = False
    
    def analyze_person_appearance(self, frame: np.ndarray, 
                                 bbox: Tuple[float, float, float, float]) -> AppearanceProfile:
        """
        Analyze appearance of person in bounding box
        
        Args:
            frame: BGR frame from OpenCV
            bbox: (x1, y1, x2, y2) bounding box
            
        Returns:
            AppearanceProfile
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Extract region
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return AppearanceProfile(
                dominant_color=(128, 128, 128),
                color_variance=0.0,
                brightness=128,
                clothing_type=ClothingType.UNKNOWN,
                has_accessories=False,
                has_face_covering=False
            )
        
        # Analyze colors
        dominant_color = self._get_dominant_color(roi)
        color_variance = self._calculate_color_variance(roi)
        brightness = self._calculate_brightness(roi)
        
        # Classify clothing
        clothing_type = self._classify_clothing(roi, brightness, color_variance)
        
        # Detect accessories/special features
        has_accessories = self._detect_accessories(roi)
        has_face_covering = self._detect_face_covering(roi, bbox[3] - bbox[1])
        
        return AppearanceProfile(
            dominant_color=dominant_color,
            color_variance=color_variance,
            brightness=brightness,
            clothing_type=clothing_type,
            has_accessories=has_accessories,
            has_face_covering=has_face_covering
        )
    
    def detect_appearance_anomalies(self, tracked_persons, frame: np.ndarray) -> List[AppearanceAnomaly]:
        """
        Detect appearance-based anomalies
        
        Args:
            tracked_persons: List of TrackedPerson objects
            frame: Current frame
            
        Returns:
            List of AppearanceAnomaly objects
        """
        anomalies = []
        
        for person in tracked_persons:
            profile = self.analyze_person_appearance(frame, person.bbox)
            
            # Store profile history
            if person.track_id not in self.appearance_profiles:
                self.appearance_profiles[person.track_id] = []
            self.appearance_profiles[person.track_id].append(profile)
            
            # Keep history bounded
            if len(self.appearance_profiles[person.track_id]) > self.history_length:
                self.appearance_profiles[person.track_id].pop(0)
            
            # Update baseline if needed
            if not self.is_baseline_set and len(self.appearance_profiles) > 10:
                self._set_baseline()
            
            # Check for anomalies
            if self.is_baseline_set:
                anomaly = self._check_person_anomaly(person.track_id, profile)
                if anomaly:
                    anomaly.profile = profile
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _get_dominant_color(self, roi: np.ndarray) -> Tuple[int, int, int]:
        """Get most common color in ROI"""
        # Reshape ROI to list of pixels
        pixels = roi.reshape(-1, 3)
        
        # Cluster to dominant color using quantization
        pixels_uint8 = np.uint8(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(pixels_uint8.astype(np.float32), 1, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        
        dominant = tuple(map(int, centers[0]))
        return dominant
    
    def _calculate_color_variance(self, roi: np.ndarray) -> float:
        """Calculate variance in colors (how uniform clothing is)"""
        pixels = roi.reshape(-1, 3).astype(np.float32)
        variance = np.std(pixels)
        return float(variance)
    
    def _calculate_brightness(self, roi: np.ndarray) -> float:
        """Calculate average brightness"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        return float(brightness)
    
    def _classify_clothing(self, roi: np.ndarray, brightness: float, 
                          color_variance: float) -> ClothingType:
        """Classify clothing type"""
        # Simple classification based on features
        if brightness > 200:
            return ClothingType.LIGHT
        elif brightness < 80:
            return ClothingType.HEAVY
        elif color_variance < 30:
            return ClothingType.FORMAL
        elif color_variance > 80:
            return ClothingType.ATHLETIC
        else:
            return ClothingType.NORMAL
    
    def _detect_accessories(self, roi: np.ndarray) -> bool:
        """Detect if person has accessories (bags, backpacks)"""
        # Simple detection: look for distinct edges/shapes
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Many contours suggest accessories
        return len(contours) > 20
    
    def _detect_face_covering(self, roi: np.ndarray, person_height: float) -> bool:
        """Detect face covering (mask, veil)"""
        # Upper portion of ROI (face region)
        face_region = roi[:int(person_height * 0.25)]
        
        if face_region.size == 0:
            return False
        
        # Check for uniform dark color in face region (mask/veil)
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Face covering detected if upper region is uniformly dark
        return brightness < 100
    
    def _set_baseline(self):
        """Set baseline appearance distribution"""
        if len(self.appearance_profiles) < 5:
            return
        
        all_colors = []
        all_brightness = []
        clothing_counts = defaultdict(int)
        
        for profiles in self.appearance_profiles.values():
            if profiles:
                latest = profiles[-1]
                all_colors.append(latest.dominant_color)
                all_brightness.append(latest.brightness)
                clothing_counts[latest.clothing_type] += 1
        
        if not all_colors:
            return
        
        self.crowd_baseline = {
            'avg_color': np.mean(all_colors, axis=0),
            'avg_brightness': np.mean(all_brightness),
            'color_std': np.std(all_colors, axis=0),
            'brightness_std': np.std(all_brightness),
            'dominant_clothing': max(clothing_counts, key=clothing_counts.get)
        }
        
        self.is_baseline_set = True
    
    def _check_person_anomaly(self, track_id: int, profile: AppearanceProfile) -> Optional[AppearanceAnomaly]:
        """Check if person's appearance deviates from crowd baseline"""
        if not self.crowd_baseline:
            return None
        
        # Check color deviation
        color_distance = np.linalg.norm(
            np.array(profile.dominant_color) - self.crowd_baseline['avg_color']
        )
        color_deviation = color_distance / (np.linalg.norm(self.crowd_baseline['color_std']) + 1e-6)
        
        # Check brightness deviation
        brightness_deviation = abs(
            profile.brightness - self.crowd_baseline['avg_brightness']
        ) / (self.crowd_baseline['brightness_std'] + 1e-6)
        
        # Check clothing type mismatch
        clothing_mismatch = 0.5 if profile.clothing_type != self.crowd_baseline['dominant_clothing'] else 0.0
        
        # Combined anomaly score
        anomaly_score = (color_deviation + brightness_deviation) / 2 + clothing_mismatch
        
        # Threshold for anomaly
        if anomaly_score > 2.0:
            return AppearanceAnomaly(
                track_id=track_id,
                reason=f"Unusual appearance (color/brightness deviation: {anomaly_score:.2f})",
                confidence=min(0.95, anomaly_score / 5.0),
                profile=profile
            )
        
        if profile.has_face_covering and not any(
            p.has_face_covering for profiles in self.appearance_profiles.values() 
            for p in profiles[-5:] if profiles
        ):
            return AppearanceAnomaly(
                track_id=track_id,
                reason="Face covering detected",
                confidence=0.6,
                profile=profile
            )
        
        return None
    
    def cleanup_person(self, track_id: int):
        """Remove tracking data for person"""
        self.appearance_profiles.pop(track_id, None)
