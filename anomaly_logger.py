"""
Anomaly Event Logging and Alert Generation
Structures and stores detection results
"""
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import csv
import json
import cv2

@dataclass
class AnomalyEvent:
    """Complete event data for an anomaly detection"""
    timestamp: str
    frame_number: int
    camera_id: str
    track_id: int
    anomaly_types: List[str]  # List of anomaly type names
    behavioral_confidence: float  # Behavioral anomaly score
    appearance_confidence: float  # Appearance anomaly score
    combined_confidence: float  # Combined score
    centroid: tuple  # (x, y)
    bbox: tuple  # (x1, y1, x2, y2)
    descriptions: List[str]  # Descriptions of each anomaly
    metadata: Dict[str, Any] = field(default_factory=dict)
    alert_generated: bool = False
    alert_level: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    snapshot_url: Optional[str] = None  # relative URL to saved frame snapshot


class AnomalyLogger:
    """
    Logs and manages anomaly events
    Provides persistence and alert generation
    """
    
    def __init__(self, output_dir: str = "anomaly_logs", 
                 alert_threshold: float = 0.7):
        """
        Args:
            output_dir: Directory to store logs
            alert_threshold: Confidence threshold for alert generation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.alert_threshold = alert_threshold
        self.events: List[AnomalyEvent] = []
        self.alert_history: Dict[int, List[AnomalyEvent]] = {}  # track_id -> events
        
        # Setup log files
        self.event_log_file = self.output_dir / "anomaly_events.jsonl"
        self.csv_log_file = self.output_dir / "anomaly_summary.csv"
        self.alert_log_file = self.output_dir / "alerts.jsonl"
        
        self._init_csv_headers()
    
    def log_anomalies(self, frame_number: int, camera_id: str,
                     behavioral_anomalies: List, 
                     appearance_anomalies: Optional[List] = None,
                     snapshot_frame=None,
                     tracked_persons=None) -> List[AnomalyEvent]:
        """
        Log detected anomalies and generate alerts
        
        Args:
            frame_number: Frame number in video
            camera_id: Camera identifier
            behavioral_anomalies: List of BehavioralAnomaly objects
            appearance_anomalies: List of AppearanceAnomaly objects
            snapshot_frame: Frame to save as snapshot
            tracked_persons: List of tracked person objects (for bbox info)
            
        Returns:
            List of AnomalyEvent objects created
        """
        events_created = []
        
        # Group by track_id
        events_by_track = {}
        
        for anomaly in behavioral_anomalies:
            if anomaly.track_id not in events_by_track:
                events_by_track[anomaly.track_id] = {
                    'behavioral': [],
                    'appearance': [],
                    'bbox': anomaly.bbox,
                    'centroid': anomaly.centroid
                }
            events_by_track[anomaly.track_id]['behavioral'].append(anomaly)
        
        # for anomaly in appearance_anomalies:
        #     if anomaly.track_id not in events_by_track:
        #         events_by_track[anomaly.track_id] = {
        #             'behavioral': [],
        #             'appearance': [],
        #             'bbox': anomaly.profile.outlier_confidence,
        #             'centroid': (0, 0)  # Will be filled from behavioral if available
        #         }
        #     events_by_track[anomaly.track_id]['appearance'].append(anomaly)
        
        # Create events
        for track_id, data in events_by_track.items():
            behavioral_conf = max(
                [a.confidence for a in data['behavioral']], 
                default=0.0
            )
            appearance_conf = max(
                [a.confidence for a in data['appearance']], 
                default=0.0
            )
            
            combined_conf = max(behavioral_conf, appearance_conf)
            if behavioral_conf > 0 and appearance_conf > 0:
                combined_conf = (behavioral_conf + appearance_conf) / 2
            
            anomaly_types = (
                [a.anomaly_type.name for a in data['behavioral']] +
                ["APPEARANCE_ANOMALY"] * len(data['appearance'])
            )
            
            descriptions = (
                [a.description for a in data['behavioral']] +
                [a.reason for a in data['appearance']]
            )
            
            event = AnomalyEvent(
                timestamp=datetime.now().isoformat(),
                frame_number=frame_number,
                camera_id=camera_id,
                track_id=track_id,
                anomaly_types=anomaly_types,
                behavioral_confidence=behavioral_conf,
                appearance_confidence=appearance_conf,
                combined_confidence=combined_conf,
                centroid=data['centroid'],
                bbox=data['bbox'],
                descriptions=descriptions
            )
            
            # Generate alert if threshold exceeded
            if event.combined_confidence >= self.alert_threshold:
                event = self._generate_alert(event)
                # Save snapshot only when alert generated and frame is provided
                if snapshot_frame is not None:
                    snapshot_url = self._save_snapshot_with_highlight(
                        snapshot_frame, frame_number, track_id, event
                    )
                    event.snapshot_url = snapshot_url
                self._write_alert_record(event)
            
            self.events.append(event)
            events_created.append(event)
            
            # Update track history
            if track_id not in self.alert_history:
                self.alert_history[track_id] = []
            self.alert_history[track_id].append(event)
        
        # Persist to logs
        self._persist_events(events_created)
        
        return events_created
    
    def _generate_alert(self, event: AnomalyEvent) -> AnomalyEvent:
        """Generate alert for high-confidence anomaly"""
        if event.combined_confidence > 0.9:
            alert_level = "CRITICAL"
        elif event.combined_confidence > 0.8:
            alert_level = "HIGH"
        else:
            alert_level = "MEDIUM"

        event.alert_generated = True
        event.alert_level = alert_level
        return event

    def _write_alert_record(self, event: AnomalyEvent) -> None:
        """Persist alert to alerts log"""
        record = {
            'timestamp': event.timestamp,
            'frame': event.frame_number,
            'camera': event.camera_id,
            'track_id': event.track_id,
            'level': event.alert_level,
            'confidence': event.combined_confidence,
            'types': event.anomaly_types,
            'descriptions': event.descriptions,
            'snapshot_url': event.snapshot_url
        }
        with open(self.alert_log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def _save_snapshot(self, frame, frame_number: int, track_id: int) -> Optional[str]:
        """Persist snapshot image to disk and return relative URL"""
        try:
            snaps_dir = self.output_dir / "snapshots"
            snaps_dir.mkdir(parents=True, exist_ok=True)
            filename = f"frame{frame_number}_track{track_id}.jpg"
            filepath = snaps_dir / filename
            cv2.imwrite(str(filepath), frame)
            return f"/snapshots/{filename}"
        except Exception:
            return None

    def _save_snapshot_with_highlight(self, frame, frame_number: int, track_id: int, event: AnomalyEvent) -> Optional[str]:
        """Save snapshot with prominent highlighting of detected anomaly"""
        try:
            # Create a copy to avoid modifying the original
            highlighted_frame = frame.copy()
            
            # Draw bright colored bounding box around the detected area
            if event.bbox:
                x1, y1, x2, y2 = event.bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Color by alert level
                if event.alert_level == "CRITICAL":
                    color = (0, 0, 255)      # Red
                    thickness = 4
                elif event.alert_level == "HIGH":
                    color = (0, 165, 255)    # Orange
                    thickness = 3
                else:
                    color = (0, 255, 255)    # Yellow
                    thickness = 3
                
                # Draw main bounding box
                cv2.rectangle(highlighted_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw corner brackets for emphasis
                corner_length = max(20, int((x2 - x1) * 0.05))
                cv2.line(highlighted_frame, (x1, y1), (x1 + corner_length, y1), color, thickness)
                cv2.line(highlighted_frame, (x1, y1), (x1, y1 + corner_length), color, thickness)
                cv2.line(highlighted_frame, (x2, y1), (x2 - corner_length, y1), color, thickness)
                cv2.line(highlighted_frame, (x2, y1), (x2, y1 + corner_length), color, thickness)
                cv2.line(highlighted_frame, (x1, y2), (x1 + corner_length, y2), color, thickness)
                cv2.line(highlighted_frame, (x1, y2), (x1, y2 - corner_length), color, thickness)
                cv2.line(highlighted_frame, (x2, y2), (x2 - corner_length, y2), color, thickness)
                cv2.line(highlighted_frame, (x2, y2), (x2, y2 - corner_length), color, thickness)
                
                # Draw anomaly type and confidence with background
                label_text = f"Track {track_id} | {', '.join(event.anomaly_types[:2])}"
                confidence_text = f"Confidence: {event.combined_confidence:.2f}"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness_text = 2
                
                # Get text size for background
                (text_width_1, text_height_1), _ = cv2.getTextSize(label_text, font, font_scale, thickness_text)
                (text_width_2, text_height_2), _ = cv2.getTextSize(confidence_text, font, font_scale, thickness_text)
                
                label_y = max(50, y1 - 30)
                confidence_y = label_y + text_height_1 + 20
                
                # Draw background rectangle for labels
                padding = 8
                cv2.rectangle(highlighted_frame, 
                            (10 - padding, label_y - text_height_1 - padding),
                            (max(text_width_1, text_width_2) + 20 + padding, confidence_y + 10),
                            (0, 0, 0), -1)  # Black background
                
                # Draw text
                cv2.putText(highlighted_frame, label_text, (10, label_y),
                           font, font_scale, color, thickness_text)
                cv2.putText(highlighted_frame, confidence_text, (10, confidence_y),
                           font, font_scale, (0, 255, 0), thickness_text)
                
                # Draw alert level badge in top-right
                alert_text = event.alert_level
                (alert_width, alert_height), _ = cv2.getTextSize(alert_text, font, 1.0, 2)
                alert_x = highlighted_frame.shape[1] - alert_width - 20
                alert_y = 50
                
                alert_color = (0, 0, 255) if event.alert_level == "CRITICAL" else (0, 165, 255) if event.alert_level == "HIGH" else (0, 255, 255)
                cv2.rectangle(highlighted_frame,
                            (alert_x - 10, alert_y - alert_height - 10),
                            (alert_x + alert_width + 10, alert_y + 10),
                            alert_color, -1)
                cv2.putText(highlighted_frame, alert_text, (alert_x, alert_y),
                           font, 1.0, (0, 0, 0), 2)
            
            # Save the highlighted snapshot
            snaps_dir = self.output_dir / "snapshots"
            snaps_dir.mkdir(parents=True, exist_ok=True)
            filename = f"frame{frame_number}_track{track_id}.jpg"
            filepath = snaps_dir / filename
            cv2.imwrite(str(filepath), highlighted_frame)
            return f"/snapshots/{filename}"
        except Exception as e:
            print(f"Error saving highlighted snapshot: {e}")
            return None
    
    def _persist_events(self, events: List[AnomalyEvent]):
        """Persist events to log files"""
        # JSONL format (one JSON per line)
        with open(self.event_log_file, 'a') as f:
            for event in events:
                event_dict = asdict(event)
                f.write(json.dumps(event_dict) + '\n')
        
        # CSV format
        with open(self.csv_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for event in events:
                writer.writerow([
                    event.timestamp,
                    event.frame_number,
                    event.camera_id,
                    event.track_id,
                    '|'.join(event.anomaly_types),
                    f"{event.behavioral_confidence:.3f}",
                    f"{event.appearance_confidence:.3f}",
                    f"{event.combined_confidence:.3f}",
                    event.alert_level if event.alert_generated else "NONE",
                    '|'.join(event.descriptions[:2])  # First 2 descriptions
                ])
    
    def _init_csv_headers(self):
        """Initialize CSV headers if file is new"""
        if not self.csv_log_file.exists():
            with open(self.csv_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp',
                    'Frame',
                    'Camera',
                    'Track_ID',
                    'Anomaly_Types',
                    'Behavioral_Confidence',
                    'Appearance_Confidence',
                    'Combined_Confidence',
                    'Alert_Level',
                    'Description'
                ])
    
    def get_track_anomalies(self, track_id: int) -> List[AnomalyEvent]:
        """Get all logged anomalies for a specific track"""
        return self.alert_history.get(track_id, [])
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of logged anomalies"""
        if not self.events:
            return {
                'total_events': 0,
                'total_alerts': 0,
                'affected_tracks': 0,
                'anomaly_types': {}
            }
        
        anomaly_counts = {}
        for event in self.events:
            for atype in event.anomaly_types:
                anomaly_counts[atype] = anomaly_counts.get(atype, 0) + 1
        
        return {
            'total_events': len(self.events),
            'total_alerts': sum(1 for e in self.events if e.alert_generated),
            'affected_tracks': len(self.alert_history),
            'anomaly_types': anomaly_counts,
            'avg_confidence': np.mean([e.combined_confidence for e in self.events]) if self.events else 0.0
        }

    def get_recent_alerts(self, level: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Return latest alerts from alerts log (filtered by level if provided)."""
        if not self.alert_log_file.exists():
            return []

        alerts: List[Dict[str, Any]] = []
        try:
            with open(self.alert_log_file, 'r') as f:
                lines = f.readlines()
            # Read most recent first
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if level and record.get('level') != level:
                    continue
                alerts.append(record)
                if len(alerts) >= limit:
                    break
        except Exception:
            return []

        # Return newest first
        return alerts

    def clear_all(self) -> None:
        """Clear all logs and state for fresh analysis."""
        self.events = []
        self.alert_history = {}
        # Remove log files
        for log_file in [self.event_log_file, self.csv_log_file, self.alert_log_file]:
            if log_file.exists():
                log_file.unlink()
        # Remove snapshots directory
        snaps_dir = self.output_dir / "snapshots"
        if snaps_dir.exists():
            import shutil
            shutil.rmtree(snaps_dir)
        # Reinitialize CSV headers
        self._init_csv_headers()
    
    def export_report(self, filepath: str):
        """Export comprehensive report"""
        stats = self.get_summary_stats()
        
        report = {
            'generation_time': datetime.now().isoformat(),
            'summary': stats,
            'top_anomaly_tracks': self._get_top_anomaly_tracks(5),
            'alert_timeline': self._generate_timeline()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def _get_top_anomaly_tracks(self, n: int = 5) -> List[Dict]:
        """Get top N tracks by anomaly frequency"""
        tracks = [
            {
                'track_id': tid,
                'event_count': len(events),
                'avg_confidence': np.mean([e.combined_confidence for e in events])
            }
            for tid, events in sorted(
                self.alert_history.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:n]
        ]
        return tracks
    
    def _generate_timeline(self) -> List[Dict]:
        """Generate timeline of high-confidence events"""
        timeline = [
            {
                'timestamp': e.timestamp,
                'frame': e.frame_number,
                'track_id': e.track_id,
                'confidence': e.combined_confidence,
                'alert_level': e.alert_level
            }
            for e in sorted(self.events, key=lambda x: x.timestamp)
            if e.alert_generated
        ]
        return timeline


import numpy as np
