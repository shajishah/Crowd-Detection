# Crowd Anomaly Detection System - Implementation Guide

## Overview
A complete implementation of an advanced crowd anomaly detection system that combines computer vision, pose analysis, crowd behavior modeling, and appearance-based anomaly detection.

## System Architecture

### 1. **Human Detection & Tracking** (`tracker.py`)
- **Multi-Object Tracker**: Centroid-based tracking with persistent ID assignment
- **Features**:
  - Tracks individuals across frames with unique persistent IDs
  - Handles occlusion through ID retention for 5+ frames
  - Maintains trajectory history (last 30 frames per person)
  - Calculates velocity vectors for motion analysis

**Key Classes**:
- `TrackedPerson`: Represents a tracked individual with ID, position, trajectory, velocity
- `MultiObjectTracker`: Main tracking engine with greedy Hungarian-style matching

---

### 2. **Pose & Body Movement Analysis** (`pose_analyzer.py`)
- **MediaPipe-based**: Extracts 33 skeletal keypoints per person
- **Features**:
  - Detects posture type: standing, crouching, sitting, lying
  - Identifies running (high knee lift > 30px)
  - Detects raised arms (unusual gestures)
  - Calculates movement speed and direction
  - Maintains pose history for trajectory analysis

**Key Classes**:
- `PoseAnalysis`: Data structure for pose detection results
- `PoseAnalyzer`: Processes video frames and extracts pose information

---

### 3. **Crowd Behavior Modeling** (`crowd_behavior.py`)
- **Collective Motion Analysis**
- **Features**:
  - Calculates dominant crowd flow direction (circular mean)
  - Measures flow magnitude (0-1: random to highly organized)
  - Tracks average movement speed and crowd density
  - Establishes baseline behavior after 30+ frames of history
  - Visualizes flow with directional arrows

**Key Classes**:
- `CrowdFlowMetrics`: Aggregated crowd statistics per frame
- `CrowdBehaviorAnalyzer`: Analyzes collective behavior patterns

---

### 4. **Behavioral Outlier Detection** (`outlier_detector.py`)
Identifies individuals with anomalous movement patterns:

**Detected Anomalies**:
1. **Against Crowd Flow**: Moving opposite to dominant direction (angle > 90°)
2. **Erratic Movement**: High angular changes in trajectory
3. **Stationary in High-Flow**: Not moving while crowd is active (15+ frames)
4. **Running**: Detected from pose analysis
5. **Raised Arms**: Unusual gestures in calm environments
6. **Crouching**: Detected from pose analysis

**Confidence Scoring**: 0.5 to 0.9 based on behavior severity

---

### 5. **Appearance Anomaly Detection** (`appearance_analyzer.py`)
Analyzes visual appearance and detects statistical deviations:

**Features**:
- Extracts dominant color from person's bounding box
- Calculates color variance (clothing uniformity)
- Measures brightness levels
- Classifies clothing type: Light, Normal, Heavy, Formal, Athletic
- Detects accessories (bags, backpacks via edge detection)
- Detects face coverings (masks, veils)
- Establishes crowd baseline after observing 5+ people
- Flags individuals >2σ deviations from baseline

---

### 6. **Anomaly Event Logger** (`anomaly_logger.py`)
Structures, stores, and reports detections:

**Output Formats**:
- **JSONL** (`anomaly_events.jsonl`): One JSON per line, detailed records
- **CSV** (`anomaly_summary.csv`): Summary table for analysis
- **Alert Log** (`alerts.jsonl`): High-confidence alerts only

**Alert Levels**:
- **CRITICAL**: Combined confidence > 0.9 (Red box)
- **HIGH**: Combined confidence 0.8-0.9 (Orange box)
- **MEDIUM**: Combined confidence 0.7-0.8 (Yellow box)
- **LOW**: Below threshold

**Data Captured**:
- Timestamp and frame number
- Track ID and camera ID
- Anomaly types and descriptions
- Behavioral and appearance confidence scores
- Bounding box and centroid coordinates

---

## Integration in main.py

### Processing Pipeline Per Frame:
```
1. YOLO Detection → Get bounding boxes
2. Multi-Object Tracking → Assign persistent IDs
3. Crowd Behavior Analysis → Calculate flow metrics
4. Pose Estimation → Extract keypoints per person
5. Behavioral Anomaly Detection → Flag unusual movements
6. Appearance Analysis → Flag unusual clothing
7. Event Logging → Record and alert
8. Visualization → Draw boxes with confidence scores
```

### Visualization Output:
- **Green Box**: Normal tracked person
- **Yellow Box**: Medium anomaly (confidence 0.7-0.8)
- **Orange Box**: High anomaly (confidence 0.8-0.9)
- **Red Box**: Critical anomaly (confidence > 0.9)
- **Text Labels**: Track ID, confidence score, anomaly types
- **Flow Arrow**: Dominant crowd direction and strength

---

## Configuration

### Key Parameters (in main.py):
```python
tracker = MultiObjectTracker(max_distance=50, max_frames_to_skip=5)
# max_distance: Centroid matching threshold (pixels)
# max_frames_to_skip: Frames to retain ID before dropping track

crowd_analyzer = CrowdBehaviorAnalyzer(history_length=60)
# history_length: Frames to maintain for baseline calculation

appearance_detector = AppearanceAnalyzer(history_length=100)
# history_length: Frames for appearance baseline

anomaly_logger = AnomalyLogger(output_dir="anomaly_logs", alert_threshold=0.7)
# alert_threshold: Confidence score for alert generation
```

---

## Output Files

### Generated in `anomaly_logs/` directory:

1. **anomaly_events.jsonl**
   - Complete event records with all metadata
   - One JSON object per line
   
2. **anomaly_summary.csv**
   - Tabular format for spreadsheet analysis
   - Columns: Timestamp, Frame, Camera, Track_ID, Anomaly_Types, Scores, Alert_Level

3. **alerts.jsonl**
   - Only high-confidence alerts (threshold >= 0.7)
   - For operator review and investigation

---

## Dependencies

### New Packages (added to requirements.txt):
- `mediapipe==0.10.8`: Pose estimation
- `supervision==0.19.0`: Detection utilities
- `scipy==1.11.4`: Scientific computing
- `pandas==2.1.3`: Data manipulation
- `deep-sort-realtime==1.3.2`: Advanced tracking (optional)

### Existing Packages:
- `torch==2.1.2`: PyTorch for inference
- `ultralytics==8.1.0`: YOLOv8 detection
- `opencv-python==4.8.1.78`: Video processing

---

## Installation & Usage

### 1. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the application:
```bash
python main.py
```

### 3. Access web interface:
```
http://localhost:5000
```

### 4. API Endpoints:
- `/` - Main page
- `/upload` - Upload video (POST)
- `/process/<video_name>` - Stream processed video
- `/download` - Download processed video
- `/anomaly_report` - Get JSON report of detected anomalies

---

## Key Metrics & Thresholds

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Flow Magnitude | > 0.3 | Crowd has clear directional flow |
| Against Flow Angle | > 90° | Person moving opposite direction |
| Stationary Frames | > 15 | Person stationary in active crowd |
| Pose Confidence | > 0.5 | Valid pose detection |
| Appearance Deviation | > 2σ | Unusual clothing/colors |
| Alert Confidence | ≥ 0.7 | Generate alert for operators |

---

## Example Output

### Console Alert:
```
Frame 245, Camera 1:
  Track ID 42: CRITICAL - Combined Confidence 0.92
  - Against Crowd Flow (0.85)
  - Raised Arms (0.78)
  - Unusual Appearance (0.89)
```

### CSV Log Entry:
```
2024-01-26T10:15:32.456,245,camera_1,42,
AGAINST_FLOW|RAISED_ARMS|APPEARANCE_ANOMALY,
0.850,0.890,0.920,CRITICAL,
Moving against flow|Unusual gesture
```

---

## Future Enhancements

1. **Advanced Tracking**: Implement DeepSORT with appearance features
2. **Crowd Counting**: Estimate total persons and density maps
3. **Multi-Camera**: Track across camera boundaries
4. **Action Recognition**: Detect fighting, falling, pushing
5. **Real-time Alerts**: Push notifications to security personnel
6. **ML-based Baselines**: Learn normal patterns per location/time
7. **GPU Acceleration**: Optimize for real-time performance
8. **Dashboard**: Web interface for operator monitoring

---

## Performance Notes

- **Frame Processing**: ~100-200ms per frame (CPU dependent)
- **Memory**: ~4-8GB for continuous processing
- **Accuracy**: Depends on video quality, lighting, and crowd density
- **False Positives**: Reduce by adjusting `alert_threshold` or confidence thresholds

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| PyTorch compatibility | Verify torch==2.1.2 installed |
| Pose detection fails | Ensure frame is RGB, good lighting |
| Tracker IDs flicker | Increase `max_frames_to_skip` |
| Too many false alerts | Increase `alert_threshold` |
| Memory issues | Process shorter videos or lower resolution |

---

## Author & Version
- Implementation Date: January 26, 2026
- Version: 1.0
- Framework: Flask + YOLO + MediaPipe
