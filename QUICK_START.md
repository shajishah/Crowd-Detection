# Quick Start Guide - Crowd Anomaly Detection System

## Installation (First Time)

```bash
# Navigate to project directory
cd "G:\Crowd Detection\Crowd-Density-Estimation"

# Activate virtual environment
# Windows:
crowd\Scripts\activate

# Install/upgrade all dependencies
pip install -r requirements.txt --upgrade

# This will install:
# - Flask 2.3.3 (web server)
# - OpenCV 4.8.1.78 (video processing)
# - YOLOv8 8.1.0 (detection)
# - MediaPipe 0.10.8 (pose estimation)
# - PyTorch 2.1.2 (deep learning)
# - Plus support libraries (scipy, pandas, supervision)
```

## Running the System

### Option 1: Web Interface (Recommended)
```bash
python main.py
```
Then open: **http://localhost:5000**

Features:
- Upload MP4/AVI/MOV videos
- Real-time stream of processed video
- Download processed output
- View anomaly reports

### Option 2: Direct Processing
```python
from main import process_video, tracker, crowd_analyzer, # ...
import cv2

input_video = "videos/crowd_scene.mp4"
for frame_bytes in process_video(input_video):
    # Process streamed frames
    pass
```

---

## What Gets Detected?

### Behavioral Anomalies:
✅ People moving **against crowd flow**  
✅ **Erratic/abrupt** movements  
✅ **Stationary** while crowd is moving  
✅ **Running** behavior  
✅ **Raised arms** gestures  
✅ **Crouching** in calm areas  

### Appearance Anomalies:
✅ **Unusual colors** (statistical outliers)  
✅ **Unusual brightness** (too dark/light clothing)  
✅ **Wrong clothing type** for environment  
✅ **Accessories** (bags inconsistent with crowd)  
✅ **Face coverings** (masks, veils unusual for setting)  

---

## Understanding the Output

### Video Overlay (Real-Time):
```
┌─────────────────────────────────────┐
│ ID:42 [0.92]                        │  ← Red box: Track ID + Confidence
│  AGAINST_FLOW                       │  ← Anomaly type
│  RAISED_ARMS                        │  ← Additional anomaly
│                                     │
│ ◄── Dominant crowd flow direction   │  ← Yellow arrow = flow strength
│ Direction: 245°                     │
│ Speed: 5.23 px/frame                │
│ Activity: 0.85                      │
└─────────────────────────────────────┘
```

### Bounding Box Colors:
- 🟩 **Green**: Normal person, no anomalies
- 🟨 **Yellow**: MEDIUM anomaly (confidence 0.7-0.8)
- 🟧 **Orange**: HIGH anomaly (confidence 0.8-0.9)
- 🟥 **Red**: CRITICAL anomaly (confidence > 0.9)

### Log Files (in `anomaly_logs/`):

**anomaly_events.jsonl**:
```json
{
  "timestamp": "2024-01-26T10:15:32.456",
  "frame_number": 245,
  "camera_id": "camera_1",
  "track_id": 42,
  "anomaly_types": ["AGAINST_FLOW", "RAISED_ARMS"],
  "combined_confidence": 0.92,
  "alert_level": "CRITICAL",
  "descriptions": ["Moving against dominant crowd flow", "Raised arms in calm environment"]
}
```

**anomaly_summary.csv**:
```
Timestamp,Frame,Camera,Track_ID,Anomaly_Types,Behavioral_Confidence,Appearance_Confidence,Combined_Confidence,Alert_Level,Description
2024-01-26T10:15:32.456,245,camera_1,42,AGAINST_FLOW|RAISED_ARMS,0.850,0.000,0.850,MEDIUM,Moving against flow|Raised arms
```

---

## API Endpoints

### Upload & Process Video
```
POST /upload
- File: video (mp4, avi, or mov)
- Returns: HTML page with stream link
```

### Stream Processed Video
```
GET /process/<video_name>
- Returns: MJPEG stream with real-time anomaly visualization
- Open in browser for live feed
```

### Download Processed Output
```
GET /download
- Returns: MP4 file with annotations saved
```

### Get Anomaly Report
```
GET /anomaly_report
- Returns: JSON with statistics
```

**Example cURL**:
```bash
# Upload video
curl -F "video=@crowd_scene.mp4" http://localhost:5000/upload

# Get report
curl http://localhost:5000/anomaly_report
```

---

## Analyzing Results

### Top Tracks Report
Most frequently flagged individuals:
```json
{
  "track_id": 42,
  "event_count": 23,
  "avg_confidence": 0.82
}
```

### Statistics Summary
```json
{
  "total_events": 156,
  "total_alerts": 47,
  "affected_tracks": 23,
  "anomaly_types": {
    "AGAINST_FLOW": 34,
    "ERRATIC_MOVEMENT": 28,
    "APPEARANCE_ANOMALY": 15,
    "RUNNING": 11,
    "RAISED_ARMS": 19
  }
}
```

---

## Configuration Tips

### Adjust Sensitivity
**In main.py**, modify:
```python
# More lenient detection (lower threshold = more alerts)
anomaly_logger = AnomalyLogger(alert_threshold=0.65)

# More strict (higher threshold = fewer false positives)
anomaly_logger = AnomalyLogger(alert_threshold=0.75)

# Tracking sensitivity (larger distance = more track drift)
tracker = MultiObjectTracker(max_distance=60)

# Crowd analysis window (smaller = faster baseline, more noise)
crowd_analyzer = CrowdBehaviorAnalyzer(history_length=45)
```

### Video Quality Settings
```python
# In process_video function:
# Lower confidence threshold = detects smaller/distant people
results = model(frame, conf=0.10)  # Currently 0.10

# For busier crowds, increase to:
results = model(frame, conf=0.20)
```

---

## Common Issues & Solutions

### Issue: Too many false positives
**Solution**:
- Increase `alert_threshold` to 0.75-0.80
- Increase `crowd_analyzer.history_length` to 90
- Adjust crowd flow threshold (currently 0.3)

### Issue: Missing detections in crowded areas
**Solution**:
- Lower YOLO confidence: change `conf=0.10` to `conf=0.05`
- Increase pose detection confidence in `pose_analyzer.py`

### Issue: IDs flickering/changing frequently
**Solution**:
- Increase `max_frames_to_skip` from 5 to 10
- Increase `max_distance` from 50 to 75

### Issue: Slow processing
**Solution**:
- Reduce video resolution
- Lower pose estimation complexity (`model_complexity=0`)
- Process shorter clips
- Use GPU if available

---

## Example Workflow

### 1. Upload Video
```
1. Open http://localhost:5000
2. Click "Choose File"
3. Select crowd_video.mp4
4. Click "Upload"
```

### 2. Monitor Processing
```
Real-time stream appears with:
- Colored bounding boxes
- Track IDs
- Confidence scores
- Anomaly types
```

### 3. Review Logs
```
Open anomaly_logs/anomaly_summary.csv in Excel:
- Sort by Combined_Confidence (descending)
- Filter by Alert_Level = "CRITICAL"
- Export for investigation
```

### 4. Export Report
```python
from anomaly_logger import anomaly_logger
anomaly_logger.export_report("report_2024-01-26.json")
```

---

## Video Requirements

| Parameter | Requirement |
|-----------|-------------|
| **Format** | MP4, AVI, or MOV |
| **Resolution** | 480p+ recommended |
| **FPS** | 25+ recommended |
| **Duration** | Tested up to 30 min |
| **File Size** | <1GB recommended |

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.8 | 3.10+ |
| **RAM** | 8GB | 16GB |
| **CPU** | Dual-core | i7/Ryzen 5+ |
| **GPU** | Optional | NVIDIA RTX 3060+ |
| **Storage** | 5GB free | 20GB free |

---

## Monitoring in Production

### Real-time Alert Endpoint
Create a separate monitoring script:
```python
import requests
import json
from datetime import datetime

while True:
    try:
        response = requests.get('http://localhost:5000/anomaly_report')
        data = response.json()
        
        if data.get('statistics', {}).get('total_alerts', 0) > 10:
            print(f"🚨 HIGH ALERT COUNT: {data['statistics']['total_alerts']}")
            print(json.dumps(data['top_tracks'][:3], indent=2))
        
        time.sleep(5)  # Check every 5 seconds
    except Exception as e:
        print(f"Error: {e}")
```

---

## Support & Debugging

### Enable Debug Logging
```python
# Add to main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Profile Performance
```python
import cProfile
cProfile.run('process_video("test.mp4")')
```

### Test Components Individually
```python
# Test tracker
from tracker import MultiObjectTracker
tracker = MultiObjectTracker()
persons = tracker.update([(10,20,50,80)])

# Test pose
from pose_analyzer import PoseAnalyzer
analyzer = PoseAnalyzer()
pose = analyzer.analyze_frame(frame_rgb)

# Test crowd analysis
from crowd_behavior import CrowdBehaviorAnalyzer
crowd = CrowdBehaviorAnalyzer()
metrics = crowd.analyze_frame(persons, frame.shape)
```

---

**Last Updated**: January 26, 2026  
**Version**: 1.0.0
