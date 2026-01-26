# Crowd Anomaly Detection System - Complete Technical Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Core Features](#core-features)
4. [Module Documentation](#module-documentation)
5. [Data Flow](#data-flow)
6. [API Endpoints](#api-endpoints)
7. [Configuration](#configuration)
8. [Installation & Setup](#installation--setup)
9. [Usage Guide](#usage-guide)
10. [Technology Stack](#technology-stack)

---

## 🎯 Project Overview

### What is This Project?

The **Crowd Anomaly Detection System** is an advanced AI-powered video analytics platform designed for real-time monitoring and analysis of crowd behavior in public spaces. It combines computer vision, deep learning, and behavioral analysis to identify unusual patterns, potential security threats, and anomalous activities in crowded environments.

### Primary Use Cases

1. **Public Safety Monitoring** - Detect suspicious behavior in airports, malls, stadiums
2. **Event Security** - Monitor large gatherings for anomalies
3. **Crowd Management** - Analyze flow patterns and density
4. **Incident Response** - Generate actionable alerts with evidence
5. **Forensic Analysis** - Post-event investigation with detailed reports

### Key Capabilities

- **Real-time Detection**: Process video streams with live anomaly detection
- **Multi-Modal Analysis**: Combines pose, appearance, behavior, and crowd flow data
- **Persistent Tracking**: Maintains person IDs across frames
- **Alert Generation**: Automatic alerts with confidence scores and snapshots
- **AI-Powered Reports**: LLM-generated comprehensive analysis reports
- **Interactive Dashboard**: Web-based UI for monitoring and review

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Interface (Flask)                     │
│  ┌──────────┐  ┌───────────┐  ┌────────────┐  ┌──────────────┐ │
│  │ Video    │  │ Live      │  │ Alert      │  │ AI Report    │ │
│  │ Upload   │  │ Stream    │  │ Dashboard  │  │ Generator    │ │
│  └──────────┘  └───────────┘  └────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Video Processing Pipeline                   │
│  ┌────────────┐   ┌──────────────┐   ┌───────────────────┐     │
│  │ YOLOv8     │ → │ Multi-Object │ → │ Feature           │     │
│  │ Detection  │   │ Tracker      │   │ Extraction        │     │
│  └────────────┘   └──────────────┘   └───────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Anomaly Analysis Layer                        │
│  ┌───────────┐  ┌──────────────┐  ┌─────────────┐  ┌─────────┐│
│  │ Pose      │  │ Crowd        │  │ Appearance  │  │ Behavior││
│  │ Analysis  │  │ Behavior     │  │ Analysis    │  │ Outlier ││
│  └───────────┘  └──────────────┘  └─────────────┘  └─────────┘│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Alert & Logging System                        │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────────────┐    │
│  │ Anomaly      │  │ Snapshot   │  │ LLM Report          │    │
│  │ Logger       │  │ Storage    │  │ Generator           │    │
│  └──────────────┘  └────────────┘  └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

1. **Input Layer**: Video upload → Frame extraction
2. **Detection Layer**: YOLOv8 person detection
3. **Tracking Layer**: Persistent ID assignment and trajectory tracking
4. **Analysis Layer**: Multi-modal anomaly detection
5. **Decision Layer**: Alert generation based on thresholds
6. **Output Layer**: Visualization, logging, and reporting

---

## ✨ Core Features

### 1. Advanced Person Detection & Tracking

**Technology**: YOLOv8m (Medium variant)
**Capabilities**:
- Real-time person detection with bounding boxes
- Confidence-based filtering
- Handles occlusions and partial views

**Tracking System**:
- **Persistent IDs**: Each person maintains same ID across frames
- **Trajectory Recording**: Last 30 positions stored per person
- **Velocity Calculation**: Motion vectors computed from trajectories
- **Ghost Track Handling**: Drops tracks after 5 missed frames
- **Distance-Based Matching**: Hungarian algorithm for optimal assignment

### 2. Multi-Modal Anomaly Detection

#### A. Behavioral Anomaly Detection
**Module**: `outlier_detector.py`

Detects:
- **Against-Flow Movement**: Person moving opposite to crowd
- **Erratic Motion**: Sudden direction changes, zigzag patterns
- **Stationary Behavior**: Loitering in high-traffic areas
- **Running/Fast Movement**: Abnormally high speeds
- **Unusual Gestures**: Arms raised, aggressive postures

**Method**: Statistical outlier detection against crowd baseline

#### B. Appearance Anomaly Detection
**Module**: `appearance_analyzer.py`

Detects:
- **Color Outliers**: Unusual clothing colors (e.g., all black)
- **Uniform Detection**: Security personnel, staff
- **Accessory Detection**: Backpacks, helmets, masks
- **Profile Clustering**: Identifies appearance patterns

**Method**: Color histogram analysis + statistical deviation

#### C. Crowd Behavior Analysis
**Module**: `crowd_behavior.py`

Analyzes:
- **Dominant Flow Direction**: Primary crowd movement angle
- **Average Speed**: Collective movement rate
- **Density Levels**: Person count per frame area
- **Activity Level**: Overall crowd dynamics
- **Flow Magnitude**: Strength of directional consensus

**Method**: Optical flow + baseline establishment

#### D. Pose Analysis
**Module**: `pose_analyzer.py`

Detects:
- **Aggressive Postures**: Raised arms, fighting stance
- **Fall Detection**: Horizontal body orientation
- **Unusual Gestures**: Arms above head
- **Body Keypoints**: 33-point MediaPipe skeleton

**Method**: MediaPipe Pose Estimation

### 3. Alert System

**Severity Levels**:
- **CRITICAL**: Confidence ≥ 0.9 (Immediate action required)
- **HIGH**: Confidence ≥ 0.8 (Priority review)
- **MEDIUM**: Confidence ≥ 0.7 (Monitor closely)
- **LOW**: Confidence < 0.7 (Log only)

**Alert Components**:
- Timestamp and frame number
- Track ID and camera ID
- Anomaly type(s) and descriptions
- Confidence scores (behavioral, appearance, combined)
- Person location (centroid, bounding box)
- **Snapshot Image**: Captured frame with annotations
- Contextual metadata (crowd density, location, time-of-day)

### 4. Snapshot Management

**Automatic Capture**:
- Triggered when alert threshold exceeded
- Saves annotated frame to `anomaly_logs/snapshots/`
- Filename: `frame{N}_track{ID}.jpg`
- Includes bounding boxes and confidence overlays

**Storage**:
- Served via Flask route `/snapshots/<filename>`
- URLs stored in alert records
- Displayed in alert detail drawer

### 5. AI-Powered Report Generation

**Supported LLM Providers**:
- **OpenAI**: GPT-3.5-turbo, GPT-4
- **Google Gemini**: gemini-pro, gemini-2.5-flash
- **Groq**: mixtral-8x7b-32768 (recommended - fast & free)
- **Ollama**: Local models (offline capability)

**Report Contents**:
- Safety assessment (Safe/Moderate/High Risk)
- Key findings and anomaly patterns
- Top persons of interest
- Actionable recommendations
- Temporal trend analysis
- Executive summary format

**Caching**:
- Reports cached for 60 seconds
- Prevents API quota exhaustion
- Fallback to template report if LLM unavailable

### 6. Interactive Web Dashboard

**Components**:
- **Video Upload Panel**: Drag-and-drop file selector
- **Live Stream Viewer**: Real-time processing with overlay
- **KPI Cards**: Total events, alerts, tracks, avg confidence
- **Critical Alerts Table**: Most recent high-priority alerts
- **Alert Timeline**: Chronological view with drill-down
- **Alert Detail Drawer**: Full metadata + snapshot viewer
- **AI Report Modal**: LLM-generated analysis with download
- **Top Tracks Table**: Frequent anomaly offenders
- **Anomaly Type Breakdown**: Distribution chart

**Features**:
- Auto-refresh every 5 seconds
- Toast notifications for new critical alerts
- Reset button to clear all stats
- Download processed video
- Export AI report as TXT

### 7. Density Heatmap

**Technology**: Gaussian blur + color mapping
**Visualization**:
- Turbo colormap (blue=low, red=high)
- Aggregates by clustering centroids
- Overlay on original video frame
- Intensity based on person count per region

### 8. Logging & Persistence

**Log Files** (in `anomaly_logs/`):
- `anomaly_events.jsonl`: Full event details (JSONL format)
- `anomaly_summary.csv`: Tabular summary for spreadsheets
- `alerts.jsonl`: Alert-only records with snapshots
- `snapshots/`: Captured frame images

**Data Retention**:
- Events stored per track ID
- Historical trends for behavioral baselines
- Exportable for external analysis

---

## 📚 Module Documentation

### 1. `main.py` - Flask Application Core

**Responsibilities**:
- Web server initialization
- Route handling
- Video processing orchestration
- Module integration

**Key Functions**:

#### `process_video(input_path)`
Main processing pipeline:
1. Opens video file with OpenCV
2. Iterates frame-by-frame
3. Runs YOLOv8 detection
4. Updates tracker with detections
5. Analyzes crowd behavior
6. Extracts pose keypoints
7. Detects behavioral anomalies
8. Detects appearance anomalies
9. Logs anomalies and generates alerts
10. Visualizes results on frame
11. Streams to browser and saves output

**Returns**: MJPEG stream for live viewing

#### `visualize_anomalies(frame, tracked_persons, anomaly_events, crowd_metrics)`
Overlay visualization:
- Color-coded bounding boxes (green=normal, yellow/orange/red=anomaly)
- Track IDs and confidence scores
- Anomaly type labels
- Crowd flow vectors

**Configuration** (via `.env`):
- `UPLOAD_FOLDER`: Where uploaded videos are stored
- `OUTPUT_FOLDER`: Processed video destination
- `ALERT_THRESHOLD`: Minimum confidence for alerts (default: 0.7)
- `MAX_TRACKING_DISTANCE`: Centroid matching threshold
- `MAX_FRAMES_TO_SKIP`: Track dropout limit

**API Endpoints**:
- `GET /`: Main dashboard
- `POST /upload`: Upload video file
- `GET /process/<video_path>`: Stream processing results
- `GET /download`: Download processed video
- `GET /anomaly_report`: Get statistics JSON
- `GET /alerts?level=CRITICAL&limit=50`: Fetch recent alerts
- `POST /reset`: Clear all logs and state
- `POST /generate_report`: Generate AI analysis report
- `GET /snapshots/<filename>`: Serve snapshot images

---

### 2. `tracker.py` - Multi-Object Tracking System

**Purpose**: Maintain persistent person IDs across video frames

**Class**: `TrackedPerson`
**Attributes**:
- `track_id`: Unique identifier
- `centroid`: (x, y) center position
- `bbox`: (x1, y1, x2, y2) bounding box
- `trajectory`: deque of last 30 positions
- `confidence`: Detection confidence
- `frames_since_seen`: Frames without detection
- `appearance_feature`: Color histogram (optional)
- `pose_keypoints`: MediaPipe skeleton (optional)

**Methods**:
- `update(centroid, bbox)`: Update position and trajectory
- `get_velocity()`: Calculate motion vector
- `get_trajectory_array()`: Return trajectory as numpy array

**Class**: `MultiObjectTracker`
**Algorithm**: Centroid tracking with Hungarian assignment

**Parameters**:
- `max_distance=50`: Maximum pixels between frames for same track
- `max_frames_to_skip=5`: Frames before dropping track

**Methods**:

#### `update(detections) → List[TrackedPerson]`
1. Calculate centroids from bounding boxes
2. Match detections to existing tracks (minimize distance)
3. Update matched tracks
4. Create new tracks for unmatched detections
5. Increment `frames_since_seen` for unmatched tracks
6. Remove stale tracks (> max_frames_to_skip)

**Matching Strategy**:
- Uses scipy's `linear_sum_assignment` (Hungarian algorithm)
- Distance matrix: Euclidean distance between centroids
- Only matches within `max_distance` threshold

**Output**: List of active `TrackedPerson` objects with updated states

---

### 3. `pose_analyzer.py` - Pose Estimation & Analysis

**Technology**: Google MediaPipe Pose
**Skeleton**: 33 keypoints per person

**Class**: `PoseAnalyzer`

**Methods**:

#### `analyze_frame(frame_rgb) → Optional[PoseData]`
- Converts BGR to RGB
- Runs MediaPipe pose detection
- Extracts 33 (x, y, z, visibility) keypoints
- Returns `PoseData` object or None

**PoseData Attributes**:
- `keypoints`: numpy array (33, 4) - [x, y, z, visibility]
- `landmarks`: MediaPipe landmark list

#### `update_trajectory(track_id, keypoints)`
- Stores pose history per track ID
- Used for gesture and anomaly detection

#### `detect_aggressive_pose(pose_data) → bool`
Detects:
- Arms raised above shoulders
- Hands above head
- Fighting stance

**Method**: Analyzes shoulder-wrist-elbow angles and elevations

#### `detect_fall(pose_data) → bool`
Detects horizontal body orientation:
- Hip-shoulder angle near horizontal
- Sudden vertical position change

**Use Cases**:
- Security: Detect fights, falls
- Health: Elderly fall detection
- Events: Crowd surfing, stampede indicators

---

### 4. `crowd_behavior.py` - Collective Behavior Analysis

**Purpose**: Establish crowd flow baselines and detect deviations

**Class**: `CrowdFlowMetrics`
**Attributes**:
- `dominant_direction`: Flow angle in degrees (0-360)
- `average_speed`: Mean velocity magnitude
- `flow_magnitude`: Directional consensus (0-1)
- `density`: Person count
- `activity_level`: Normalized movement intensity (0-1)

**Class**: `CrowdBehaviorAnalyzer`

**Parameters**:
- `history_length=60`: Frames to maintain for baseline

**State**:
- `flow_history`: deque of dominant directions
- `speed_history`: deque of average speeds
- `density_history`: deque of person counts
- `baseline_direction`: Normal flow direction (learned)
- `baseline_speed`: Normal speed (learned)
- `is_baseline_set`: Whether baseline is established

**Methods**:

#### `analyze_frame(tracked_persons, frame_shape) → CrowdFlowMetrics`
1. Extract motion vectors from person trajectories
2. Calculate dominant direction (circular mean)
3. Compute average speed
4. Determine flow magnitude (consensus strength)
5. Count density
6. Normalize activity level

**Returns**: `CrowdFlowMetrics` object

#### `update_baseline()`
Called after history buffer fills:
- Sets `baseline_direction` to median of flow history
- Sets `baseline_speed` to median of speed history
- Enables deviation detection

#### `visualize_flow(frame, metrics) → frame`
Overlays:
- Flow arrow indicating dominant direction
- Density text
- Speed indicator
- Activity level bar

**Use Cases**:
- Detect against-flow movement
- Identify bottlenecks
- Monitor crowd panic (sudden speed changes)
- Evacuation efficiency analysis

---

### 5. `outlier_detector.py` - Behavioral Anomaly Detection

**Purpose**: Detect individuals behaving differently from crowd

**Class**: `BehavioralAnomaly`
**Attributes**:
- `track_id`: Person identifier
- `anomaly_type`: Enum (AGAINST_FLOW, ERRATIC, STATIONARY, RUNNING, GESTURE)
- `confidence`: Score 0-1
- `description`: Human-readable explanation
- `centroid`, `bbox`: Location data

**Class**: `BehavioralOutlierDetector`

**Methods**:

#### `detect_anomalies(tracked_persons, crowd_metrics, pose_data) → List[BehavioralAnomaly]`
Runs all detection algorithms:
1. Against-flow detection
2. Erratic motion detection
3. Stationary detection
4. Running detection
5. Unusual gesture detection

**Returns**: List of detected anomalies

#### `_detect_against_flow(person, crowd_metrics) → Optional[BehavioralAnomaly]`
- Computes person's direction from velocity
- Compares to `dominant_direction` from crowd
- Flags if angle difference > 120° (opposite direction)
- Confidence based on angle magnitude

#### `_detect_erratic_motion(person) → Optional[BehavioralAnomaly]`
- Analyzes trajectory for sudden direction changes
- Computes angle variance between consecutive segments
- High variance = erratic behavior

#### `_detect_stationary(person, crowd_metrics) → Optional[BehavioralAnomaly]`
- Calculates person's average speed
- Compares to crowd's `average_speed`
- Flags if person ≤ 20% of crowd speed while crowd active

#### `_detect_running(person, crowd_metrics) → Optional[BehavioralAnomaly]`
- Identifies abnormally fast movement
- Flags if speed > 3× crowd average
- High confidence if speed > 15 pixels/frame

#### `_detect_unusual_gestures(person, pose_data) → Optional[BehavioralAnomaly]`
- Uses pose analyzer's aggressive pose detection
- Checks for raised arms, fighting stances
- Requires pose keypoints for person

**Thresholds** (tunable):
- Against-flow: 120° deviation
- Stationary: < 20% of crowd speed
- Running: > 3× crowd speed
- Erratic: High trajectory angle variance

---

### 6. `appearance_analyzer.py` - Visual Appearance Analysis

**Purpose**: Detect visually anomalous individuals

**Class**: `AppearanceProfile`
**Attributes**:
- `track_id`: Person identifier
- `color_histogram`: BGR histogram (normalized)
- `dominant_color`: RGB tuple
- `brightness`: Mean intensity (0-255)
- `outlier_confidence`: Anomaly score (0-1)
- `last_updated`: Timestamp

**Class**: `AppearanceAnomaly`
**Attributes**:
- `track_id`, `profile`, `reason`, `confidence`
- Same as behavioral anomaly but for appearance

**Class**: `AppearanceAnalyzer`

**Parameters**:
- `history_length=100`: Profiles to maintain for baseline

**Methods**:

#### `detect_appearance_anomalies(tracked_persons, frame) → List[AppearanceAnomaly]`
1. Extracts appearance profile from bounding box
2. Compares to historical profiles
3. Detects color outliers
4. Checks for unusual brightness
5. Identifies uncommon accessories

**Returns**: List of appearance anomalies

#### `extract_profile(frame, bbox) → AppearanceProfile`
- Crops person from frame
- Computes BGR color histogram (16 bins per channel)
- Calculates dominant color
- Measures brightness

#### `_is_color_outlier(profile) → bool`
- Compares histogram to population mean
- Uses Bhattacharyya distance
- Flags if distance > 2 standard deviations

**Detection Criteria**:
- **Unusual Colors**: All black/all white clothing
- **Accessories**: Detected via color pattern analysis
- **Uniform Detection**: Consistent color blocks

**Use Cases**:
- Security: Identify masked individuals
- Retail: Detect shoplifting (unusual accessories)
- Events: Find lost persons (distinctive clothing)

---

### 7. `anomaly_logger.py` - Event Logging & Alert Management

**Purpose**: Persist anomaly events and manage alerts

**Class**: `AnomalyEvent`
**Attributes**:
- `timestamp`: ISO format datetime
- `frame_number`: Video frame index
- `camera_id`: Camera identifier (future multi-cam)
- `track_id`: Person ID
- `anomaly_types`: List of anomaly names
- `behavioral_confidence`, `appearance_confidence`: Separate scores
- `combined_confidence`: Max or average of both
- `centroid`, `bbox`: Location
- `descriptions`: List of anomaly descriptions
- `metadata`: Additional context (location, weather, etc.)
- `alert_generated`: Boolean flag
- `alert_level`: CRITICAL/HIGH/MEDIUM/LOW
- `snapshot_url`: Path to saved frame image

**Class**: `AnomalyLogger`

**Parameters**:
- `output_dir="anomaly_logs"`: Log directory
- `alert_threshold=0.7`: Minimum confidence for alerts

**State**:
- `events`: In-memory list of all events
- `alert_history`: Dict mapping track_id → events
- `event_log_file`: JSONL file path
- `csv_log_file`: CSV summary path
- `alert_log_file`: Alerts-only JSONL

**Methods**:

#### `log_anomalies(frame_number, camera_id, behavioral_anomalies, appearance_anomalies, snapshot_frame) → List[AnomalyEvent]`
1. Groups anomalies by track_id
2. Computes combined confidence scores
3. Creates `AnomalyEvent` objects
4. Generates alerts if threshold exceeded
5. Saves snapshots for critical alerts
6. Persists to JSONL and CSV files

**Returns**: List of created events

#### `_generate_alert(event) → AnomalyEvent`
- Determines alert level based on confidence:
  - CRITICAL: ≥ 0.9
  - HIGH: ≥ 0.8
  - MEDIUM: ≥ 0.7
  - LOW: < 0.7
- Sets `alert_generated=True`
- Adds contextual metadata

#### `_save_snapshot(frame, frame_number, track_id) → Optional[str]`
- Creates `anomaly_logs/snapshots/` directory
- Saves frame as JPG: `frame{N}_track{ID}.jpg`
- Returns relative URL: `/snapshots/filename.jpg`
- Handles errors gracefully (returns None)

#### `_write_alert_record(event)`
- Writes alert to `alerts.jsonl`
- Includes snapshot URL
- Used by `/alerts` endpoint

#### `get_recent_alerts(level=None, limit=50) → List[Dict]`
- Reads `alerts.jsonl` in reverse
- Filters by alert level if specified
- Returns newest `limit` alerts

#### `get_summary_stats() → Dict`
Returns:
- `total_events`: Count of all anomaly events
- `total_alerts`: Count of threshold-exceeding events
- `affected_tracks`: Number of unique person IDs
- `anomaly_types`: Dict of type → count
- `avg_confidence`: Mean combined confidence

#### `clear_all()`
- Resets in-memory state
- Deletes all log files
- Removes snapshots directory
- Reinitializes CSV headers
- Used by `/reset` endpoint

**File Formats**:

**JSONL** (`anomaly_events.jsonl`):
```json
{"timestamp": "2026-01-26T12:00:00", "frame_number": 123, "track_id": 5, ...}
```

**CSV** (`anomaly_summary.csv`):
```
Timestamp,Frame,Camera,Track_ID,Anomaly_Types,Behavioral_Confidence,Appearance_Confidence,Combined_Confidence,Alert_Level,Description
```

**Alerts JSONL** (`alerts.jsonl`):
```json
{"timestamp": "...", "level": "CRITICAL", "snapshot_url": "/snapshots/...", ...}
```

---

### 8. `llm_reporter.py` - AI Report Generation

**Purpose**: Generate natural language analysis reports using LLMs

**Supported Providers**:
1. **OpenAI** (GPT-3.5-turbo, GPT-4) - Requires API key
2. **Google Gemini** (gemini-pro, gemini-2.5-flash) - Free tier available
3. **Groq** (mixtral-8x7b-32768) - Recommended: fast & generous free tier
4. **Ollama** (local models) - Offline capability

**Class**: `LLMReporter`

**Parameters**:
- `provider`: "openai" | "gemini" | "groq" | "ollama"
- `model`: Optional model override (uses defaults if None)

**Configuration** (via `.env`):
- `LLM_PROVIDER`: Selected provider
- `LLM_MODEL`: Optional model name
- `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`: Authentication

**State**:
- `last_report_cache`: Cached report text
- `last_report_time`: Cache timestamp
- `cache_duration=60`: Cache TTL in seconds

**Methods**:

#### `generate_report(metrics) → str`
1. Checks cache (returns cached if < 60s old)
2. Builds detailed prompt from metrics
3. Queries selected LLM provider
4. Falls back to template report if LLM fails
5. Caches result

**Input Metrics**:
- `total_events`, `total_alerts`, `affected_tracks`
- `avg_confidence`, `anomaly_types` (dict)
- `top_tracks` (list), `alert_timeline` (list)
- `crowd_density`, `timestamp`

**Prompt Template**:
```
Analyze the following crowd anomaly detection report and provide:
1. Overall safety assessment (Safe/Moderate/High Risk)
2. Key findings and patterns
3. Top anomalies and affected persons
4. Recommendations for action
5. Time-based patterns

METRICS DATA:
- Total Events: {total_events}
- Total Alerts: {total_alerts}
...
```

#### `_query_openai(prompt) → Optional[str]`
- Uses OpenAI Chat Completions API
- Temperature: 0.7, Max tokens: 500
- System role: "Security analyst"

#### `_query_gemini(prompt) → Optional[str]`
- Auto-detects available models
- Uses GenerativeModel API
- Handles quota errors gracefully

#### `_query_groq(prompt) → Optional[str]`
- Uses Groq Chat API (OpenAI-compatible)
- Fast inference (5-10x faster than others)
- Best free tier option

#### `_query_ollama(prompt) → Optional[str]`
- Queries local Ollama server (localhost:11434)
- Offline capability
- Requires model download: `ollama pull neural-chat`

#### `_fallback_report(metrics) → str`
Template-based report when LLM unavailable:
- Risk level determination
- Basic statistics
- Top anomaly type
- Generic recommendations

**Example Output**:
```
CROWD ANOMALY DETECTION REPORT
═════════════════════════════════════════

SAFETY ASSESSMENT: HIGH RISK
Multiple high-confidence anomalies detected (15 alerts). Immediate review recommended.

KEY METRICS:
• Total Events: 32
• Total Alerts: 15
• Affected Individuals: 8
• Average Confidence Score: 0.87/1.0

TOP ANOMALY DETECTED: AGAINST_FLOW

RECOMMENDATIONS:
- Immediate security review required
- Check top persons of interest (see dashboard)
- Investigate alert hotspots on timeline

Generated: 2026-01-26T12:00:00
```

**Caching Benefits**:
- Prevents API quota exhaustion
- Reduces cost (OpenAI charges per token)
- Faster repeat requests
- Graceful degradation on quota limits

---

### 9. `algo_factory.py` - Clustering Algorithms

**Purpose**: Group person detections into crowd clusters

**Supported Algorithms**:
1. **DBSCAN** (Density-Based Spatial Clustering)
2. **K-Means** (Centroid-Based Clustering)

**Configuration** (`config.py`):
- `ALGORITHM="DBSCAN"`: Selected algorithm
- `NO_OF_CLUSTERS=10`: K-Means cluster count
- `EPSILON=20.0`: DBSCAN neighborhood radius
- `MIN_SAMPLES=3`: DBSCAN minimum cluster size

**Function**: `getcentroidalgorithm() → ClusteringAlgorithm`
Factory pattern - returns configured algorithm instance

**Methods**:
- `getcentroids(boxes) → List[Box]`: Assigns cluster labels to boxes
- `getnoofcentroids() → int`: Returns cluster count

**Use Case**: Heatmap generation - visualize crowd density regions

---

### 10. `b_box.py` - Bounding Box Data Structure

**Class**: `Box`
**Attributes**:
- `x_left_top`, `y_left_top`: Top-left corner
- `x_right_bottom`, `y_right_bottom`: Bottom-right corner
- `x_centroid`, `y_centroid`: Center point
- `centroid_class`: Cluster ID (from algo_factory)

**Purpose**: Standardized representation of person detection boxes

---

## 🔄 Data Flow

### Complete Processing Pipeline

```
1. VIDEO INPUT
   ↓
2. FRAME EXTRACTION (OpenCV VideoCapture)
   ↓
3. PERSON DETECTION (YOLOv8)
   - Input: RGB frame
   - Output: List of (x1, y1, x2, y2) bounding boxes
   ↓
4. MULTI-OBJECT TRACKING (tracker.py)
   - Input: Bounding boxes
   - Output: List of TrackedPerson with IDs
   ↓
5. FEATURE EXTRACTION (Parallel)
   ├─ POSE ANALYSIS (pose_analyzer.py)
   │  - Input: RGB frame
   │  - Output: 33-point skeleton per person
   │
   ├─ APPEARANCE ANALYSIS (appearance_analyzer.py)
   │  - Input: Frame + bounding boxes
   │  - Output: Color histograms, profiles
   │
   └─ CROWD BEHAVIOR (crowd_behavior.py)
      - Input: All tracked persons
      - Output: CrowdFlowMetrics
   ↓
6. ANOMALY DETECTION (Parallel)
   ├─ BEHAVIORAL (outlier_detector.py)
   │  - Uses: Trajectories, crowd metrics, poses
   │  - Detects: Against-flow, erratic, stationary, running, gestures
   │
   └─ APPEARANCE (appearance_analyzer.py)
      - Uses: Color histograms, profiles
      - Detects: Color outliers, unusual accessories
   ↓
7. ANOMALY LOGGING (anomaly_logger.py)
   - Combines behavioral + appearance anomalies
   - Computes combined confidence
   - Generates alerts if threshold exceeded
   - Saves snapshots for critical alerts
   - Writes to JSONL, CSV, alerts log
   ↓
8. VISUALIZATION (visualize_anomalies)
   - Draws bounding boxes (color-coded by severity)
   - Overlays track IDs, confidence scores
   - Adds anomaly type labels
   - Renders crowd flow vectors
   ↓
9. OUTPUT
   ├─ LIVE STREAM (MJPEG to browser)
   ├─ PROCESSED VIDEO (Saved to outputs/)
   ├─ LOGS (JSONL, CSV in anomaly_logs/)
   ├─ SNAPSHOTS (JPG in anomaly_logs/snapshots/)
   └─ AI REPORT (Generated on demand via LLM)
```

### Data Dependencies

- **Tracking** depends on **Detection**
- **Crowd Behavior** depends on **Tracking** (needs trajectories)
- **Behavioral Anomalies** depend on **Tracking** + **Crowd Behavior**
- **Pose-based Anomalies** depend on **Pose Analysis**
- **Appearance Anomalies** are independent (can run in parallel)
- **Alert Generation** depends on all anomaly types
- **Snapshots** depend on **Alerts**
- **AI Reports** depend on **Logs** (uses historical data)

---

## 🌐 API Endpoints

### HTTP Routes

#### `GET /`
**Description**: Main dashboard page
**Returns**: HTML dashboard with upload form, live stream, KPIs, alerts
**Template**: `templates/index.html`

#### `POST /upload`
**Description**: Upload video file for processing
**Parameters**:
- `video` (multipart/form-data): Video file (MP4, AVI, MOV)
**Validation**: File extension check
**Returns**: Redirect to dashboard with video path
**Side Effects**: Saves to `uploads/`, sets session variables

#### `GET /process/<video_path>`
**Description**: Stream processed video with anomaly overlays
**Parameters**:
- `video_path` (URL param): Filename in uploads folder
**Returns**: MJPEG stream (multipart/x-mixed-replace)
**Content-Type**: `multipart/x-mixed-replace; boundary=frame`
**Frames**: JPEG-encoded with annotations

#### `GET /download`
**Description**: Download processed video file
**Returns**: Video file as attachment
**Filename**: From session variable
**Validation**: Checks session for output path

#### `GET /anomaly_report`
**Description**: Get real-time statistics JSON
**Returns**:
```json
{
  "status": "success",
  "statistics": {
    "total_events": 32,
    "total_alerts": 15,
    "affected_tracks": 8,
    "anomaly_types": {"AGAINST_FLOW": 10, "RUNNING": 5},
    "avg_confidence": 0.85
  },
  "top_tracks": [
    {"track_id": 5, "event_count": 8, "avg_confidence": 0.92},
    ...
  ]
}
```

#### `GET /alerts?level=CRITICAL&limit=50`
**Description**: Fetch recent alerts
**Query Parameters**:
- `level` (optional): Filter by CRITICAL/HIGH/MEDIUM/LOW
- `limit` (optional, default=50): Max alerts to return
**Returns**:
```json
{
  "status": "success",
  "alerts": [
    {
      "timestamp": "2026-01-26T12:00:00",
      "frame": 123,
      "camera": "camera_1",
      "track_id": 5,
      "level": "CRITICAL",
      "confidence": 0.95,
      "types": ["AGAINST_FLOW", "RUNNING"],
      "descriptions": ["Moving opposite to crowd", "Fast movement"],
      "snapshot_url": "/snapshots/frame123_track5.jpg"
    },
    ...
  ]
}
```

#### `POST /reset`
**Description**: Clear all logs, stats, and snapshots
**Returns**:
```json
{
  "status": "success",
  "message": "All logs and stats cleared"
}
```
**Side Effects**:
- Deletes `anomaly_events.jsonl`, `anomaly_summary.csv`, `alerts.jsonl`
- Removes `anomaly_logs/snapshots/` directory
- Resets in-memory event lists

#### `POST /generate_report`
**Description**: Generate AI-powered analysis report
**Returns**:
```json
{
  "status": "success",
  "report": "CROWD ANOMALY DETECTION REPORT\n...",
  "timestamp": "2026-01-26T12:00:00"
}
```
**Process**:
1. Collects stats, top tracks, timeline
2. Enriches with crowd density classification
3. Sends to configured LLM (OpenAI/Gemini/Groq/Ollama)
4. Falls back to template if LLM unavailable
**Caching**: Results cached for 60 seconds

#### `GET /snapshots/<filename>`
**Description**: Serve snapshot image
**Parameters**:
- `filename` (path param): Image filename (e.g., `frame123_track5.jpg`)
**Returns**: JPEG image file
**Directory**: `anomaly_logs/snapshots/`

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# LLM API Keys
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
GROQ_API_KEY=gsk_...

# LLM Provider Selection
LLM_PROVIDER=groq              # openai | gemini | groq | ollama
LLM_MODEL=                     # Optional: override default model

# Flask Server
FLASK_DEBUG=False
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

# File Paths
UPLOAD_FOLDER=uploads
OUTPUT_FOLDER=outputs
MAX_VIDEO_SIZE_MB=1000

# Detection Thresholds
ALERT_THRESHOLD=0.7           # Minimum confidence for alerts
CONFIDENCE_THRESHOLD=0.5      # YOLO detection threshold

# Tracking Parameters
MAX_TRACKING_DISTANCE=50      # Pixels for centroid matching
MAX_FRAMES_TO_SKIP=5          # Frames before dropping track

# Analysis History
POSE_HISTORY_LENGTH=60
APPEARANCE_HISTORY_LENGTH=100
CROWD_HISTORY_LENGTH=60
```

### Configuration Files

#### `config.py`
- `ALGORITHM="DBSCAN"`: Clustering algorithm
- `NO_OF_CLUSTERS=10`: K-Means clusters
- `EPSILON=20.0`: DBSCAN radius
- `MIN_SAMPLES=3`: DBSCAN min points
- `COLORMAP_30`: 30 distinct colors for visualization
- `VISUALIZE_BOXES=False`: Show individual bounding boxes

---

## 🚀 Installation & Setup

### System Requirements

- **OS**: Windows, Linux, or macOS
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: Optional but recommended for faster processing (CUDA-compatible)
- **Storage**: 5GB+ for models and videos

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd Crowd-Density-Estimation
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv crowd
crowd\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv crowd
source crowd/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- Flask 2.3.3 - Web framework
- OpenCV 4.8.1 - Video processing
- ultralytics 8.1.0 - YOLOv8
- torch 2.1.2 - PyTorch (CPU/GPU)
- mediapipe 0.10.8 - Pose estimation
- scikit-learn 1.3.1 - Clustering
- google-generativeai 0.3.1 - Gemini
- groq 0.4.1 - Groq API
- python-dotenv 1.0.0 - Environment variables

### Step 4: Download YOLO Model

```bash
# YOLOv8m will auto-download on first run
# Or manually download:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
```

### Step 5: Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use any text editor
```

### Step 6: Create Directories

```bash
mkdir -p uploads outputs anomaly_logs
```

### Step 7: Run Application

```bash
python main.py
```

**Server starts at**: `http://localhost:5000`

---

## 📖 Usage Guide

### Basic Workflow

1. **Open Dashboard**: Navigate to `http://localhost:5000`
2. **Upload Video**: Click upload area or drag-drop MP4/AVI/MOV file
3. **Processing Starts**: Live stream appears with anomaly overlays
4. **Monitor KPIs**: Watch real-time stats update every 5 seconds
5. **Review Alerts**: Critical alerts table shows high-priority events
6. **Click Alert**: Opens detail drawer with snapshot and metadata
7. **Generate Report**: Click "🤖 Generate AI Report" for analysis
8. **Download**: 
   - Processed video (with overlays)
   - AI report (as TXT file)
9. **Reset**: Click "Reset Stats & Logs" for fresh analysis

### Advanced Features

#### Custom Alert Thresholds

Edit `.env`:
```bash
ALERT_THRESHOLD=0.8  # Higher = fewer alerts (more strict)
```

#### Multi-Camera Support (Future)

```python
# In process_video(), change camera_id per stream
anomaly_logger.log_anomalies(count, "camera_2", ...)
```

#### Export Logs for Analysis

```bash
# Logs are in anomaly_logs/
cat anomaly_logs/anomaly_events.jsonl | jq .  # View JSON
open anomaly_logs/anomaly_summary.csv          # Open in Excel
```

#### Change LLM Provider

Edit `.env`:
```bash
LLM_PROVIDER=openai  # Switch from Groq to OpenAI
```

#### Offline Mode (Ollama)

```bash
# Install Ollama: https://ollama.ai
ollama pull neural-chat

# Edit .env
LLM_PROVIDER=ollama
```

### Troubleshooting

**Issue**: Video not processing
- **Fix**: Check file format (MP4/AVI/MOV only)
- **Fix**: Ensure video file < 1GB

**Issue**: LLM report fails
- **Fix**: Verify API key in `.env`
- **Fix**: Check API quota at provider console
- **Fallback**: System uses template report automatically

**Issue**: Slow processing
- **Fix**: Reduce video resolution (resize before upload)
- **Fix**: Use GPU acceleration (install CUDA)
- **Fix**: Lower YOLO confidence threshold

**Issue**: Snapshots not showing
- **Fix**: Check `anomaly_logs/snapshots/` directory exists
- **Fix**: Ensure write permissions

---

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**: Core language
- **Flask 2.3.3**: Web framework
- **OpenCV 4.8.1**: Video processing and computer vision
- **NumPy 1.24.3**: Numerical computations

### Machine Learning
- **YOLOv8m (ultralytics)**: Person detection
- **MediaPipe 0.10.8**: Pose estimation
- **PyTorch 2.1.2**: Deep learning framework
- **scikit-learn 1.3.1**: Clustering (DBSCAN, K-Means)

### LLM Integration
- **OpenAI API**: GPT-3.5/GPT-4
- **Google Gemini API**: gemini-pro
- **Groq API**: mixtral-8x7b-32768
- **Ollama**: Local models (offline)

### Frontend
- **HTML5**: Dashboard structure
- **Tailwind CSS**: Styling (CDN)
- **JavaScript (Vanilla)**: Interactivity
- **MJPEG Streaming**: Live video display

### Data Persistence
- **JSONL**: Event logs (newline-delimited JSON)
- **CSV**: Tabular summaries
- **JPEG**: Snapshot images

### Development Tools
- **python-dotenv**: Environment configuration
- **Werkzeug**: File upload handling

---

## 📊 Performance Characteristics

### Processing Speed
- **YOLOv8m Inference**: ~30-50ms per frame (GPU) / ~200-300ms (CPU)
- **Tracking**: ~5-10ms per frame
- **Pose Analysis**: ~20-30ms per person
- **Overall**: ~10-15 FPS on GPU, ~2-5 FPS on CPU

### Memory Usage
- **Base**: ~2GB (models loaded)
- **Per Video**: +500MB-2GB (depends on resolution)
- **Logs**: ~1MB per 1000 events

### Storage Requirements
- **YOLOv8m Model**: ~50MB
- **MediaPipe Models**: ~10MB
- **Processed Videos**: ~Same size as input
- **Snapshots**: ~100KB per image

### Scalability
- **Concurrent Videos**: Limited by RAM and CPU/GPU
- **Alert History**: Unlimited (disk-based)
- **Dashboard Updates**: 5-second polling interval

---

## 🔮 Future Enhancements

### Planned Features
1. **Multi-Camera Fusion**: Track persons across camera views
2. **Real-Time Alerts**: WebSocket push notifications
3. **Database Backend**: PostgreSQL/MongoDB for scalability
4. **User Authentication**: Role-based access control
5. **Advanced Analytics**: Heatmaps, trajectory clustering
6. **Export Improvements**: PDF reports, video clips
7. **Mobile App**: iOS/Android monitoring
8. **Edge Deployment**: Run on embedded devices (Jetson Nano)

### Optimization Opportunities
1. **Frame Skipping**: Process every Nth frame for speed
2. **ROI Detection**: Focus on specific areas
3. **Model Quantization**: Faster inference with INT8
4. **Batch Processing**: Queue videos for background processing
5. **Caching**: Redis for frequent queries

---

## 📝 License & Credits

**YOLOv8**: Ultralytics (AGPL-3.0)
**MediaPipe**: Google (Apache 2.0)
**Flask**: Pallets (BSD-3-Clause)

---

## 📞 Support & Contact

For issues, feature requests, or questions:
- Open GitHub issue
- Check documentation files (QUICK_START.md, ANOMALY_DETECTION_GUIDE.md)
- Review logs in `anomaly_logs/`

---

**Document Version**: 1.0
**Last Updated**: January 26, 2026
**Total Modules**: 10 core + 3 supporting
**Lines of Code**: ~3500+
**Supported Video Formats**: MP4, AVI, MOV
**Supported LLMs**: 4 providers (OpenAI, Gemini, Groq, Ollama)
