from flask import Flask, render_template, request, Response, redirect, url_for, send_file, session, jsonify, send_from_directory
import os
import cv2
import numpy as np
import json
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from b_box import Box
from algo_factory import getcentroidalgorithm, COLORMAP_30, VISUALIZE_BOXES
from tracker import MultiObjectTracker
from pose_analyzer import PoseAnalyzer
from crowd_behavior import CrowdBehaviorAnalyzer
from outlier_detector import BehavioralOutlierDetector
from appearance_analyzer import AppearanceAnalyzer
from anomaly_logger import AnomalyLogger
from llm_reporter import LLMReporter
from datetime import datetime
from outlier_detector import BehavioralAnomaly, AnomalyType

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Read LLM configuration from .env
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "")
llm_reporter = LLMReporter(provider=LLM_PROVIDER, model=LLM_MODEL if LLM_MODEL else None)

app.secret_key = 'your-unique-secret-key-1234567890'  # Set a unique and secret key

# Global dict to track video processing progress
processing_progress = {}

# Configure folders from .env or defaults
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "outputs")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("yolov8m.pt")

# Read analysis parameters from .env
MAX_TRACKING_DISTANCE = int(os.getenv("MAX_TRACKING_DISTANCE", "50"))
MAX_FRAMES_TO_SKIP = int(os.getenv("MAX_FRAMES_TO_SKIP", "5"))
POSE_HISTORY_LENGTH = int(os.getenv("POSE_HISTORY_LENGTH", "60"))
APPEARANCE_HISTORY_LENGTH = int(os.getenv("APPEARANCE_HISTORY_LENGTH", "100"))
CROWD_HISTORY_LENGTH = int(os.getenv("CROWD_HISTORY_LENGTH", "60"))
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.7"))

# Initialize analysis modules
tracker = MultiObjectTracker(max_distance=MAX_TRACKING_DISTANCE, max_frames_to_skip=MAX_FRAMES_TO_SKIP)
pose_analyzer = PoseAnalyzer()
crowd_analyzer = CrowdBehaviorAnalyzer(history_length=CROWD_HISTORY_LENGTH)
behavioral_detector = BehavioralOutlierDetector()
appearance_detector = AppearanceAnalyzer(history_length=APPEARANCE_HISTORY_LENGTH)
anomaly_logger = AnomalyLogger(output_dir="anomaly_logs", alert_threshold=ALERT_THRESHOLD)
appearance_detector = AppearanceAnalyzer(history_length=100)
anomaly_logger = AnomalyLogger(output_dir="anomaly_logs", alert_threshold=0.7)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def visualize_anomalies(frame, tracked_persons, anomaly_events, crowd_metrics):
    """
    Visualize tracked persons and detected anomalies on frame
    
    Shows:
    - Tracked person IDs
    - Colored bounding boxes (red=critical, orange=high, yellow=medium)
    - Confidence scores
    - Anomaly types
    - Crowd flow visualization
    """
    
    # Draw tracked persons
    for person in tracked_persons:
        x1, y1, x2, y2 = person.bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Check if this person has detected anomalies
        event = next((e for e in anomaly_events if e.track_id == person.track_id), None)
        
        if event:
            # Color code by alert level
            if event.alert_level == "CRITICAL":
                color = (0, 0, 255)  # Red for critical
                thickness = 3
            elif event.alert_level == "HIGH":
                color = (0, 165, 255)  # Orange for high
                thickness = 2
            else:
                color = (0, 255, 255)  # Yellow for medium
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw track ID and confidence score
            conf_text = f"ID:{person.track_id} [{event.combined_confidence:.2f}]"
            cv2.putText(frame, conf_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw anomaly types (up to 2)
            for i, atype in enumerate(event.anomaly_types[:2]):
                cv2.putText(frame, atype, (x1, y2 + 15 + i*18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        else:
            # Normal tracking - green box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, f"ID:{person.track_id}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Visualize crowd flow metrics
    if crowd_metrics:
        frame = crowd_analyzer.visualize_flow(frame, crowd_metrics)
    
    return frame

def detect_crowd_anomalies(crowd_metrics, tracked_persons, frame_shape):
    """
    Detect crowd-level anomalies based on density and flow patterns
    
    Returns List[BehavioralAnomaly] to match the format expected by anomaly_logger
    """
    anomalies = []
    
    if not crowd_metrics:
        return anomalies
    
    # Calculate crowd density
    total_people = len(tracked_persons)
    frame_area = frame_shape[0] * frame_shape[1]  # height * width
    density_per_pixel = total_people / frame_area if frame_area > 0 else 0
    
    # Thresholds (adjust based on your use case)
    HIGH_DENSITY_THRESHOLD = 0.00005  # ~1 person per 20,000 pixels
    CRITICAL_DENSITY_THRESHOLD = 0.0001  # ~1 person per 10,000 pixels
    
    # Check for high density (assign to all tracked persons in dense areas)
    if density_per_pixel > CRITICAL_DENSITY_THRESHOLD:
        for person in tracked_persons:
            anomalies.append(BehavioralAnomaly(
                track_id=person.track_id,
                anomaly_type=AnomalyType.CRITICAL_OVERCROWDING,
                confidence=min(0.95, density_per_pixel / CRITICAL_DENSITY_THRESHOLD),
                description=f'{total_people} people in frame (critical density)',
                centroid=((person.bbox[0] + person.bbox[2]) / 2, (person.bbox[1] + person.bbox[3]) / 2),
                bbox=person.bbox
            ))
    elif density_per_pixel > HIGH_DENSITY_THRESHOLD:
        for person in tracked_persons:
            anomalies.append(BehavioralAnomaly(
                track_id=person.track_id,
                anomaly_type=AnomalyType.HIGH_CROWD_DENSITY,
                confidence=min(0.85, density_per_pixel / HIGH_DENSITY_THRESHOLD),
                description=f'{total_people} people in frame (high density)',
                centroid=((person.bbox[0] + person.bbox[2]) / 2, (person.bbox[1] + person.bbox[3]) / 2),
                bbox=person.bbox
            ))
    
    # Check for unusual flow patterns
    if hasattr(crowd_metrics, 'avg_flow_magnitude'):
        if crowd_metrics.avg_flow_magnitude > 15.0:  # Rapid movement
            for person in tracked_persons:
                # Check if this person already has an anomaly from density check
                has_crowd_anomaly = any(a.track_id == person.track_id for a in anomalies)
                if not has_crowd_anomaly:
                    anomalies.append(BehavioralAnomaly(
                        track_id=person.track_id,
                        anomaly_type=AnomalyType.PANIC_FLOW,
                        confidence=0.75,
                        description=f'Rapid crowd movement (flow: {crowd_metrics.avg_flow_magnitude:.1f})',
                        centroid=((person.bbox[0] + person.bbox[2]) / 2, (person.bbox[1] + person.bbox[3]) / 2),
                        bbox=person.bbox
                    ))
    
    # Check for crowd congestion (low movement in high density)
    if total_people > 30 and hasattr(crowd_metrics, 'avg_flow_magnitude'):
        if crowd_metrics.avg_flow_magnitude < 2.0:  # Very slow movement
            for person in tracked_persons:
                # Check if this person already has an anomaly
                has_anomaly = any(a.track_id == person.track_id for a in anomalies)
                if not has_anomaly:
                    anomalies.append(BehavioralAnomaly(
                        track_id=person.track_id,
                        anomaly_type=AnomalyType.CROWD_CONGESTION,
                        confidence=0.70,
                        description=f'{total_people} people with minimal movement',
                        centroid=((person.bbox[0] + person.bbox[2]) / 2, (person.bbox[1] + person.bbox[3]) / 2),
                        bbox=person.bbox
                    ))
    
    return anomalies

def process_video(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n' + b"Cannot open video file." + b'\r\n')
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define output path within the function
    output_filename = f"processed_{os.path.basename(input_path)}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    # Initialize progress tracking
    video_id = os.path.basename(input_path)
    processing_progress[video_id] = {'current': 0, 'total': total_frames, 'percentage': 0}

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.10)
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        if count > 0:
            heatmap *= 0.9

        list_of_boxes = []
        for x1, y1, x2, y2 in results[0].boxes.xyxy:
            centroid_x = int((x1 + x2) / 2)
            centroid_y = int((y1 + y2) / 2)
            bbox = Box(int(x1), int(y1), int(x2), int(y2), centroid_x, centroid_y, 0)
            list_of_boxes.append(bbox)

        centroid_algorithm = getcentroidalgorithm()
        centroid_and_boxes_list = centroid_algorithm.getcentroids(list_of_boxes)

        if VISUALIZE_BOXES:
            for box in centroid_and_boxes_list:
                color = (0, 0, 0) if box.centroid_class == -1 else COLORMAP_30[box.centroid_class % len(COLORMAP_30)]
                cv2.circle(frame, (box.x_centroid, box.y_centroid), 4, color, -1)

        centroid_aggregator = {}
        for i in range(centroid_algorithm.getnoofcentroids()):
            centroid_aggregator[i] = {
                "min_x": 5000, "min_y": 5000, "max_x": -5000, "max_y": -5000,
                "centroid_x": -1, "centroid_y": -1, "count": 0, "label": i
            }

        for bbox in centroid_and_boxes_list:
            if bbox.centroid_class != -1:
                centroid_aggregator[bbox.centroid_class]["count"] += 1
                centroid_aggregator[bbox.centroid_class]["min_x"] = min(
                    centroid_aggregator[bbox.centroid_class]["min_x"], bbox.x_left_top)
                centroid_aggregator[bbox.centroid_class]["min_y"] = min(
                    centroid_aggregator[bbox.centroid_class]["min_y"], bbox.y_left_top)
                centroid_aggregator[bbox.centroid_class]["max_x"] = max(
                    centroid_aggregator[bbox.centroid_class]["max_x"], bbox.x_right_bottom)
                centroid_aggregator[bbox.centroid_class]["max_y"] = max(
                    centroid_aggregator[bbox.centroid_class]["max_y"], bbox.y_right_bottom)
                centroid_aggregator[bbox.centroid_class]["centroid_x"] = bbox.x_centroid
                centroid_aggregator[bbox.centroid_class]["centroid_y"] = bbox.y_centroid

        for label, values in centroid_aggregator.items():
            xlefttop = int(values["min_x"])
            ylefttop = int(values["min_y"])
            xrightbottom = int(values["max_x"])
            yrightbottom = int(values["max_y"])
            x_centroid = int(values["centroid_x"])
            y_centroid = int(values["centroid_y"])
            radius_x = max(x_centroid - xlefttop, xrightbottom - x_centroid)
            radius_y = max(y_centroid - ylefttop, yrightbottom - y_centroid)
            radius = max(radius_x, radius_y)
            intensity = min(1.0, values["count"] / 10.0)
            if xlefttop != 5000:
                cv2.circle(heatmap, (x_centroid, y_centroid), radius, intensity, -1)

        heatmap = cv2.GaussianBlur(heatmap, (91, 91), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        colored = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_TURBO)
        overlay = cv2.addWeighted(frame, 0.5, colored, 0.5, 0)

        # ===== Advanced Anomaly Detection Pipeline =====
        
        # 1. Track individuals with persistent IDs
        detections = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in results[0].boxes.xyxy]
        tracked_persons = tracker.update(detections)
        
        # 2. Analyze crowd behavior
        crowd_metrics = crowd_analyzer.analyze_frame(tracked_persons, frame.shape)
        
        # 3. Analyze poses and movements (Optimized: Crop-based)
        pose_data = {}
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        for person in tracked_persons:
            # Create crop for this person
            x1, y1, x2, y2 = person.bbox
            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
            
            # Skip invalid crops
            if x2 <= x1 or y2 <= y1:
                continue
                
            person_crop = frame_rgb[y1:y2, x1:x2]
            
            # Analyze pose on crop
            pose = pose_analyzer.analyze_frame(person_crop, offset=(x1, y1))
            
            if pose:
                pose_analyzer.update_trajectory(person.track_id, pose.keypoints)
                pose_data[person.track_id] = pose
        
        # 4. Detect behavioral anomalies (against flow, erratic, stationary, running, gestures)
        behavioral_anomalies = behavioral_detector.detect_anomalies(
            tracked_persons, crowd_metrics, pose_data
        )
        
        # 5. Detect crowd density anomalies (overcrowding, high density, unusual flow)
        crowd_anomalies = detect_crowd_anomalies(crowd_metrics, tracked_persons, frame.shape)
        
        # 6. Merge behavioral and crowd anomalies (both are lists now)
        all_anomalies = behavioral_anomalies + crowd_anomalies
        
        # 7. Detect appearance anomalies (unusual clothing, colors, accessories)
        # Throttling: only run every 30 frames per person to save performance
        appearance_anomalies = []
        if count % 30 == 0:
             appearance_anomalies = appearance_detector.detect_appearance_anomalies(
                tracked_persons, frame
            )
        else:
            # Keep previous anomalies valid? 
            # Ideally we'd cache them, but for now just running periodically is a safe start
            # To avoid flickering, we might want to store them in a persistent state manager,
            # but let's start with periodic checks.
            pass
        
        # 8. Log anomalies and generate alerts (pass frame for snapshot saving)
        # 8. Log anomalies and generate alerts (pass frame for snapshot saving)
        anomaly_events = anomaly_logger.log_anomalies(
            count, "camera_1", 
            behavioral_anomalies=all_anomalies, 
            appearance_anomalies=appearance_anomalies,
            snapshot_frame=overlay, 
            tracked_persons=tracked_persons
        )
        
        # 7. Visualize results with bounding boxes and confidence scores
        overlay = visualize_anomalies(
            overlay, tracked_persons, anomaly_events, crowd_metrics
        )

        out.write(overlay)

        _, buffer = cv2.imencode('.jpg', overlay)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        count += 1
        
        # Update progress
        if total_frames > 0:
            processing_progress[video_id] = {
                'current': count,
                'total': total_frames,
                'percentage': int((count / total_frames) * 100)
            }

    cap.release()
    out.release()
    
    # Mark as complete
    if video_id in processing_progress:
        processing_progress[video_id]['percentage'] = 100

@app.route('/')
def index():
    return render_template('index.html', video_path=None)

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(url_for('index'))

    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)

    output_filename = f"processed_{filename}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    # Clear old output file if it exists to prevent false completion detection
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Clear old progress data for this video
    if filename in processing_progress:
        del processing_progress[filename]

    # Store output path and input filename in session for progress tracking
    session['output_path'] = output_path
    session['filename'] = output_filename
    session['input_filename'] = filename  # Store input filename for progress tracking

    return render_template('index.html', video_path=filename)

@app.route('/process/<video_path>')
def process(video_path):
    video_full_path = os.path.join(app.config['UPLOAD_FOLDER'], video_path)
    return Response(process_video(video_full_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download')
def download_video():
    output_path = session.get('output_path')
    filename = session.get('filename')

    if not output_path or not os.path.exists(output_path):
        return redirect(url_for('index'))

    return send_file(output_path, as_attachment=True, download_name=filename)


@app.route('/processing_status')
def processing_status():
    """Check if video processing is complete"""
    output_path = session.get('output_path')
    if not output_path:
        return jsonify({'status': 'idle', 'ready': False})
    
    # Check progress data instead of just file existence
    # (VideoWriter creates the file immediately, so it exists even at 0%)
    video_id = session.get('input_filename', '')
    
    if video_id in processing_progress:
        progress_data = processing_progress[video_id]
        is_complete = progress_data.get('percentage', 0) >= 100
        return jsonify({
            'status': 'complete' if is_complete else 'processing',
            'ready': is_complete
        })
    
    # If no progress data and file exists, it might be from a previous run
    # Check file size to see if it's actually complete (not just created)
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        # If file is larger than 1MB, it's probably complete from previous run
        is_ready = file_size > 1024 * 1024
        return jsonify({
            'status': 'complete' if is_ready else 'processing',
            'ready': is_ready
        })
    
    return jsonify({'status': 'processing', 'ready': False})


@app.route('/processing_progress')
def processing_progress_status():
    """Get current video processing progress"""
    video_path = session.get('output_path')
    if not video_path:
        return jsonify({'progress': 0, 'status': 'idle'})
    
    # Get the input filename from session (this matches the video_id used in process_video)
    video_id = session.get('input_filename', '')
    if not video_id:
        return jsonify({'progress': 0, 'status': 'idle'})
    
    if video_id in processing_progress:
        progress_data = processing_progress[video_id]
        return jsonify({
            'progress': progress_data.get('percentage', 0),
            'current_frame': progress_data.get('current', 0),
            'total_frames': progress_data.get('total', 0),
            'status': 'processing'
        })
    
    # Check if already complete
    if os.path.exists(video_path):
        return jsonify({'progress': 100, 'status': 'complete'})
    
    return jsonify({'progress': 0, 'status': 'starting'})


@app.route('/anomaly_report')
def anomaly_report():
    """Generate and return anomaly analysis report"""
    try:
        stats = anomaly_logger.get_summary_stats()
        return {
            'status': 'success',
            'statistics': stats,
            'top_tracks': anomaly_logger._get_top_anomaly_tracks(10)
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


@app.route('/alerts')
def alerts():
    """Return recent alerts (defaults to CRITICAL/HIGH if level param provided)."""
    level = request.args.get('level')
    limit = request.args.get('limit', default=50, type=int)
    try:
        alerts = anomaly_logger.get_recent_alerts(level=level, limit=limit)
        return jsonify({'status': 'success', 'alerts': alerts})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/alerts_by_type')
def alerts_by_type():
    """Return alerts filtered by anomaly type."""
    anomaly_type = request.args.get('type')
    limit = request.args.get('limit', default=100, type=int)
    
    if not anomaly_type:
        return jsonify({'status': 'error', 'message': 'Anomaly type parameter required'}), 400
    
    try:
        # Filter events by anomaly type
        filtered_alerts = [
            event for event in anomaly_logger.events
            if anomaly_type in event.anomaly_types
        ]
        
        # Sort by timestamp (newest first) and limit
        filtered_alerts.sort(key=lambda x: x.timestamp, reverse=True)
        filtered_alerts = filtered_alerts[:limit]
        
        # Convert to dict format
        alerts_data = [
            {
                'timestamp': event.timestamp,
                'frame': event.frame_number,
                'frame_number': event.frame_number,
                'camera': event.camera_id,
                'camera_id': event.camera_id,
                'track_id': event.track_id,
                'types': event.anomaly_types,
                'type': event.anomaly_types,
                'confidence': event.combined_confidence,
                'combined_confidence': event.combined_confidence,
                'level': event.alert_level,
                'descriptions': event.descriptions,
                'snapshot_url': event.snapshot_url
            }
            for event in filtered_alerts
        ]
        
        return jsonify({
            'status': 'success',
            'anomaly_type': anomaly_type,
            'count': len(alerts_data),
            'alerts': alerts_data
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/snapshots/<path:filename>')
def snapshots(filename):
    """Serve saved alert snapshots."""
    snaps_dir = os.path.join(str(anomaly_logger.output_dir), "snapshots")
    return send_from_directory(snaps_dir, filename)


@app.route('/reset', methods=['POST'])
def reset():
    """Reset all anomaly logs and state for fresh analysis."""
    try:
        anomaly_logger.clear_all()
        return jsonify({'status': 'success', 'message': 'All logs and stats cleared'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generate LLM-powered conclusion report from metrics"""
    try:
        stats = anomaly_logger.get_summary_stats()
        top_tracks = anomaly_logger._get_top_anomaly_tracks(10)
        alert_timeline = anomaly_logger._generate_timeline()
        
        # Enrich metrics for LLM
        metrics = {
            **stats,
            'top_tracks': top_tracks,
            'alert_timeline': alert_timeline,
            'timestamp': datetime.now().isoformat(),
            'crowd_density': 'HIGH' if stats.get('total_events', 0) > 50 else 'MEDIUM' if stats.get('total_events', 0) > 20 else 'LOW'
        }
        
        report = llm_reporter.generate_report(metrics)
        if not report:
            report = llm_reporter._fallback_report(metrics)
        
        return jsonify({
            'status': 'success',
            'report': report,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

 
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
