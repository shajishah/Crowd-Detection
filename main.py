from flask import Flask, render_template, request, Response, redirect, url_for, send_file, session
import os
import cv2
import numpy as np
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from b_box import Box
from algo_factory import getcentroidalgorithm, COLORMAP_30, VISUALIZE_BOXES

app = Flask(__name__)
app.secret_key = 'your-unique-secret-key-1234567890'  # Set a unique and secret key

# Configure folders
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("yolov8m.pt")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n' + b"Cannot open video file." + b'\r\n')
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Define output path within the function
    output_filename = f"processed_{os.path.basename(input_path)}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

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

        out.write(overlay)

        _, buffer = cv2.imencode('.jpg', overlay)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        count += 1

    cap.release()
    out.release()

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

    # Store output path in session
    session['output_path'] = output_path
    session['filename'] = output_filename

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

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)