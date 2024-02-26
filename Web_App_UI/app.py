import os
import torch
import time
from flask import Flask, render_template, request, Response
from ultralytics import YOLO  # Make sure this import is correct and the library is installed
import cv2
import base64
import mediapipe as mp
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo  # Added import for model_zoo
from flask import jsonify

setup_logger()

app = Flask(__name__)
models_directory = "models"
current_model = None
mediapipe_detector = None

mediapipe_labels = {
    0: 'background', # Note: MediaPipe typically expects class 0 to be background
    1: 'Gun',
    2: 'Axe'
}

DETECTRON2_CLASS_NAMES = ["Axe", "Gun", "Knife"]

def load_model(model_name):
    global current_model, mediapipe_detector
    model_path = os.path.join(models_directory, model_name)
    if model_name.endswith('.pt') and 'yolov' in model_name.lower():
        current_model = YOLO(model_path)
        print("Loaded YOLO model:", model_path)
        mediapipe_detector = None
    elif model_name.endswith('.tflite'):
        mediapipe_detector = mp.solutions.object_detection.ObjectDetection(model_path=model_path)
        print("Loaded Mediapipe model:", model_path)
        current_model = None
    elif model_name.endswith('.pth'):  # Adjusted for Detectron2 model
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # Example configuration
        cfg.MODEL.WEIGHTS = model_path  # Model weights
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Adjust based on your model
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Minimum detection threshold
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        current_model = DefaultPredictor(cfg)
        print("Loaded Detectron2 model:", model_path)
    else:
        print("Unsupported model format:", model_path)

def detect_objects(frame):
    if isinstance(current_model, DefaultPredictor):  # Check if current_model is a Detectron2 model
        outputs = current_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        instances = outputs["instances"].to("cpu")
        if instances.has("pred_boxes"):
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            for box, score, class_idx in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box.astype(int)
                # Use the mapping to get the actual class name
                label = f"{DETECTRON2_CLASS_NAMES[class_idx]}: {score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    elif current_model:  # YOLOv8 model handling
        results = current_model(frame, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                confidence = box.conf[0]
                if confidence >= 0.6:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = current_model.names[int(box.cls[0])]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(frame, f'{label}: {confidence:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
    elif mediapipe_detector:  # MediaPipe model handling
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mediapipe_detector.process(frame_rgb)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = bboxC.xmin * iw, bboxC.ymin * ih, bboxC.width * iw, bboxC.height * ih
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)
                # Assuming the category ID is directly available; adjust if necessary
                label = mediapipe_labels.get(detection.label_id[0], 'Unknown')
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame

def generate_frames():
    cap = cv2.VideoCapture(0)
    prev_frame_time = 0
    new_frame_time = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        new_frame_time = time.time()

        # Perform object detection based on the selected model
        if current_model or mediapipe_detector:
            frame = detect_objects(frame)

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        yield f'data: {{ "type": "frame", "data": "{frame_base64}" }}\n\n'  # Video frame event
        yield f'data: {{ "type": "fps", "data": "{fps:.2f}" }}\n\n'  # FPS event

@app.route('/')
def index():
    model_files = []
    for root, dirs, files in os.walk(models_directory):
        for file in files:
            if file.endswith('.pt') or file.endswith('.tflite') or file.endswith('.pth'):
                model_files.append(os.path.relpath(os.path.join(root, file), models_directory))
    return render_template('index.html', model_files=model_files)

@app.route('/load_model', methods=['POST'])
def handle_load_model():
    model_name = request.form.get('model_name')
    load_model(model_name)
    return "Model loaded successfully", 200

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)