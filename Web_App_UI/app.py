import os
from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import cv2
import base64
import mediapipe as mp

app = Flask(__name__)
models_directory = "models"
current_model = None
mediapipe_detector = None

mediapipe_labels = {
    0: 'background', # Note: MediaPipe typically expects class 0 to be background
    1: 'Gun',
    2: 'Axe'
}

def load_model(model_name):
    global current_model, mediapipe_detector
    model_path = os.path.join(models_directory, model_name)
    if model_name.endswith('.pt'):
        current_model = YOLO(model_path)
        print("Loaded YOLO model:", model_path)
        mediapipe_detector = None
    elif model_name.endswith('.tflite'):
        # Initialize Mediapipe object detection
        mediapipe_detector = mp.solutions.object_detection.ObjectDetection(model_name=model_path)
        #mediapipe_detector_index = current_model.names
        print("Loaded Mediapipe model:", model_path)
        current_model = None
    else:
        print("Unsupported model format:", model_path)

def load_model(model_name):
    global current_model, mediapipe_detector
    model_path = os.path.join(models_directory, model_name)
    if model_name.endswith('.pt'):
        current_model = YOLO(model_path)
        print("Loaded YOLO model:", model_path)
        mediapipe_detector = None
    elif model_name.endswith('.tflite'):
        #mediapipe_detector = mp.solutions.object_detection.ObjectDetection(model_path=model_path)
        print("Loaded Mediapipe model:", model_path)
        current_model = None
    else:
        print("Unsupported model format:", model_path)

def detect_objects(frame):
    if current_model:  # YOLOv8 model handling
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
            mp_drawing = mp.solutions.drawing_utils 
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3) 

                label = mediapipe_labels.get(detection.label_id[0], 'Unknown')
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame


def generate_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        # Perform object detection based on the selected model
        if current_model or mediapipe_detector:
            frame = detect_objects(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        yield (b'data: ' + frame_base64.encode() + b'\n\n')


@app.route('/')
def index():
    model_files = []
    for root, dirs, files in os.walk(models_directory):
        for file in files:
            if file.endswith('.pt') or file.endswith('.tflite'):
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
