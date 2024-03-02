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
import numpy as np
import random

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
    global current_model, mediapipe_detector, model_type  # Declare as global
    model_path = os.path.join(models_directory, model_name)
    if model_name.endswith('.pt') and 'yolov' in model_name.lower():
        if 'segmentation' in model_name.lower():
            model_type = 'yolo_segmentation'
        else:
            model_type = 'yolo_detection'
        current_model = YOLO(model_path)  # Ensure YOLO is correctly imported
        print(f"Loaded YOLO model: {model_path}, Type: {model_type}")
        mediapipe_detector = None
    elif model_name.endswith('.tflite'):
        mediapipe_detector = mp.solutions.object_detection.ObjectDetection(model_path=model_path)
        print("Loaded Mediapipe model:", model_path)
        current_model = None
        model_type = 'mediapipe'
    elif model_name.endswith('.pth'):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        current_model = DefaultPredictor(cfg)
        model_type = 'detectron2'
        print("Loaded Detectron2 model:", model_path)
    else:
        print("Unsupported model format:", model_path)
        model_type = None
        current_model = None
        
def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    
    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    # color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined
            

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

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

    if model_type == 'yolo_detection':
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

    elif model_type == 'yolo_segmentation':
        results = current_model(frame, stream=True)
        class_names = current_model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]
        for r in results:
            boxes = r.boxes
            masks = r.masks
            if masks is not None:
                masks = masks.data.cpu()
                for seg, box in zip(masks.data.cpu().numpy(), boxes):
                    color = colors[int(box.cls)]  # You'll need to generate colors for classes
                    seg = cv2.resize(seg, (frame.shape[1], frame.shape[0]))
                    frame = overlay(frame, seg, color, 0.4)
                    xmin = int(box.data[0][0])
                    ymin = int(box.data[0][1])
                    xmax = int(box.data[0][2])
                    ymax = int(box.data[0][3])
                    plot_one_box([xmin, ymin, xmax, ymax], frame, color, f'{class_names[int(box.cls)]} {float(box.conf):.2f}')
        return frame
                
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