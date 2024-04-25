import os
import torch
import cv2
import base64
import numpy as np
import random
import time
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from ultralytics import YOLO
from utility_functions import overlay, plot_one_box
from concurrent.futures import ThreadPoolExecutor
from face_blurring import blur_faces 

models_directory = "models"
current_model = None
model_type = None  # To track the current model type

DETECTRON2_CLASS_NAMES = ["Axe", "Gun", "Knife"]

executor = ThreadPoolExecutor(max_workers=8)  # Adjust as needed

def load_model(model_name):
    global current_model, model_type
    model_path = os.path.join(models_directory, model_name)

    if model_name.endswith('.pth') and 'detectron' in model_name.lower():
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        cfg.MODEL.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        current_model = DefaultPredictor(cfg)
        model_type = 'detectron2'
        print(f"Loaded Detectron2 model: {model_path}")

    elif model_name.endswith('.pt') and 'yolov' in model_name.lower():
        if 'segmentation' in model_name.lower():
            model_type = 'yolo_segmentation'
        else:
            model_type = 'yolo_detection'
        current_model = YOLO(model_path)
        print(f"Loaded YOLO model: {model_path}")

    else:
        print(f"Unsupported model format: {model_path}")
        current_model = None
        model_type = None

def unload_model():
    global current_model, model_type
    current_model = None
    model_type = None
    print("Model has been unloaded.")

def detect_objects(frame):
    total_box_area = 0  # Variable to track total bounding box area
    frame_height, frame_width, _ = frame.shape
    total_camera_area = frame_width * frame_height  # Total frame area

    bbox_dimensions = []  # New: List to store bounding box dimensions

    if current_model is None:
        return frame, 0, []  # Return original frame, 0% coverage, and empty list for bounding box dimensions

    # Ensure non-negative bounding box areas
    def correct_coordinates(x1, x2, y1, y2):
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        return x1, x2, y1, y2

    if isinstance(current_model, DefaultPredictor):
        outputs = current_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        instances = outputs["instances"].to("cpu")
        if instances.has("pred_boxes"):
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            
            for box, score, class_idx in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box.astype(int)
                x1, x2, y1, y2 = correct_coordinates(x1, x2, y1, y2)  # Ensure correct coordinates
                box_area = (x2 - x1) * (y2 - y1)  # Calculate bounding box area
                total_box_area += max(0, box_area)  # Ensure non-negative box areas
                
                # New: Add bounding box dimensions
                bbox_dimensions.append(f"{x2 - x1} × {y2 - y1}")
                
                label = f"{DETECTRON2_CLASS_NAMES[class_idx]}: {score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw box
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Draw label

    elif model_type == 'yolo_detection':
        results = current_model(frame, stream=True)
        for r in results:
            for box in r.boxes:
                confidence = box.conf[0]
                if confidence >= 0.6:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, x2, y1, y2 = correct_coordinates(x1, x2, y1, y2)  # Ensure correct coordinates
                    box_area = (x2 - x1) * (y2 - y1)  # Calculate bounding box area
                    total_box_area += max(0, box_area)  # Ensure non-negative box areas
                    
                    # New: Add bounding box dimensions
                    bbox_dimensions.append(f"{x2 - x1} × {y2 - y1}")
                    
                    if current_model is not None:
                        label = current_model.names[int(box.cls[0])]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)  # Draw box
                        cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # Draw label

    elif model_type == 'yolo_segmentation':
        results = current_model(frame, stream=True)
        class_names = current_model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]
        
        for r in results:
            boxes = r.boxes
            masks = r.masks
            
            if masks is not None:
                masks = masks.data.cpu()  # Make sure it's on CPU
                for seg, box in zip(masks.data.cpu().numpy(), boxes):
                    color = colors[int(box.cls)]
                    seg = cv2.resize(seg, (frame.shape[1], frame.shape[0]))
                    frame = overlay(frame, seg, color, 0.4)

                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                    xmin, xmax, ymin, ymax = correct_coordinates(xmin, xmax, ymin, ymax)  # Correct coordinates
                    box_area = (xmax - xmin) * (ymax - ymin)  # Calculate bounding box area
                    total_box_area += max(0, box_area)  # Ensure non-negative box areas
                    
                    # New: Add bounding box dimensions
                    bbox_dimensions.append(f"{xmax - xmin} × {ymax - ymin}")
                    
                    plot_one_box([xmin, ymin, xmax, ymax], frame, color, f"{class_names[int(box.cls)]} {float(box.conf):.2f}")

    # Calculate object coverage percentage
    object_coverage = (total_box_area / total_camera_area) * 100 if total_camera_area > 0 else 0  # Ensure positive coverage
    return frame, object_coverage, bbox_dimensions  # Return frame, object coverage, and bounding box dimensions


def generate_frames(camera_index):
    cap = cv2.VideoCapture(camera_index)  # Start capturing from the camera
    prev_frame_time = 0  # For calculating FPS
    new_frame_time = 0

    # Check if the camera opened successfully
    if not cap.isOpened():
        return "Error opening camera", 500  # Handle camera open errors

    while cap.isOpened():
        success, frame = cap.read()  # Capture a frame
        if not success:
            break  # Stop if frame capture fails

        # Submit frame processing to the thread pool
        future = executor.submit(detect_objects, frame)  # Object detection processing
        processed_frame, object_coverage, bbox_dimensions = future.result()  # Get processed data

        # Calculate FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Encode processed frame to JPEG and convert to base64
        _, buffer = cv2.imencode(".jpg", processed_frame)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        # Extract camera dimensions
        frame_height, frame_width, _ = frame.shape  # Camera feed dimensions
        camera_dimensions = f"{frame_width} × {frame_height}"

        # Send frame, FPS, object coverage, camera dimensions, and bounding box dimensions
        yield f'data: {{"type": "frame", "data": "{frame_base64}"}}\n\n'
        yield f'data: {{"type": "fps", "data": "{fps:.2f}"}}\n\n'
        yield f'data: {{"type": "object_coverage", "data": "{object_coverage:.2f}"}}\n\n'
        yield f'data: {{"type": "camera_dimensions", "data": "{camera_dimensions}"}}\n\n'
        yield f'data: {{"type": "bbox_dimensions", "data": "{bbox_dimensions}"}}\n\n'  # New addition

    # Release the camera resource
    cap.release()