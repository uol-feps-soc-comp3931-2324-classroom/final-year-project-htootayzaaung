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
from concurrent.futures import ThreadPoolExecutor
from face_blurring import blur_faces 
from detect import detect_objects

models_directory = "models"
current_model = None
model_type = None  # To track the current model type

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

# Generator function to stream frames and data
def generate_frames(camera_index):
    cap = cv2.VideoCapture(camera_index)  # Start camera capture
    prev_frame_time = 0  # For FPS calculation
    new_frame_time = 0

    # Check if the camera opens successfully
    if not cap.isOpened():
        return "Error opening camera", 500  # Error handling

    while cap.isOpened():
        success, frame = cap.read()  # Capture a frame
        if not success:
            break  # Exit if frame capture fails

        # Apply facial blurring before other processing
        blurred_frame = blur_faces(frame)  # New: Apply facial blurring

        # Submit frame processing to the thread pool
        future = executor.submit(detect_objects, blurred_frame, current_model, model_type, camera_index)  # Object detection after blurring
        processed_frame, object_coverage, bbox_dimensions = future.result()  # Get results

        # Calculate FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Encode to JPEG and convert to base64
        _, buffer = cv2.imencode(".jpg", processed_frame)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        # Extract camera dimensions
        frame_height, frame_width, _ = frame.shape
        camera_dimensions = f"{frame_width} × {frame_height}"

        # Send frame, FPS, object coverage, camera dimensions, and bounding box dimensions
        yield f'data: {{"type": "frame", "data": "{frame_base64}"}}\n\n'
        yield f'data: {{"type": "fps", "data": "{fps:.2f}"}}\n\n'
        yield f'data: {{"type": "object_coverage", "data": "{object_coverage:.2f}"}}\n\n'
        yield f'data: {{"type": "camera_dimensions", "data": "{camera_dimensions}"}}\n\n'
        yield f'data: {{"type": "bbox_dimensions", "data": "{bbox_dimensions}"}}\n\n'
        yield f'data: {{"type": "object_count", "data": "{len(bbox_dimensions)}"}}\n\n'

    # Release camera resource
    cap.release()  # Release the camera