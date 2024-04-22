import cv2
import base64
import time
from object_detection import detect_objects
from utility_functions import plot_one_box, overlay

def generate_frames(camera_index):
    cap = cv2.VideoCapture(camera_index)
    prev_frame_time = 0
    new_frame_time = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        new_frame_time = time.time()

        # Object detection with the current model
        if current_model:
            frame = detect_objects(frame)

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        _, buffer = cv2.imencode(".jpg", frame)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")
        yield f'data: {{"type": "frame", "data": "{frame_base64}"}}\n\n'
        yield f'data: {{"type": "fps", "data": "{fps:.2f}"}}\n\n'

