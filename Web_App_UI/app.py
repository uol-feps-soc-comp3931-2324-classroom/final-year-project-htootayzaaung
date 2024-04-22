import os
import time
from flask import Flask, render_template, request, Response
import base64
import cv2
from ultralytics import YOLO  # Make sure this import is correct and the library is installed
from detectron2.utils.logger import setup_logger
from object_detection import load_model, generate_frames  # Import from new module

setup_logger()

app = Flask(__name__)
models_directory = "models"

@app.route('/')
def index():
    model_files = []
    for root, dirs, files in os.walk(models_directory):
        for file in files:
            if file.endswith('.pt') or file.endswith('.pth'):
                model_files.append(os.path.relpath(os.path.join(root, file), models_directory))
    return render_template('index.html', model_files=model_files)

@app.route('/load_model', methods=['POST'])
def handle_load_model():
    model_name = request.form.get('model_name')
    load_model(model_name)  # Use the function from the new module
    return "Model loaded successfully", 200

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='text/event-stream')  # Use the generator from the new module

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
