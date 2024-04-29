import os
import time
from flask import Flask, render_template, request, Response, jsonify
import base64
import cv2
from ultralytics import YOLO
from detectron2.utils.logger import setup_logger
from object_detection import load_model, unload_model, generate_frames  # Functions imported
from flask_mail import Mail, Message

setup_logger()

app = Flask(__name__)
app.jinja_env.autoescape = True 
models_directory = "models"
camera_indices = [0, 4]  # Known camera indexes

camera_locations = {
    0: "Entrance, Gate A",
    4: "Lobby, Long A",
    # Add other camera indices and their locations here
}

# Flask-Mail configuration
app.config['MAIL_SERVER']="smtp.gmail.com"
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = "guardlens2024@gmail.com"
app.config['MAIL_PASSWORD'] = "bgly sbeu gcer xvkv"
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

mail = Mail(app)  # Initialize Flask-Mail

@app.route('/')
def index():
    model_files = []
    for root, dirs, files in os.walk(models_directory):
        for file in files:
            if file.endswith('.pt') or file.endswith('.pth'):
                model_files.append(os.path.relpath(os.path.join(root, file), models_directory))
    
    # Include the camera locations in the context passed to the template
    return render_template(
        'index.html', 
        model_files=model_files, 
        camera_indices=camera_indices, 
        camera_locations=camera_locations
    )

@app.route('/load_model', methods=['POST'])
def handle_load_model():
    model_name = request.form.get('model_name')
    load_model(model_name)
    return "Model loaded successfully", 200

@app.route('/unload_model', methods=['POST'])
def handle_unload_model():
    unload_model()
    return "Model unloaded successfully", 200

@app.route('/video_feed/<int:camera_index>')  # Corrected route definition
def video_feed(camera_index):
    # Ensure valid camera index
    if camera_index not in camera_indices:  # Validate against the list of known camera indexes
        return "Invalid camera index", 400
    return Response(generate_frames(camera_index), mimetype='text/event-stream')

@app.route('/send_alert_email', methods=['POST'])
def send_alert_email():
    recipients_list = ['htootayzaaung.01@gmail.com', 'vladtheimpaler969@gmail.com']

    msg = Message(
        subject='Alert Notification: Lethal Object Detected!',
        sender=app.config['MAIL_USERNAME'],
        recipients=recipients_list,
    )

    image_filename = request.json.get("image_filename")
    camera_index = request.json.get("camera_index")

    # Retrieve camera location based on the index
    if camera_index in camera_locations:
        camera_location = camera_locations[camera_index]
    else:
        camera_location = "Unknown"

    image_html = ""
    if image_filename and os.path.exists(image_filename):
        with open(image_filename, "rb") as image_file:
            image_data = image_file.read()
            msg.attach(os.path.basename(image_filename), "image/jpeg", image_data)
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            image_html = f'<img src="data:image/jpeg;base64,{image_base64}" alt="Detected Object">'

    msg.html = render_template(
        'alert_email.html',
        image_html=image_html,
        camera_location=camera_location,
    )

    mail.send(msg)
    return jsonify({"status": "Email sent successfully"}), 200

if __name__ == '__main__':
    app.run(debug=True, threaded=True)  # Start the Flask server
    
    