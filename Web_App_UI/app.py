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
    # List model files
    model_files = []
    for root, dirs, files in os.walk(models_directory):
        for file in files:
            if file.endswith('.pt') or file.endswith('.pth'):
                model_files.append(os.path.relpath(os.path.join(root, file), models_directory))
    return render_template('index.html', model_files=model_files, camera_indices=camera_indices)

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
    # Email recipient (this could be dynamic)
    recipient = 'htootayzaaung.01@gmail.com'
    
    # Create email message
    msg = Message(
        subject='Alert Notification: Lethal Object Detected!',
        sender=app.config['MAIL_USERNAME'],  # Email sender
        recipients=[recipient],  # List of recipients
    )
    
    # Render email content with HTML template
    msg.html = render_template('alert_email.html')  # Rendered HTML content
    
    # Send the email
    mail.send(msg)
    
    return "Email sent successfully", 200


if __name__ == '__main__':
    app.run(debug=True, threaded=True)  # Start the Flask server
    
    