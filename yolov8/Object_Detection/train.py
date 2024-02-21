import os
import subprocess

def train_yolov8(data_yaml, model_weights, img_size=640, epochs=20, batch_size=8):
    """
    Train a YOLOv8 model with specified parameters.

    Parameters:
    - data_yaml: Path to the YAML file with dataset configuration.
    - model_weights: Path to the pre-trained model weights.
    - img_size: Image size for training.
    - epochs: Number of training epochs.
    - batch_size: Batch size for training.
    """
    # Construct the command to run the training
    train_cmd = f"yolo task=detect mode=train epochs={epochs} data={data_yaml} model={model_weights} imgsz={img_size} batch={batch_size}"
    
    # Execute the training command
    try:
        print(f"Starting training with command: {train_cmd}")
        subprocess.run(train_cmd, check=True, shell=True)
        print("Training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during training: {e}")

# Define your configuration
data_yaml = 'data_custom.yaml'
model_weights = 'yolov8m.pt'
img_size = 640
epochs = 50
batch_size = 8
device = 'gpu'

# Train the model
train_yolov8(data_yaml, model_weights, img_size, epochs, batch_size)

