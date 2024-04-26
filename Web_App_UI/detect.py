import cv2
from utility_functions import overlay, plot_one_box, correct_coordinates
from detectron2.engine import DefaultPredictor
import random

DETECTRON2_CLASS_NAMES = ["Axe", "Gun", "Knife"]

# Constants for standardization
STANDARD_BORDER_THICKNESS = 3  # Consistent border thickness
STANDARD_FONT_STYLE = cv2.LINE_AA  # Consistent font style
STANDARD_FONT_SIZE = 1  # Consistent font size
CLASS_COLORS = {
    "Axe": (0, 255, 0),  # Green
    "Gun": (255, 0, 0),  # Blue
    "Knife": (0, 0, 255)  # Red
}

import cv2
from utility_functions import correct_coordinates
from detectron2.engine import DefaultPredictor

# Constants for standardization
STANDARD_BORDER_THICKNESS = 3  # Consistent border thickness
STANDARD_FONT_STYLE = cv2.FONT_HERSHEY_SIMPLEX  # Consistent font style
STANDARD_FONT_SIZE = 1  # Consistent font size
TEXT_COLOR = (255, 255, 255)  # White text color

# Predefined class-specific colors
CLASS_COLORS = {
    "Axe": (0, 255, 0),  # Green
    "Gun": (255, 0, 0),  # Blue
    "Knife": (0, 0, 255)  # Red
}

# Function to draw text label with a background, ensuring proper alignment with the bounding box
def draw_label_with_background(frame, label, position, class_name):
    class_color = CLASS_COLORS[class_name]  # Use class-specific color

    # Calculate text size and adjust the position to reduce the gap
    text_size = cv2.getTextSize(label, STANDARD_FONT_STYLE, STANDARD_FONT_SIZE, STANDARD_BORDER_THICKNESS)[0]

    # Adjust the filled rectangle position to minimize the gap
    rectangle_top_left = (position[0], position[1] - text_size[1])  # Aligns closely with bounding box
    rectangle_bottom_right = (position[0] + text_size[0], position[1] - 1)  # -1 to bring it close to the box edge

    # Draw filled rectangle for the label's background
    cv2.rectangle(frame, rectangle_top_left, rectangle_bottom_right, class_color, -1)

    # Draw the text label, ensuring proper alignment
    cv2.putText(frame, label, (position[0], rectangle_top_left[1] + text_size[1]), STANDARD_FONT_STYLE, STANDARD_FONT_SIZE, TEXT_COLOR, STANDARD_BORDER_THICKNESS)

def detect_objects(frame, current_model, model_type):
    if current_model is None:
        return frame, 0, []  # Return the original frame, 0% coverage, and empty bbox_dimensions

    total_box_area = 0  # Variable to track total bounding box area
    frame_height, frame_width, _ = frame.shape
    total_camera_area = frame_width * frame_height  # Total frame area

    bbox_dimensions = []  # List to store bounding box dimensions

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
                box_area = (x2 - x1) * (y2 - y1)
                total_box_area += max(0, box_area)

                # Store bounding box dimensions
                bbox_dimensions.append(f"{x2 - x1} × {y2 - y1}")
                
                class_name = DETECTRON2_CLASS_NAMES[class_idx]
                class_color = CLASS_COLORS[class_name]  # Obtain the specific class color
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), class_color, STANDARD_BORDER_THICKNESS)
                
                label = f"{class_name}: {score:.2f}"
                draw_label_with_background(frame, label, (x1, y1), class_name)  # Draw label with consistent background

    elif model_type == 'yolo_detection':
        if current_model is not None:
            results = current_model(frame, stream=True)
            for r in results:
                for box in r.boxes:
                    confidence = box.conf[0]
                    if confidence >= 0.6:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x1, x2, y1, y2 = correct_coordinates(x1, x2, y1, y2)  # Ensure correct coordinates
                        box_area = (x2 - x1) * (y2 - y1)
                        total_box_area += max(0, box_area)
                        
                        # Store bounding box dimensions
                        bbox_dimensions.append(f"{x2 - x1} × {y2 - y1}")

                        if current_model is not None:
                            class_name = current_model.names[int(box.cls[0])]
                            class_color = CLASS_COLORS[class_name]  # Ensure correct color based on class
                            cv2.rectangle(frame, (x1, y1), (x2, y2), class_color, STANDARD_BORDER_THICKNESS)  # Draw bounding box
                            label = f"{class_name}: {confidence:.2f}"
                            draw_label_with_background(frame, label, (x1, y1), class_name)  # Draw label with consistent background
                            
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

    # Calculate object coverage
    object_coverage = (total_box_area / total_camera_area) * 100 if total_camera_area > 0 else 0

    return frame, object_coverage, bbox_dimensions
