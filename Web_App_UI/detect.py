import cv2
from utility_functions import overlay, plot_one_box, correct_coordinates, draw_label_with_background, CLASS_COLORS, STANDARD_BORDER_THICKNESS, DETECTRON2_CLASS_NAMES
from detectron2.engine import DefaultPredictor
import random

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
                if score >= 0.6:
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
        
        for r in results:
            boxes = r.boxes
            masks = r.masks
            
            if masks is not None:
                masks = masks.data.cpu()  # Ensure on CPU
                for seg, box in zip(masks.data.cpu().numpy(), boxes):
                    if box.conf[0] >= 0.6:
                        class_name = class_names[int(box.cls)]
                        class_color = CLASS_COLORS[class_name]  # Consistent class-specific color
                    
                        seg = cv2.resize(seg, (frame.shape[1], frame.shape[0]))
                        frame = overlay(frame, seg, class_color, 0.4)  # Apply the consistent overlay color

                        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                        xmin, xmax, ymin, ymax = correct_coordinates(xmin, xmax, ymin, ymax)  # Correct coordinates
                        box_area = (xmax - xmin) * (ymax - ymin)
                        total_box_area += max(0, box_area)

                        # Add bounding box dimensions
                        bbox_dimensions.append(f"{xmax - xmin} × {ymax - ymin}")
                    
                        # Draw bounding box with consistent class-specific color
                        plot_one_box([xmin, ymin, xmax, ymax], frame, class_color, f"{class_name} {float(box.conf):.2f}")

    # Calculate object coverage
    object_coverage = (total_box_area / total_camera_area) * 100 if total_camera_area > 0 else 0

    return frame, object_coverage, bbox_dimensions
