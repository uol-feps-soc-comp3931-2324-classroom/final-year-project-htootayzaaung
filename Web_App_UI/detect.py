import cv2
from utility_functions import overlay, plot_one_box, correct_coordinates
from detectron2.engine import DefaultPredictor
import random

DETECTRON2_CLASS_NAMES = ["Axe", "Gun", "Knife"]

def detect_objects(frame, current_model, model_type):
    total_box_area = 0  # Variable to track total bounding box area
    frame_height, frame_width, _ = frame.shape
    total_camera_area = frame_width * frame_height  # Total frame area

    bbox_dimensions = []  # New: List to store bounding box dimensions

    if current_model is None:
        return frame, 0, []  # Return original frame, 0% coverage, and empty list for bounding box dimensions

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