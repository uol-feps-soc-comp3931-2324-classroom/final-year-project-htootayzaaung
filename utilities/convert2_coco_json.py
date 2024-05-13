import os
import json
import glob
import cv2

def convert_yolo_to_coco(yolo_dir, output_json):
    """
    Converts YOLO annotations to COCO JSON format for a specific directory (train, val, test).
    """
    images = []
    annotations = []
    categories = [
        {"id": 1, "name": "Axe"},
        {"id": 2, "name": "Handgun"},
        {"id": 3, "name": "Knife"}
    ]

    image_files = glob.glob(os.path.join(yolo_dir, "images", "*.jpg"))
    label_dir = os.path.join(yolo_dir, "labels")
    image_id = 0
    annotation_id = 0

    print(f"Processing directory: {yolo_dir}")

    for image_path in image_files:
        # Retrieve image metadata
        filename = os.path.basename(image_path)
        height, width = cv2.imread(image_path).shape[:2]

        images.append({
            "id": image_id,
            "file_name": filename,
            "height": height,
            "width": width
        })

        # Locate and read the corresponding YOLO annotation file
        label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))
        if not os.path.exists(label_path):
            print(f"Warning: No annotation found for image {filename}")
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            # YOLO format: <class_id> <x_center> <y_center> <width> <height>
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, box_width, box_height = map(float, parts[1:])

            # Convert to absolute coordinates
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height
            x_min = x_center - box_width / 2
            y_min = y_center - box_height / 2

            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id + 1,  # YOLO class IDs start from 0
                "bbox": [x_min, y_min, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

    # Create the COCO-style JSON output
    coco_format = {
        "info": {"year": "2024", "version": "1.0", "description": "Weapon Detection Dataset"},
        "licenses": [{"id": 1, "name": "CC BY 4.0"}],
        "categories": categories,
        "images": images,
        "annotations": annotations
    }

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_json)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=4)

    print(f"Finished writing {output_json}")

def process_datasets(base_dir):
    """
    Processes train, val, and test datasets to convert YOLO annotations to COCO JSON format.
    """
    datasets = ["train", "val", "test"]
    for dataset in datasets:
        yolo_dir = os.path.join(base_dir, dataset)
        output_json = os.path.join(base_dir, "annotations", f"{dataset}_annotations.json")
        print(f"Processing {dataset}...")
        convert_yolo_to_coco(yolo_dir, output_json)

# Example usage:
base_dir = os.getcwd()  # Adjust this to the root directory where the "train", "val", and "test" folders are located
process_datasets(base_dir)

