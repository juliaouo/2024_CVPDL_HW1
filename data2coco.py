import os
import json
from tqdm import tqdm
from PIL import Image

mode = "valid" # train, valid, test
input_dir = "path/to/your/input_dir"
output_dir = "path/to/your/output_dir"


image_dir = os.path.join(input_dir, f"{mode}/images")
if mode == "test":
    annotation_dir = ""
else:
    annotation_dir = os.path.join(input_dir, f"{mode}/labels")
if mode == "valid":
    output_json = os.path.join(output_dir, f"instances_val.json")
else:
    output_json = os.path.join(output_dir, f"instances_{mode}.json")

categories = [
    {"id": 0, "name": "person"},
    {"id": 1, "name": "ear"},
    {"id": 2, "name": "ear-mufs"},
    {"id": 3, "name": "face"},
    {"id": 4, "name": "face-guard"},
    {"id": 5, "name": "face-mask"},
    {"id": 6, "name": "foot"},
    {"id": 7, "name": "tool"},
    {"id": 8, "name": "glasses"},
    {"id": 9, "name": "gloves"},
    {"id": 10, "name": "helmet"},
    {"id": 11, "name": "hands"},
    {"id": 12, "name": "head"},
    {"id": 13, "name": "medical-suit"},
    {"id": 14, "name": "shoes"},
    {"id": 15, "name": "safety-suit"},
    {"id": 16, "name": "safety-vest"}
]

coco_format = {
    "info": {},
    "licenses": [],
    "categories": categories,
    "images": [],
    "annotations": []
}

def data_to_coco_bbox(data_bbox, img_width, img_height):
    x_center, y_center, width, height = data_bbox
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    bbox_width = width * img_width
    bbox_height = height * img_height
    return [x_min, y_min, bbox_width, bbox_height]

def add_image(image_id, img_path):
    with Image.open(img_path) as img:
        width, height = img.size
    return {
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": os.path.basename(img_path)
    }

def add_annotations(annotation_id, image_id, data_ann_path, img_width, img_height):
    annotations = []
    with open(data_ann_path, 'r') as file:
        for line in file.readlines():
            category_id, x_center, y_center, width, height = map(float, line.split())
            bbox = data_to_coco_bbox([x_center, y_center, width, height], img_width, img_height)
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(category_id),
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            }
            annotations.append(annotation)
            annotation_id += 1
    return annotations

image_files = [f for f in os.listdir(image_dir)]
annotation_id = 0

for image_id, img_file in tqdm(enumerate(image_files), desc="Processing Images", ncols=100):
    img_path = os.path.join(image_dir, img_file)
    
    image_info = add_image(image_id, img_path)
    coco_format["images"].append(image_info)
    
    data_ann_file = os.path.join(annotation_dir, img_file.split('.')[0] + '.txt')
    
    if os.path.exists(data_ann_file):
        img_width = image_info["width"]
        img_height = image_info["height"]
        annotations = add_annotations(annotation_id, image_id, data_ann_file, img_width, img_height)
        coco_format["annotations"].extend(annotations)
        annotation_id += len(annotations)

with open(output_json, 'w') as f:
    json.dump(coco_format, f, indent=4)