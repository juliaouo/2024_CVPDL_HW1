import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Filter object detection results based on score threshold.')
parser.add_argument('--threshold', type=float, default=0.3, help='Score threshold for filtering results (default: 0.3)')
parser.add_argument('--input_file', type=str, default="out/result.json", help='Input path for the JSON score files')
parser.add_argument('--output_file', type=str, default="out/result_thresh_0.3.json", help='Output path for the filtered JSON files')

args = parser.parse_args()

threshold = args.threshold
input_file = args.input_file
output_file = args.output_file

with open(input_file, 'r') as f:
    data = json.load(f)

filtered_data = {}

for name, content in tqdm(data.items()):
    scores = content['scores']
    boxes = content['boxes']
    labels = content['labels']

    if threshold == 0:
        filtered_data[name] = {
            'boxes': boxes,
            'labels': labels
        }
    else:
        filtered_boxes = []
        filtered_labels = []

        for i, score in enumerate(scores):
            if score > threshold:
                filtered_boxes.append(boxes[i])
                filtered_labels.append(labels[i])

        filtered_data[name] = {
            'boxes': filtered_boxes,
            'labels': filtered_labels
        }

with open(output_file, 'w') as f:
    json.dump(filtered_data, f, indent=4)
