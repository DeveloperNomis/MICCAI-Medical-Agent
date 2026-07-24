import json
import yaml

# Path to the YOLO dataset configuration file.
DATASET_YAML = "/path/to/data.example.yaml"

# Path to the ground-truth relation file.
ORG_GT_PATH = "/path/to/organs_gt.json"

# Load YOLO class names from data.example.yaml.
with open(DATASET_YAML, "r", encoding="utf-8") as f:
    data_yaml = yaml.safe_load(f)

yolo_names = {v.lower(): int(k) for k, v in data_yaml["names"].items()}

# Load grayscale labels from organs_gt.json.
with open(ORG_GT_PATH, "r", encoding="utf-8") as f:
    org_data = json.load(f)

gray_to_yolo = {}

for entry in org_data:
    for qa in entry.get("question_answer", []):
        # Map both queried objects, object1 and object2.
        for key in ["object1", "object2"]:
            name = qa[f"{key}_name"].lower().strip()
            gray = int(qa[f"{key}_gray"])

            if name in yolo_names:
                gray_to_yolo[gray] = yolo_names[name]

print("Automatically generated grayscale-to-YOLO mapping:")
print(gray_to_yolo)