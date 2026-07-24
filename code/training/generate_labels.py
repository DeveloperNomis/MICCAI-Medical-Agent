import json
import os
from PIL import Image

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------

# JSON file containing filenames and spatial QA annotations.
JSON_FILE = "/path/to/qa.json"

# Existing YOLO dataset directory.
DATASET_DIR = "../data"

IMAGE_DIRS = {
    "train": f"{DATASET_DIR}/images/train",
    "val": f"{DATASET_DIR}/images/val",
    "test": f"{DATASET_DIR}/images/test",
}

LABEL_DIRS = {
    "train": f"{DATASET_DIR}/labels/train",
    "val": f"{DATASET_DIR}/labels/val",
    "test": f"{DATASET_DIR}/labels/test",
}

DUMMY_W = 0.05
DUMMY_H = 0.05


# --------------------------------------------------------------
# Determine whether an image belongs to train, val, or test
# --------------------------------------------------------------

def find_split(filename):
    for split, img_dir in IMAGE_DIRS.items():
        if os.path.exists(os.path.join(img_dir, filename)):
            return split
    return None


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------

def main():
    print("=== Generating YOLOv8 pose labels ===")

    # Load JSON annotations.
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract class names and sort alphabetically for a stable mapping.
    class_names = set()
    for item in data:
        qa = item["question_answer"][0]
        class_names.add(qa["object1_name"])
        class_names.add(qa["object2_name"])

    class_names = sorted(class_names)
    class_to_id = {name: i for i, name in enumerate(class_names)}

    print("Class mapping created:")
    for cls, idx in class_to_id.items():
        print(f"  {idx}: {cls}")

    created_files = 0

    # Each JSON entry generates labels for one image.
    for item in data:
        filename = item["filename"]
        qa = item["question_answer"][0]

        split = find_split(filename)
        if split is None:
            print(f"WARNING: {filename} not found in train/val/test. Skipping.")
            continue

        image_path = os.path.join(IMAGE_DIRS[split], filename)
        label_path = os.path.join(LABEL_DIRS[split], filename.replace(".png", ".txt"))

        # Load image to obtain width and height.
        if not os.path.exists(image_path):
            print(f"WARNING: image missing: {image_path}. Skipping.")
            continue

        img = Image.open(image_path)
        W, H = img.size

        # Object 1.
        cid1 = class_to_id[qa["object1_name"]]
        x1 = qa["object1_center_x"] / W
        y1 = qa["object1_center_y"] / H

        # Object 2.
        cid2 = class_to_id[qa["object2_name"]]
        x2 = qa["object2_center_x"] / W
        y2 = qa["object2_center_y"] / H

        # Write YOLOv8 pose label file.
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(f"{cid1} {x1:.6f} {y1:.6f} {DUMMY_W} {DUMMY_H} {x1:.6f} {y1:.6f} 2\n")
            f.write(f"{cid2} {x2:.6f} {y2:.6f} {DUMMY_W} {DUMMY_H} {x2:.6f} {y2:.6f} 2\n")

        created_files += 1

    print(f"\nDone. Created {created_files} label files.")

    # Save class list.
    with open(f"{DATASET_DIR}/yolo_classes_example.txt", "w", encoding="utf-8") as f:
        for cls in class_names:
            f.write(cls + "\n")

    print("Class list saved to dataset/yolo_classes_example.txt.")


if __name__ == "__main__":
    main()