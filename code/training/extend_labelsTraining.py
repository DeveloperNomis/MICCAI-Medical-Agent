import os
import json

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

# JSON file containing image filenames and YOLO bounding-box annotations.
MERGED_JSON = "/path/to/big_dataset_training.json"

# Root directory of the YOLO dataset.
YOLO_ROOT = "/path/to/yolo_dataset"

TRAIN_IMG_DIR = f"{YOLO_ROOT}/images/train"
TRAIN_LBL_DIR = f"{YOLO_ROOT}/labels/train"

os.makedirs(TRAIN_LBL_DIR, exist_ok=True)

# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    print("=== Generating YOLO detection labels from normalized bounding boxes ===")

    with open(MERGED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --------------------------------------------------
    # 1. Build class list from all label_info entries
    # --------------------------------------------------

    class_names = set()
    for entry in data:
        for obj in entry.get("label_info", []):
            class_names.add(obj["class_name"])

    class_names = sorted(class_names)
    class_to_id = {cls: i for i, cls in enumerate(class_names)}

    print(f"Found {len(class_names)} classes.")

    # Save class list for data.example.yaml creation.
    with open(f"{YOLO_ROOT}/classes.txt", "w") as f:
        for cls in class_names:
            f.write(cls + "\n")

    # --------------------------------------------------
    # 2. Generate labels for training images
    # --------------------------------------------------

    created = 0
    skipped = 0

    for entry in data:
        fname = entry["filename"]

        img_path = os.path.join(TRAIN_IMG_DIR, fname)
        lbl_path = os.path.join(TRAIN_LBL_DIR, fname.replace(".png", ".txt"))

        # Only create labels for images that are actually part of the training split.
        if not os.path.exists(img_path):
            continue

        # Do not overwrite existing label files.
        if os.path.exists(lbl_path):
            skipped += 1
            continue

        lines = []

        for obj in entry.get("label_info", []):
            cls_name = obj["class_name"]
            bbox = obj["yolo_bbox"]

            cid = class_to_id[cls_name]
            xc = bbox["x_center_norm"]
            yc = bbox["y_center_norm"]
            w = bbox["width_norm"]
            h = bbox["height_norm"]

            # YOLO detection format:
            # class_id x_center y_center width height
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        if not lines:
            continue

        with open(lbl_path, "w") as f:
            f.write("\n".join(lines))

        created += 1

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print(f"\nLabels created: {created}")
    print(f"Skipped existing labels: {skipped}")
    print("YOLO label generation completed.")


if __name__ == "__main__":
    main()