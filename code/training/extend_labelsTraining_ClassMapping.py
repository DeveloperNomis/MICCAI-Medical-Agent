import os
import json
from collections import defaultdict
from tqdm import tqdm

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

# JSON file containing image filenames and bounding-box annotations.
MERGED_JSON = "/path/to/big_dataset_training.json"

# Root directory of the YOLO dataset.
YOLO_ROOT = "/path/to/yolo_dataset"

TRAIN_IMG_DIR = f"{YOLO_ROOT}/images/train"
TRAIN_LBL_DIR = f"{YOLO_ROOT}/labels/train"
os.makedirs(TRAIN_LBL_DIR, exist_ok=True)

# --------------------------------------------------
# TARGET CLASSES
# Must match the class order used in data.yaml.
# --------------------------------------------------

TARGET_CLASSES = [
    "aorta",
    "brachiocephalic trunk",
    "colon",
    "duodenum",
    "esophagus",
    "gallbladder",
    "heart",
    "inferior vena cava",
    "left atrial appendage",
    "left autochthon",
    "left brachiocephalic vein",
    "left clavicula",
    "left femur",
    "left gluteus maximus",
    "left gluteus medius",
    "left gluteus minimus",
    "left hip",
    "left humerus",
    "left iliac artery",
    "left iliac vena",
    "left iliopsoas",
    "left kidney",
    "left kidney cyst",
    "left lung lower lobe",
    "left lung upper lobe",
    "left rib",
    "left scapula",
    "liver",
    "pancreas",
    "portal vein and splenic vein",
    "prostate",
    "pulmonary vein",
    "right adrenal gland",
    "right autochthon",
    "right clavicula",
    "right femur",
    "right gluteus maximus",
    "right gluteus medius",
    "right gluteus minimus",
    "right hip",
    "right humerus",
    "right iliac artery",
    "right iliac vena",
    "right iliopsoas",
    "right kidney",
    "right kidney cyst",
    "right lung lower lobe",
    "right lung middle lobe",
    "right lung upper lobe",
    "right scapula",
    "sacrum",
    "skull",
    "small bowel",
    "spinal cord",
    "spleen",
    "sternum",
    "stomach",
    "superior vena cava",
    "trachea",
    "urinary bladder",
    "vertebrae",
]

CLASS_TO_ID = {c: i for i, c in enumerate(TARGET_CLASSES)}

# --------------------------------------------------
# DROP AND MAPPING RULES
# --------------------------------------------------

DROP_CLASSES = {
    "brain",
    "thyroid_gland",
    "costal_cartilages",
    "common_carotid_artery_left",
    "common_carotid_artery_right",
    "subclavian_artery_left",
    "subclavian_artery_right",
}


def map_class(name: str):
    # Map individual vertebrae to a single vertebrae class.
    if name.startswith("vertebrae_"):
        return "vertebrae"

    # Map ribs to the detector rib class.
    if name.startswith("rib_left_") or name.startswith("rib_right_"):
        return "left rib"

    # Map side-aware organ names.
    if name.endswith("_left"):
        return "left " + name.replace("_left", "").replace("_", " ")
    if name.endswith("_right"):
        return "right " + name.replace("_right", "").replace("_", " ")

    # Default mapping: replace underscores with spaces.
    return name.replace("_", " ")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    print("=== Generating YOLO detection labels with final class mapping ===")

    with open(MERGED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    class_counter = defaultdict(int)
    dropped_counter = defaultdict(int)

    created = 0
    skipped_no_img = 0

    for entry in tqdm(data):
        fname = entry["filename"]
        img_path = os.path.join(TRAIN_IMG_DIR, fname)

        if not os.path.exists(img_path):
            skipped_no_img += 1
            continue

        label_lines = []

        for obj in entry.get("label_info", []):
            raw_cls = obj["class_name"]

            if raw_cls in DROP_CLASSES:
                dropped_counter[raw_cls] += 1
                continue

            mapped = map_class(raw_cls)

            if mapped not in CLASS_TO_ID:
                dropped_counter[raw_cls] += 1
                continue

            bbox = obj["yolo_bbox"]
            cid = CLASS_TO_ID[mapped]

            label_lines.append(
                f"{cid} "
                f"{bbox['x_center_norm']:.6f} "
                f"{bbox['y_center_norm']:.6f} "
                f"{bbox['width_norm']:.6f} "
                f"{bbox['height_norm']:.6f}"
            )

            class_counter[mapped] += 1

        if not label_lines:
            continue

        out_path = os.path.join(TRAIN_LBL_DIR, fname.replace(".png", ".txt"))

        # Uncomment the following lines to write label files.
        with open(out_path, "w") as f:
            f.write("\n".join(label_lines))

        created += 1

    # --------------------------------------------------
    # REPORT
    # --------------------------------------------------

    print("\nClass statistics:")
    total_objs = 0
    for cls in TARGET_CLASSES:
        n = class_counter.get(cls, 0)
        total_objs += n
        print(f"{cls:30s} -> {n:6d}")

    print("\nDropped classes:")
    for cls, n in dropped_counter.items():
        print(f"{cls:35s} -> {n}")

    print("\nSummary:")
    print(f"Images with generated labels: {created}")
    print(f"Total objects: {total_objs}")
    print(f"Images skipped because no training image was found: {skipped_no_img}")


if __name__ == "__main__":
    main()