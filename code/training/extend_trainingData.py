import os
import json
import shutil
from tqdm import tqdm

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

# JSON file containing image filenames and patient/volume identifiers.
MERGED_JSON = "/path/to/big_dataset_training.json"

# Directory containing the complete source image collection.
IMAGES_SRC_DIR = "/path/to/images_complete"

# Root directory of the YOLO dataset.
YOLO_ROOT = "/path/to/yolo_dataset"

TRAIN_IMG_DIR = f"{YOLO_ROOT}/images/train"
VAL_IMG_DIR = f"{YOLO_ROOT}/images/val"
TEST_IMG_DIR = f"{YOLO_ROOT}/images/test"

os.makedirs(TRAIN_IMG_DIR, exist_ok=True)

# --------------------------------------------------
# 1. Collect patients/volumes from validation and test splits
# --------------------------------------------------

def get_patients_from_dir(img_dir):
    patients = set()
    for fname in os.listdir(img_dir):
        if "_slice" in fname:
            patients.add(fname.split("_slice")[0])
    return patients


val_patients = get_patients_from_dir(VAL_IMG_DIR)
test_patients = get_patients_from_dir(TEST_IMG_DIR)

excluded_patients = val_patients | test_patients
print(f"Excluded patients/volumes from val/test: {len(excluded_patients)}")

# --------------------------------------------------
# 2. Load JSON and filter training entries
# --------------------------------------------------

with open(MERGED_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

train_entries = [
    entry for entry in data
    if entry["base_name"] not in excluded_patients
]

print("Excluded patients/volumes:", excluded_patients)
print(f"Potential new training entries: {len(train_entries)}")

# --------------------------------------------------
# 3. Copy training images
# --------------------------------------------------

added = 0
skipped_existing = 0
missing = 0

for entry in tqdm(train_entries):
    fname = entry["filename"]

    src = os.path.join(IMAGES_SRC_DIR, fname)
    dst = os.path.join(TRAIN_IMG_DIR, fname)

    if not os.path.exists(src):
        missing += 1
        continue

    if os.path.exists(dst):
        skipped_existing += 1
        continue

    shutil.copy(src, dst)
    added += 1

print("\nSummary")
print(f"New training images added: {added}")
print(f"Already existing images skipped: {skipped_existing}")
print(f"Missing source files: {missing}")