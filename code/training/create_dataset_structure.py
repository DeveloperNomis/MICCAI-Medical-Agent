import os
import shutil
import json
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------

# Path to the JSON file containing filenames and patient/volume identifiers.
JSON_FILE = "/path/to/qa.json"

# Directory containing the source PNG images.
IMAGES_SRC_DIR = "/path/to/images"

# Output directory for the YOLO dataset.
OUT_DIR = "../data"

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15  # Remaining patients/volumes are used for testing.

# --------------------------------------------------------------
# Create directory structure
# --------------------------------------------------------------

def create_structure():
    dirs = [
        f"{OUT_DIR}/images/train",
        f"{OUT_DIR}/images/val",
        f"{OUT_DIR}/images/test",
        f"{OUT_DIR}/labels/train",
        f"{OUT_DIR}/labels/val",
        f"{OUT_DIR}/labels/test",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    print("YOLO directory structure created.")


# --------------------------------------------------------------
# Patient-level split
# --------------------------------------------------------------

def load_patient_mapping():
    data = json.load(open(JSON_FILE))
    patient_to_files = {}

    for item in data:
        patient = item["base_name"]  # e.g., "amos_0002.nii"
        fname = item["filename"]

        if patient not in patient_to_files:
            patient_to_files[patient] = []
        patient_to_files[patient].append(fname)

    print(f"{len(patient_to_files)} patients/volumes found.")
    return patient_to_files


def split_patients(patient_to_files):
    patients = list(patient_to_files.keys())

    train_patients, temp_patients = train_test_split(
        patients, train_size=TRAIN_SPLIT, random_state=42, shuffle=True
    )

    val_ratio_adjusted = VAL_SPLIT / (1 - TRAIN_SPLIT)
    val_patients, test_patients = train_test_split(
        temp_patients, train_size=val_ratio_adjusted, random_state=42, shuffle=True
    )

    print(f"Train: {len(train_patients)} patients/volumes")
    print(f"Val:   {len(val_patients)} patients/volumes")
    print(f"Test:  {len(test_patients)} patients/volumes")

    return train_patients, val_patients, test_patients


# --------------------------------------------------------------
# Copy images
# --------------------------------------------------------------

def copy_files(patient_to_files, patients, split_name):
    img_target = f"{OUT_DIR}/images/{split_name}"

    count = 0
    for p in patients:
        for fname in patient_to_files[p]:
            src = os.path.join(IMAGES_SRC_DIR, fname)
            dst = os.path.join(img_target, fname)

            if not os.path.exists(src):
                print(f"WARNING: image not found: {src}")
                continue

            shutil.copy(src, dst)
            count += 1

    print(f"{count} images copied to {split_name}/.")


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------

if __name__ == "__main__":
    print("=== YOLO Dataset Generator ===")

    create_structure()
    patient_to_files = load_patient_mapping()
    train_patients, val_patients, test_patients = split_patients(patient_to_files)

    copy_files(patient_to_files, train_patients, "train")
    copy_files(patient_to_files, val_patients, "val")
    copy_files(patient_to_files, test_patients, "test")

    print("\nDone. Train/val/test dataset structure created.")