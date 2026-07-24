# Detector training utilities

This folder contains optional utilities for preparing YOLO-style detector training data.

The original CT images, segmentation masks, YOLO labels, and trained detector checkpoints are not included in this repository.

## Expected YOLO dataset layout

```text
/path/to/yolo_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

An example configuration is provided in:

```text
configs/data.example.yaml
```

Before training, replace the `path` field in `data.yaml` with the local YOLO dataset root.

## Data preparation scripts

Typical preparation order:

1. `create_dataset_structure.py`  
   Creates a patient-level train/validation/test split and YOLO directory structure.

2. `generate_labels.py`  
   Generates initial YOLO label files from QA or annotation metadata.

3. `generate_gray_yolo_IDs.py`  
   Creates a mapping between segmentation grayscale identifiers and YOLO class IDs.

4. `extend_trainingData.py`  
   Adds additional training images while preserving patient-level separation from validation/test patients.

5. `extend_labelsTraining.py`  
   Generates YOLO labels for the extended training set.

6. `extend_labelsTraining_ClassMapping.py`  
   Maps dataset-specific anatomical labels to the final detector ontology.

Before running these scripts, replace placeholder paths such as `/path/to/...` with local dataset paths.

## Train YOLO detector

The best detector configuration used standard YOLOv8 training settings without a custom hyperparameter YAML file.

Example command:

```bash
yolo detect train \
  model=yolov8s.pt \
  data=/path/to/yolo_dataset/data.yaml \
  imgsz=512 \
  epochs=500 \
  batch=128 \
  freeze=2
```

Adjust `model`, `batch`, `epochs`, and hardware-specific settings as needed.

## Use the trained checkpoint

After training, set:

```bash
export YOLO_WEIGHTS_PATH=/path/to/runs/detect/train/weights/best.pt
```

This checkpoint is required for hybrid-agent inference.