# Data format

The original medical images, segmentation masks, YOLO labels, benchmark files, and trained checkpoints are not included in this repository.

This document describes the expected local data format.

## Image directory

The hybrid agent and direct VLM baselines expect axial CT slices as image files.

Example:

```text
/path/to/test/images/
├── case_0001.png
├── case_0002.png
└── ...
```

## Spatial QA file

The evaluation expects a JSON file containing image-level question-answer pairs.

Example:

```json
[
  {
    "filename": "case_0001.png",
    "question_answer": [
      {
        "question": "Is the liver left of the spleen?",
        "answer": 1,
        "object1_name": "liver",
        "object2_name": "spleen",
        "relation": "left of",
        "object1_center_x": 123.4,
        "object1_center_y": 210.7,
        "object2_center_x": 256.1,
        "object2_center_y": 215.9
      }
    ]
  }
]
```

## YOLO data configuration

An example detector configuration is provided in:

```text
configs/data.example.yaml
```

Before training or inference, replace the `path` field with the local YOLO dataset root.

## Not included

The following files are intentionally excluded:

```text
data/images/
data/labels/
data/relations/
models/
results/
runs/
logs/
```