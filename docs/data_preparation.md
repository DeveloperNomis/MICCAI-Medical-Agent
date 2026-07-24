# Data preparation

This repository does not include medical image data, segmentation masks, YOLO labels, trained checkpoints, generated predictions, or experiment logs.

## MIRP benchmark

The spatial QA benchmark data can be obtained from the official MIRP Benchmark repository:

```text
https://github.com/Wolfda95/MIRP_Benchmark
```

Follow the dataset guide in the MIRP repository to download the benchmark data.

## Relevant subset for this repository

For the main spatial relation verification setting, this repository expects CT slices and binary question-answer pairs.

Typical local layout:

```text
/path/to/MIRP_Benchmark/RQ1/
├── images/
│   ├── case_0001.png
│   ├── case_0002.png
│   └── ...
└── qa.json
```

Set:

```bash
export ORG_GT_PATH=/path/to/MIRP_Benchmark/RQ1/qa.json
export IMG_DIR=/path/to/MIRP_Benchmark/RQ1/images
```

## Expected QA format

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

The fields `question` and `answer` are used for inference and evaluation.

The object-center fields are used only for evaluation and failure attribution. They are not used by the hybrid agent to make predictions during inference.

## YOLO detector data

The YOLO detector requires a separate local detector dataset:

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

Use `configs/data.example.yaml` as a template.

Set:

```bash
export DATA_YAML_PATH=/path/to/yolo_dataset/data.yaml
export YOLO_WEIGHTS_PATH=/path/to/yolo_detector_best.pt
```

## Files excluded from Git

The following files should remain local:

```text
data/
models/
results/
runs/
logs/
*.pt
*.pth
*.ckpt
*.safetensors
*.nii
*.nii.gz
*.dcm
*.zip
```