# A Modular Medical Imaging Agent for Reliable and Auditable Spatial Relation Verification in CT Scans

Code for the MICCAI Agent Workshop submission:

**A Modular Medical Imaging Agent for Reliable and Auditable Spatial Relation Verification in CT Scans**

This repository implements a modular medical imaging agent for binary spatial relation verification in axial CT slices.

Given a CT slice and a question such as:

```text
Is the liver left of the spleen?
```

the system extracts a structured spatial query, localizes the queried anatomical structures with a YOLO-based detector, and verifies the spatial relation using deterministic geometric rules.

The final spatial decision is not directly predicted by a vision-language model. Instead, the language model is used for language/query handling, while the final answer is computed from detected object centers.

---

## Overview

The hybrid agent decomposes spatial medical image QA into explicit and auditable stages:

1. question extraction,
2. task routing,
3. language parsing / query extraction,
4. ontology matching,
5. YOLO-based anatomical localization,
6. deterministic geometric verification,
7. binary answer formatting and audit logging.

The direct VLM baselines receive the same image and question directly and are prompted to output exactly one binary character:

```text
1 = Yes
0 = No
```

---

## Repository structure

```text
spatial-relation-verification-agent/
├── README.md
├── requirements.txt
├── requirements-vlm.txt
├── .gitignore
│
├── configs/
│   ├── data.example.yaml
│   └── classes.example.txt
│
├── code/
│   ├── README.md
│   │
│   ├── hybrid_agent/
│   │   ├── README.md
│   │   ├── agent_eval.py
│   │   ├── main.py
│   │   ├── agent/
│   │   ├── language_parsing/
│   │   ├── perception/
│   │   ├── tasks/
│   │   └── utils/
│   │
│   ├── evaluation/
│   │   ├── README.md
│   │   ├── final_evaluate_agent.py
│   │   └── final_evaluate_agent_failure_attribution.py
│   │
│   └── vlm_baselines/
│       ├── README.md
│       ├── run_vlm_baseline.py
│       └── evaluate_vlm_outputs.py
│
├── training/
│   ├── README.md
│   ├── create_dataset_structure.py
│   ├── extend_labelsTraining.py
│   ├── extend_labelsTraining_ClassMapping.py
│   ├── extend_trainingData.py
│   ├── generate_gray_yolo_IDs.py
│   └── generate_labels.py
│
└── docs/
    ├── data_preparation.md
    ├── data_format.md
    └── evaluation.md
```

---

## What is not included

This repository intentionally excludes medical data, checkpoints, generated predictions, logs, and cluster-specific scripts.

Not included:

```text
data/
datasets/
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

Users must provide local data, model checkpoints, and output directories.

---

## Installation

We recommend Python 3.10+ and a CUDA-capable GPU for full inference.

Clone the repository:

```bash
git clone https://github.com/<anonymous>/<repo-name>.git
cd <repo-name>
```

Create and activate the hybrid-agent environment:

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm

export PYTHONPATH=$PWD/code:$PYTHONPATH
```

Optional import check:

```bash
python - <<'PY'
import torch
import ultralytics
import spacy
import transformers
import cv2
import numpy
import pandas
import yaml
import sklearn

print("All core imports OK")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY
```

For direct VLM baselines, use the separate VLM environment:

```bash
python -m venv .venv-vlm
source .venv-vlm/bin/activate

pip install --upgrade pip
pip install -r requirements-vlm.txt
```

The VLM environment requires a CUDA-capable Linux setup compatible with vLLM. Depending on the local CUDA driver, GPU type, PyTorch version, and vLLM version, users may need to adapt the PyTorch or vLLM installation commands.

---

## Data download and preparation

The spatial QA benchmark data are not included in this repository.

The benchmark data can be downloaded from the official MIRP Benchmark repository:

```text
https://github.com/Wolfda95/MIRP_Benchmark
```

For this repository, the relevant local files are CT slice images and the corresponding binary spatial question-answer file.

A typical local layout is:

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

More details are provided in:

```text
docs/data_preparation.md
docs/data_format.md
```

---

## Required local files

Before running inference or evaluation, provide the following local paths:

| Variable | Description |
|---|---|
| `DATA_YAML_PATH` | local YOLO `data.yaml` file with detector class names |
| `YOLO_WEIGHTS_PATH` | local trained YOLO detector checkpoint, e.g. `best.pt` |
| `ORG_GT_PATH` | MIRP-style QA file, e.g. `RQ1/qa.json` |
| `IMG_DIR` | directory containing the corresponding CT slice images |
| `OUTPUT_DIR` | directory where predictions and evaluation outputs will be written |
| `LOCAL_LLM_MODEL` | local or Hugging Face model identifier for the agent language component |
| `JOB_ID` | run identifier used in output filenames |

An example YOLO data configuration is provided in:

```text
configs/data.example.yaml
```

Replace the `path` field in that file with the local YOLO dataset root before training or inference.

---

## Detector checkpoint

Hybrid-agent inference requires a trained YOLO detector checkpoint.

The checkpoint is not included in this repository. Provide a local checkpoint and set:

```bash
export YOLO_WEIGHTS_PATH=/path/to/yolo_detector_best.pt
```

If no checkpoint is available, train a detector first using the instructions in:

```text
training/README.md
```

After training with Ultralytics YOLO, the checkpoint is typically located at:

```text
runs/detect/train/weights/best.pt
```

Set `YOLO_WEIGHTS_PATH` to that local file.

---

## Quickstart: hybrid-agent inference

Set the required paths:

```bash
export PROJECT_ROOT=$PWD
export PYTHONPATH=$PWD/code:$PYTHONPATH

export DATA_YAML_PATH=/path/to/yolo_dataset/data.yaml
export YOLO_WEIGHTS_PATH=/path/to/yolo_detector_best.pt

export ORG_GT_PATH=/path/to/MIRP_Benchmark/RQ1/qa.json
export IMG_DIR=/path/to/MIRP_Benchmark/RQ1/images
export OUTPUT_DIR=/path/to/output/hybrid_agent

export LOCAL_LLM_MODEL=google/medgemma-4b-it
export JOB_ID=run

mkdir -p "$OUTPUT_DIR"
```

Run benchmark inference:

```bash
python code/hybrid_agent/agent_eval.py
```

Expected output files:

```text
agent_spatial_predictions_<JOB_ID>.jsonl
agent_spatial_predictions_<JOB_ID>.json
agent_spatial_summary_<JOB_ID>.json
agent_spatial_errors_<JOB_ID>.json
```

For single-case debugging, see:

```text
code/hybrid_agent/README.md
```

---

## Quickstart: hybrid-agent evaluation

After inference, compute the final metrics:

```bash
export PROJECT_ROOT=$PWD
export PYTHONPATH=$PWD/code:$PYTHONPATH
export JOB_ID=run

export ORG_GT_PATH=/path/to/MIRP_Benchmark/RQ1/qa.json
export IMG_DIR=/path/to/MIRP_Benchmark/RQ1/images
export PREDICTIONS_JSONL=/path/to/output/hybrid_agent/agent_spatial_predictions_<JOB_ID>.jsonl
export OUTPUT_DIR=/path/to/output/evaluation

mkdir -p "$OUTPUT_DIR"

python code/evaluation/final_evaluate_agent.py
```

Run stage-wise failure attribution:

```bash
python code/evaluation/final_evaluate_agent_failure_attribution.py
```

More details are provided in:

```text
code/evaluation/README.md
docs/evaluation.md
```

---

## Quickstart: direct VLM baselines

Direct VLM baselines are evaluated without YOLO, tool use, or deterministic geometry.

Set the local MIRP paths:

```bash
export IMAGE_DIR=/path/to/MIRP_Benchmark/RQ1/images
export QA_FILE=/path/to/MIRP_Benchmark/RQ1/qa.json
export OUTPUT_DIR=/path/to/output/vlm_baselines

mkdir -p "$OUTPUT_DIR"
```

Example Qwen2-VL run:

```bash
python code/vlm_baselines/run_vlm_baseline.py   --model_path Qwen/Qwen2-VL-7B-Instruct   --image_dir "$IMAGE_DIR"   --qa_file "$QA_FILE"   --output_root "$OUTPUT_DIR"   --model_name qwen2vl_direct   --tokenizer_mode auto   --temperature 0   --max_tokens 1   --dtype auto   --trust_remote_code
```

If Qwen2-VL does not load correctly in the local vLLM version, retry with:

```bash
--tokenizer_mode qwen_vl
```

Example MedGemma run:

```bash
python code/vlm_baselines/run_vlm_baseline.py   --model_path google/medgemma-4b-it   --image_dir "$IMAGE_DIR"   --qa_file "$QA_FILE"   --output_root "$OUTPUT_DIR"   --model_name medgemma_direct   --tokenizer_mode auto   --temperature 0   --max_tokens 1   --dtype auto   --trust_remote_code
```

The tokenizer mode is model- and vLLM-version-dependent. Start with `--tokenizer_mode auto`. If another model requires a different tokenizer mode, processor, chat template, image input format, maximum context length, or vLLM backend option, adapt the command and, if necessary, the model-loading section in:

```text
code/vlm_baselines/run_vlm_baseline.py
```

Evaluate direct VLM outputs:

```bash
python code/vlm_baselines/evaluate_vlm_outputs.py   --results_path "$OUTPUT_DIR"   --output_dir "$OUTPUT_DIR/evaluation"
```

More details are provided in:

```text
code/vlm_baselines/README.md
```

---

## YOLO detector training

Optional detector data-preparation utilities are provided in:

```text
training/
```

The original CT images, segmentation masks, YOLO labels, and trained detector checkpoints are not included.

The best detector configuration used standard YOLOv8 training settings without a custom hyperparameter YAML file.

Example training command:

```bash
yolo detect train   model=yolov8m.pt   data=/path/to/yolo_dataset/data.yaml   imgsz=512   epochs=500   batch=128   freeze=2
```

Adjust `model`, `batch`, `epochs`, and hardware-specific settings as needed.

For details, see:

```text
training/README.md
```

---

## Output directories

Generated files are written locally and are excluded from version control.

Typical local output layout:

```text
results/
  inference/
    agent_eval/
      agent_spatial_predictions_<JOB_ID>.jsonl
      agent_spatial_predictions_<JOB_ID>.json
      agent_spatial_summary_<JOB_ID>.json
      agent_spatial_errors_<JOB_ID>.json

  evaluation/
    agent_eval_metrics_<JOB_ID>.json
    agent_failure_attribution_<JOB_ID>.json

  vlm_baselines/
    qwen2vl_direct/
    medgemma_direct/
    evaluation/

runs/
  detect/
    train*/
      weights/
        best.pt

logs/
  *.out
  *.err
```

These folders are ignored by `.gitignore`.

---

## Results

Main results under the unified prompt protocol:

| System | Accuracy | F1 | Invalid |
|---|---:|---:|---:|
| MedGemma direct | 51.8 | 67.2 | 0.0 |
| MedGemma + hybrid agent | 91.6 | 91.3 | 0.0 |
| Qwen2-VL direct | 51.6 | 56.8 | 0.0 |
| Qwen2-VL + hybrid agent | **94.1** | **94.2** | 0.0 |

Accuracy and F1 are reported in percent. Invalid outputs are counted as incorrect under strict evaluation.

---

## Failure attribution summary

For the best-performing hybrid-agent configuration, Qwen2-VL + hybrid agent, the 55 incorrect cases were assigned to the earliest identifiable failing stage.

| Failure stage | Cases | Share |
|---|---:|---:|
| Question extraction | 0 | 0.0 |
| Routing | 0 | 0.0 |
| Parsing / query extraction | 15 | 27.3 |
| Ontology matching | 0 | 0.0 |
| Missing detection | 12 | 21.8 |
| Imprecise localization | 28 | 50.9 |
| Geometry ambiguity | 0 | 0.0 |
| Formatting/runtime | 0 | 0.0 |

Most remaining errors arise from perception-related failures, especially imprecise localization and missing detections. Full details and the evaluation protocol are described in:

```text
docs/evaluation.md
code/evaluation/README.md
```

## Reproducibility notes

- All systems use the same external binary prompt protocol.
- Decoding is deterministic where applicable.
- The hybrid agent receives the same user-facing prompt as the direct VLM baselines.
- The final spatial decision in the hybrid agent is computed from YOLO detections and deterministic geometry.
- The language model is used for language/query handling and optional fallback parsing, not for directly predicting the spatial truth value.
- Segmentation masks are not used during inference.
- Patient-level separation is enforced between detector training data and held-out evaluation cases.
- Generated outputs, logs, checkpoints, and medical data are intentionally excluded from the repository.
- Cluster-specific Slurm scripts are not included. The commands above can be run directly after setting the required environment variables.

---

## Additional documentation

| File | Content |
|---|---|
| `code/README.md` | overview of the code folder |
| `code/hybrid_agent/README.md` | hybrid-agent execution and configuration |
| `code/vlm_baselines/README.md` | direct VLM baseline inference and evaluation |
| `code/evaluation/README.md` | hybrid-agent evaluation and failure attribution |
| `training/README.md` | YOLO detector training utilities |
| `docs/data_preparation.md` | data download and preparation |
| `docs/data_format.md` | expected local data format |
| `docs/evaluation.md` | evaluation protocol |

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{anonymous2026modular,
  title={A Modular Medical Imaging Agent for Reliable and Auditable Spatial Relation Verification in CT Scans},
  author={Anonymous},
  booktitle={MICCAI Agent Workshop},
  year={2026}
}
```

---

## License

This repository is released for research use.

Please check the licenses of all external datasets, pretrained models, and medical data sources before use.

---

## Acknowledgements

This work builds on open-source tools including YOLOv8, Transformers, LangChain, spaCy, PyTorch, and vLLM.
