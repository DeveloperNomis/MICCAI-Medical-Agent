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

The external prompt follows the same binary protocol as the direct VLM baselines:

```text
This is a 2D axial CT slice.

Question: {question}

Answer the question with exactly one character:
1 if the statement is true.
0 if the statement is false.

Do not output any explanation.
```

The key idea is that the external interface remains comparable to a conventional VLM, while the hybrid agent internally performs explicit tool-based verification.

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
│       ├── run_direct_vlm_baseline.py
│       └── evaluate_direct_vlm_outputs.py
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

The following files are not included in this repository:

```text
data/images/
data/labels/
data/relations/
models/
results/
runs/
logs/
*.pt
*.pth
*.ckpt
*.safetensors
```

In particular, the repository does not include:

- original medical images,
- segmentation masks,
- YOLO label files derived from medical data,
- benchmark ground-truth files,
- trained detector checkpoints,
- VLM checkpoints,
- generated prediction files,
- logs or cluster-specific outputs.

These files must be provided locally by the user.

---

## Installation

We recommend Python 3.10+ and a CUDA-capable GPU for full inference.

Clone the repository:

```bash
git clone https://github.com/<anonymous>/<repo-name>.git
cd <repo-name>
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the hybrid-agent dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Set the Python path:

```bash
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

The direct VLM baseline environment is installed separately because it may require additional inference backends such as vLLM:

```bash
python -m venv .venv-vlm
source .venv-vlm/bin/activate

pip install --upgrade pip
pip install -r requirements-vlm.txt
```

The direct VLM environment requires a CUDA-capable Linux setup compatible with vLLM. Depending on the local CUDA driver and hardware, users may need to adapt the PyTorch or vLLM installation commands.

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

Set the corresponding local paths before running inference:

```bash
export ORG_GT_PATH=/path/to/MIRP_Benchmark/RQ1/qa.json
export IMG_DIR=/path/to/MIRP_Benchmark/RQ1/images
export OUTPUT_DIR=/path/to/output
```

More detailed instructions for preparing the local data structure, configuring paths, and preparing YOLO detector data are provided in:

```text
docs/data_preparation.md
```

---

## Required local files

Before running inference or evaluation, provide the following local files.

| Variable | Description |
|---|---|
| `DATA_YAML_PATH` | Local YOLO `data.yaml` file with detector class names |
| `YOLO_WEIGHTS_PATH` | Local trained YOLO detector checkpoint, e.g. `best.pt` |
| `ORG_GT_PATH` | MIRP-style QA file, e.g. `RQ1/qa.json` |
| `IMG_DIR` | Directory containing the corresponding CT slice images |
| `OUTPUT_DIR` | Directory where predictions and evaluation outputs will be written |
| `LOCAL_LLM_MODEL` | Local or Hugging Face model identifier for the agent language component |
| `JOB_ID` | Run identifier used in output filenames |

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
export YOLO_WEIGHTS_PATH=/path/to/best.pt
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

Optional: if an anonymized detector checkpoint is made available separately, place the download link here and set `YOLO_WEIGHTS_PATH` to the downloaded checkpoint.

---

## Model checkpoints and VLM identifiers

The repository does not include trained checkpoints or downloaded VLM weights.

For hybrid-agent inference, provide:

| Component | How to configure |
|---|---|
| YOLO detector | set `YOLO_WEIGHTS_PATH=/path/to/best.pt` |
| Agent language model | set `LOCAL_LLM_MODEL=/path/to/model` or a Hugging Face model identifier |

For direct VLM baselines, the model is passed through the `--model_path` argument.

Example Qwen2-VL:

```bash
--model_path Qwen/Qwen2-VL-7B-Instruct
```

Example MedGemma:

```bash
--model_path google/medgemma-4b-it
```

Depending on the model license and access settings, users may need to authenticate with Hugging Face before downloading or running the model.

---

## Configuration

Set all paths explicitly before running the hybrid agent. No Slurm script is required.

```bash
export PROJECT_ROOT=$PWD
export PYTHONPATH=$PWD/code:$PYTHONPATH

export DATA_YAML_PATH=/path/to/yolo_dataset/data.yaml
export YOLO_WEIGHTS_PATH=/path/to/best.pt

export ORG_GT_PATH=/path/to/MIRP_Benchmark/RQ1/qa.json
export IMG_DIR=/path/to/MIRP_Benchmark/RQ1/images
export OUTPUT_DIR=/path/to/output/hybrid_agent

export LOCAL_LLM_MODEL=google/medgemma-4b-it
export JOB_ID=run
```

`LOCAL_LLM_MODEL` can be either a Hugging Face model identifier or a local model path, for example:

```bash
export LOCAL_LLM_MODEL=google/medgemma-4b-it
```

or:

```bash
export LOCAL_LLM_MODEL=/path/to/local/medgemma-4b-it
```

---

## Running the hybrid agent

### Single-case demo

The script `code/hybrid_agent/main.py` provides a lightweight single-case demo.

```bash
export PROJECT_ROOT=$PWD
export PYTHONPATH=$PWD/code:$PYTHONPATH

export DATA_YAML_PATH=/path/to/yolo_dataset/data.yaml
export YOLO_WEIGHTS_PATH=/path/to/best.pt
export LOCAL_LLM_MODEL=google/medgemma-4b-it

export IMAGE_PATH=/path/to/example_image.png
export QUESTION="Is the liver left of the spleen?"
export OUTPUT_DIR=/path/to/output/single_case

mkdir -p "$OUTPUT_DIR"

python code/hybrid_agent/main.py
```

This script is intended for quick testing and debugging. It is not used to compute the paper metrics.

### Benchmark inference

The script `code/hybrid_agent/agent_eval.py` runs the hybrid agent on the spatial QA benchmark.

```bash
export PROJECT_ROOT=$PWD
export PYTHONPATH=$PWD/code:$PYTHONPATH

export DATA_YAML_PATH=/path/to/yolo_dataset/data.yaml
export YOLO_WEIGHTS_PATH=/path/to/best.pt

export ORG_GT_PATH=/path/to/MIRP_Benchmark/RQ1/qa.json
export IMG_DIR=/path/to/MIRP_Benchmark/RQ1/images
export OUTPUT_DIR=/path/to/output/hybrid_agent

export LOCAL_LLM_MODEL=google/medgemma-4b-it
export JOB_ID=run

mkdir -p "$OUTPUT_DIR"

python code/hybrid_agent/agent_eval.py
```

Expected output files:

```text
agent_spatial_predictions_<JOB_ID>.jsonl
agent_spatial_predictions_<JOB_ID>.json
agent_spatial_summary_<JOB_ID>.json
agent_spatial_errors_<JOB_ID>.json
```

---

## Evaluation

After running hybrid-agent inference, compute the final metrics:

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

This computes:

- strict accuracy,
- valid-only accuracy,
- precision,
- recall,
- F1,
- invalid outputs,
- wrong cases.

Invalid, missing, failed, or non-binary outputs are counted as incorrect under strict evaluation.

---

## Failure attribution

The hybrid agent stores intermediate information for each case, enabling stage-wise error analysis.

Run:

```bash
export PROJECT_ROOT=$PWD
export PYTHONPATH=$PWD/code:$PYTHONPATH
export JOB_ID=run

export ORG_GT_PATH=/path/to/MIRP_Benchmark/RQ1/qa.json
export IMG_DIR=/path/to/MIRP_Benchmark/RQ1/images
export PREDICTIONS_JSONL=/path/to/output/hybrid_agent/agent_spatial_predictions_<JOB_ID>.jsonl
export OUTPUT_DIR=/path/to/output/evaluation

mkdir -p "$OUTPUT_DIR"

python code/evaluation/final_evaluate_agent_failure_attribution.py
```

Failure stages:

| Failure stage | Description |
|---|---|
| Question extraction | The full prompt could not be converted into the core spatial question. |
| Routing | The query was routed to the wrong task pathway. |
| Parsing / query extraction | The wrong entity or relation was extracted. |
| Ontology matching | An extracted entity could not be mapped to a detector class. |
| Missing detection | At least one queried anatomical structure was not detected. |
| Imprecise localization | A detected center was too imprecise for the relation. |
| Geometry ambiguity | The spatial configuration was borderline or ambiguous. |
| Formatting/runtime | The system produced an invalid output or failed during execution. |

Failure attribution assigns each wrong or invalid prediction to the earliest identifiable failing stage.

---

## Direct VLM baselines

Direct VLM baselines are evaluated in a separate environment because they may require additional inference backends such as vLLM.

Direct VLM baseline scripts are located in:

```text
code/vlm_baselines/
```

The folder contains the direct VLM inference script and the corresponding evaluation script:

```text
code/vlm_baselines/run_direct_vlm_baseline.py
code/vlm_baselines/evaluate_direct_vlm_outputs.py
```

The direct baselines use the same external binary prompt protocol as the hybrid agent.

The final evaluated direct baselines are:

- MedGemma direct,
- Qwen2-VL direct.

### Configure input data

```bash
export IMAGE_DIR=/path/to/MIRP_Benchmark/RQ1/images
export QA_FILE=/path/to/MIRP_Benchmark/RQ1/qa.json
export OUTPUT_DIR=/path/to/output/vlm_baselines

mkdir -p "$OUTPUT_DIR"
```

### Run Qwen2-VL direct baseline

Example using a Hugging Face model identifier:

```bash
python code/vlm_baselines/run_direct_vlm_baseline.py \
  --model_path Qwen/Qwen2-VL-7B-Instruct \
  --image_dir "$IMAGE_DIR" \
  --qa_file "$QA_FILE" \
  --output_root "$OUTPUT_DIR" \
  --model_name qwen2vl_direct \
  --temperature 0 \
  --max_tokens 1 \
  --dtype auto \
  --trust_remote_code
```

Example using a local checkpoint:

```bash
python code/vlm_baselines/run_direct_vlm_baseline.py \
  --model_path /path/to/Qwen2-VL-7B-Instruct \
  --image_dir "$IMAGE_DIR" \
  --qa_file "$QA_FILE" \
  --output_root "$OUTPUT_DIR" \
  --model_name qwen2vl_direct \
  --temperature 0 \
  --max_tokens 1 \
  --dtype auto \
  --trust_remote_code
```

### Run MedGemma direct baseline

Example using a Hugging Face model identifier:

```bash
python code/vlm_baselines/run_direct_vlm_baseline.py \
  --model_path google/medgemma-4b-it \
  --image_dir "$IMAGE_DIR" \
  --qa_file "$QA_FILE" \
  --output_root "$OUTPUT_DIR" \
  --model_name medgemma_direct \
  --temperature 0 \
  --max_tokens 1 \
  --dtype auto \
  --trust_remote_code
```

Example using a local checkpoint:

```bash
python code/vlm_baselines/run_direct_vlm_baseline.py \
  --model_path /path/to/medgemma-4b-it \
  --image_dir "$IMAGE_DIR" \
  --qa_file "$QA_FILE" \
  --output_root "$OUTPUT_DIR" \
  --model_name medgemma_direct \
  --temperature 0 \
  --max_tokens 1 \
  --dtype auto \
  --trust_remote_code
```

### Evaluate direct VLM outputs

```bash
python code/vlm_baselines/evaluate_direct_vlm_outputs.py \
  --results_path "$OUTPUT_DIR" \
  --output_dir "$OUTPUT_DIR/evaluation"
```

Expected output files:

```text
direct_vlm_evaluation_summary.json
direct_vlm_evaluation_summary.csv
direct_vlm_evaluation_summary.xlsx
direct_vlm_invalid_cases.json
```

Generated VLM predictions and evaluation outputs are not included in this repository.

---

## YOLO detector training utilities

Optional detector data-preparation utilities are provided in:

```text
training/
```

These scripts were used to prepare YOLO-style detector training data from segmentation-derived annotations.

The original CT images, segmentation masks, labels, and trained detector checkpoints are not included.

The best detector configuration used standard YOLOv8 training settings without a custom hyperparameter YAML file. Exploratory custom hyperparameter files are not included because they were not used for the final reported configuration.

### Expected YOLO dataset layout

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

### Example YOLO training command

```bash
yolo detect train \
  model=yolov8m.pt \
  data=/path/to/yolo_dataset/data.yaml \
  imgsz=512 \
  epochs=500 \
  batch=128 \
  freeze=2
```

Adjust `model`, `batch`, `epochs`, and hardware-specific settings as needed.

After training, set:

```bash
export YOLO_WEIGHTS_PATH=/path/to/runs/detect/train/weights/best.pt
```

This checkpoint is required for hybrid-agent inference.

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
