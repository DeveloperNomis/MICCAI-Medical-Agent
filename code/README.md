# Code overview

This folder contains the code used for the final experiments reported in the paper.

All commands below are intended to be run from the repository root.

## Structure

```text
code/
├── hybrid_agent/
├── vlm_baselines/
└── evaluation/
```

## hybrid_agent/

Contains the final hybrid-agent implementation.

The agent decomposes spatial medical image QA into:

1. natural-language query handling,
2. task routing,
3. YOLO-based anatomical localization,
4. deterministic geometric verification,
5. final binary answer generation.

The spatial decision is not produced directly by the language model. Instead, the final answer is derived from detected anatomical object centers and explicit geometric rules.

Main scripts:

- `agent_eval.py`  
  Runs the hybrid agent on the spatial QA benchmark and generates prediction files used by the evaluation scripts.

- `main.py`  
  Provides a lightweight single-case demo for interactive testing. This script is not used to compute the paper metrics.

## vlm_baselines/

Contains scripts for evaluating direct Vision-Language Model baselines under the unified binary prompt protocol.

The evaluated models are prompted to answer each spatial question with a strict binary output:

```text
1 = Yes
0 = No
```

Main scripts:

- `run_vlm_baseline.py`  
  Runs direct VLM inference for models such as Qwen2-VL or MedGemma.

- `evaluate_vlm_outputs.py`  
  Evaluates generated VLM prediction files and writes JSON, CSV, and Excel summaries.

The direct VLM baselines are evaluated without YOLO, tool use, or deterministic geometry.

## evaluation/

Contains scripts for computing the final hybrid-agent metrics and stage-wise error analysis.

Main scripts:

- `final_evaluate_agent.py`  
  Computes strict accuracy, valid-only accuracy, precision, recall, and F1 for hybrid-agent predictions.

- `final_evaluate_agent_failure_attribution.py`  
  Computes the same evaluation metrics and additionally assigns incorrect or invalid predictions to pipeline stages such as parsing, ontology matching, missing detection, imprecise localization, geometry ambiguity, or formatting/runtime failure.

## Configuration

The scripts use local file paths for data, predictions, model checkpoints, and output directories. These paths should be configured through environment variables or command-line arguments.

### Hybrid-agent configuration

```bash
export PROJECT_ROOT=/path/to/repo
export PYTHONPATH=$PROJECT_ROOT/code:$PYTHONPATH

export DATA_YAML_PATH=/path/to/yolo_dataset/data.yaml
export YOLO_WEIGHTS_PATH=/path/to/yolo_detector_best.pt

export ORG_GT_PATH=/path/to/MIRP_Benchmark/RQ1/qa.json
export IMG_DIR=/path/to/MIRP_Benchmark/RQ1/images
export OUTPUT_DIR=/path/to/output/hybrid_agent

export LOCAL_LLM_MODEL=google/medgemma-4b-it
export JOB_ID=run
```

### Direct VLM configuration

```bash
export IMAGE_DIR=/path/to/MIRP_Benchmark/RQ1/images
export QA_FILE=/path/to/MIRP_Benchmark/RQ1/qa.json
export OUTPUT_DIR=/path/to/output/vlm_baselines
```

## Running the code

The evaluation scripts assume that prediction files have already been generated.

### 1. Run hybrid-agent inference

Run the hybrid agent on the spatial QA benchmark to generate prediction files:

```bash
export PROJECT_ROOT=/path/to/repo
export PYTHONPATH=$PROJECT_ROOT/code:$PYTHONPATH

export DATA_YAML_PATH=/path/to/yolo_dataset/data.yaml
export YOLO_WEIGHTS_PATH=/path/to/yolo_detector_best.pt

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

### 2. Evaluate hybrid-agent predictions

After inference has finished, compute the final hybrid-agent metrics:

```bash
export PROJECT_ROOT=/path/to/repo
export PYTHONPATH=$PROJECT_ROOT/code:$PYTHONPATH
export JOB_ID=run

export ORG_GT_PATH=/path/to/MIRP_Benchmark/RQ1/qa.json
export IMG_DIR=/path/to/MIRP_Benchmark/RQ1/images
export PREDICTIONS_JSONL=/path/to/output/hybrid_agent/agent_spatial_predictions_<JOB_ID>.jsonl
export OUTPUT_DIR=/path/to/output/evaluation

mkdir -p "$OUTPUT_DIR"

python code/evaluation/final_evaluate_agent.py
```

This computes accuracy, precision, recall, F1, invalid outputs, and wrong cases.

### 3. Run failure attribution

To compute the stage-wise failure attribution used in the paper:

```bash
export PROJECT_ROOT=/path/to/repo
export PYTHONPATH=$PROJECT_ROOT/code:$PYTHONPATH
export JOB_ID=run

export ORG_GT_PATH=/path/to/MIRP_Benchmark/RQ1/qa.json
export IMG_DIR=/path/to/MIRP_Benchmark/RQ1/images
export PREDICTIONS_JSONL=/path/to/output/hybrid_agent/agent_spatial_predictions_<JOB_ID>.jsonl
export OUTPUT_DIR=/path/to/output/evaluation

mkdir -p "$OUTPUT_DIR"

python code/evaluation/final_evaluate_agent_failure_attribution.py
```

This script expects the hybrid-agent prediction files from the inference step and writes enriched wrong cases, evaluation metrics, and LaTeX table rows.

### 4. Run direct VLM baselines

Direct VLM baselines are evaluated separately under the unified binary prompt protocol.

First set the local MIRP paths:

```bash
export IMAGE_DIR=/path/to/MIRP_Benchmark/RQ1/images
export QA_FILE=/path/to/MIRP_Benchmark/RQ1/qa.json
export OUTPUT_DIR=/path/to/output/vlm_baselines

mkdir -p "$OUTPUT_DIR"
```

Example Qwen2-VL run:

```bash
python code/vlm_baselines/run_vlm_baseline.py \
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

Example MedGemma run:

```bash
python code/vlm_baselines/run_vlm_baseline.py \
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

Evaluate direct VLM outputs:

```bash
python code/vlm_baselines/evaluate_vlm_outputs.py \
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

## Data and checkpoints

The original medical images, segmentation masks, benchmark data, generated prediction files, and trained model checkpoints are not included in this repository.

Expected data formats are described in the main repository README and in the documentation files.
