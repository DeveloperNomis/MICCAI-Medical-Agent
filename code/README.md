# Code overview

This folder contains the code used for the final experiments reported in the paper.

## Structure

```text
code/
├── hybrid_agent/
├── direct_vlm/
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

## direct_vlm/

Contains scripts for evaluating direct Vision-Language Model baselines under the unified binary prompt protocol.

The evaluated models are prompted to answer each spatial question with a strict binary output:

```text
1 = Yes
0 = No
```

These scripts are used for the direct MedGemma and Qwen2-VL baseline results.

## evaluation/

Contains scripts for computing the final paper metrics and error analysis.

Main scripts:

- `final_evaluate_agent.py`  
  Computes strict accuracy, valid-only accuracy, precision, recall, and F1 for hybrid-agent predictions.

- `final_evaluate_agent_failure_attribution.py`  
  Computes the same evaluation metrics and additionally assigns incorrect or invalid predictions to pipeline stages such as parsing, ontology matching, missing detection, imprecise localization, geometry ambiguity, or formatting/runtime failure.

## Configuration

The scripts use local file paths for data, predictions, model checkpoints, and output directories. These paths should be configured through environment variables or local configuration files.

Example:

```bash
export PROJECT_ROOT=/path/to/repo
export ORG_GT_PATH=/path/to/organs_gt.json
export IMG_DIR=/path/to/test/images
export PREDICTIONS_JSONL=/path/to/agent_spatial_predictions.jsonl
export OUTPUT_DIR=/path/to/output
```

## Running the code

The evaluation scripts assume that prediction files have already been generated.

### 1. Run hybrid-agent inference

Run the hybrid agent on the spatial QA benchmark to generate prediction files:

```bash
python code/hybrid_agent/agent_eval.py
```

This should create prediction files such as:

```text
results/inference/agent_eval/agent_spatial_predictions_<JOB_ID>.jsonl
results/inference/agent_eval/agent_spatial_predictions_<JOB_ID>.json
```

The exact input data paths, model checkpoint paths, and output directory should be configured locally or via environment variables.

### 2. Evaluate hybrid-agent predictions

After inference has finished, compute the final metrics:

```bash
export PROJECT_ROOT=/path/to/repo
export JOB_ID=run

python code/evaluation/final_evaluate_agent.py
```

This computes accuracy, precision, recall, F1, invalid outputs, and wrong cases.

### 3. Run failure attribution

To compute the stage-wise failure attribution used in the paper:

```bash
export JOB_ID=run
export PROJECT_ROOT=/path/to/repo

python code/evaluation/final_evaluate_agent_failure_attribution.py
```

This script expects the hybrid-agent prediction files from the inference step and writes enriched wrong cases, evaluation metrics, and LaTeX table rows.

### 4. Run direct VLM baselines

Direct VLM baselines are evaluated separately under the unified binary prompt protocol:

```bash
python code/direct_vlm/eval_qwen_direct.py
python code/direct_vlm/eval_medgemma_direct.py
```

These scripts generate prediction files for the direct Qwen2-VL and MedGemma baselines. The resulting outputs can then be summarized using the corresponding evaluation scripts.

## Data and checkpoints

The original medical images, segmentation masks, benchmark data, and trained model checkpoints are not included in this repository.

Expected data formats are described in the main repository README and in the documentation files.