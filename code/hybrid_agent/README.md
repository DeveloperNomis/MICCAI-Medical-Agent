# Hybrid agent

This folder contains the final hybrid-agent implementation used for spatial relation verification.

## Required local files

Before running the agent, provide:

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

## Single-case demo

```bash
export IMAGE_PATH=/path/to/example_image.png
export QUESTION="Is the liver left of the spleen?"
export OUTPUT_DIR=/path/to/output/single_case

python code/hybrid_agent/main.py
```

This is intended for debugging and does not compute the paper metrics.

## Benchmark inference

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

## Notes

- No Slurm script is required.
- The final spatial answer is computed from YOLO detections and deterministic geometry.
- The language model is used for routing, query handling, and optional fallback parsing.
- Segmentation masks are not used during inference.