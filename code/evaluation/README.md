# Evaluation scripts

This folder contains evaluation scripts for hybrid-agent outputs.

## Evaluate hybrid-agent predictions

```bash
export ORG_GT_PATH=/path/to/MIRP_Benchmark/RQ1/qa.json
export IMG_DIR=/path/to/MIRP_Benchmark/RQ1/images
export PREDICTIONS_JSONL=/path/to/output/hybrid_agent/agent_spatial_predictions_<JOB_ID>.jsonl
export OUTPUT_DIR=/path/to/output/evaluation

python code/evaluation/final_evaluate_agent.py
```

## Failure attribution

```bash
export ORG_GT_PATH=/path/to/MIRP_Benchmark/RQ1/qa.json
export IMG_DIR=/path/to/MIRP_Benchmark/RQ1/images
export PREDICTIONS_JSONL=/path/to/output/hybrid_agent/agent_spatial_predictions_<JOB_ID>.jsonl
export OUTPUT_DIR=/path/to/output/evaluation

python code/evaluation/final_evaluate_agent_failure_attribution.py
```

Invalid, missing, failed, or non-binary outputs are counted as incorrect under strict evaluation.