# VLM baselines

This folder contains scripts for direct Vision-Language Model baselines.

The direct VLM baselines receive the CT slice and question directly and must output exactly one binary character:

```text
1 = Yes
0 = No
```

No YOLO detector, tool use, or deterministic geometry is used in this setting.

## Installation

Use the separate VLM environment:

```bash
python -m venv .venv-vlm
source .venv-vlm/bin/activate

pip install --upgrade pip
pip install -r requirements-vlm.txt
```

The direct VLM environment requires a CUDA-capable Linux setup compatible with vLLM. Depending on the local CUDA driver and hardware, users may need to adapt the PyTorch or vLLM installation commands.

## Input data

Set the local MIRP paths:

```bash
export IMAGE_DIR=/path/to/MIRP_Benchmark/RQ1/images
export QA_FILE=/path/to/MIRP_Benchmark/RQ1/qa.json
export OUTPUT_DIR=/path/to/output/vlm_baselines
```

## Run Qwen2-VL direct baseline

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

## Run MedGemma direct baseline

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

## Changing models

To evaluate another VLM, change:

```bash
--model_path /path/to/model-or-huggingface-id
--model_name custom_model_name
```

If the model requires a specific tokenizer mode, processor, chat template, or backend option, adapt the loading section inside:

```text
run_direct_vlm_baseline.py
```

## Evaluate direct VLM outputs

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

Generated predictions and evaluation outputs are not included in this repository.