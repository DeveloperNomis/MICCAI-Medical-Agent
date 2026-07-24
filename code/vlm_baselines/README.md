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

The direct VLM environment requires a CUDA-capable Linux setup compatible with vLLM. Depending on the local CUDA driver, GPU type, PyTorch version, and vLLM version, users may need to adapt the PyTorch or vLLM installation commands.

## Input data

Set the local MIRP paths:

```bash
export IMAGE_DIR=/path/to/MIRP_Benchmark/RQ1/images
export QA_FILE=/path/to/MIRP_Benchmark/RQ1/qa.json
export OUTPUT_DIR=/path/to/output/vlm_baselines

mkdir -p "$OUTPUT_DIR"
```

## Model loading options

The script loads models through vLLM. Different VLMs may require different loading options.

Important arguments:

| Argument | Meaning |
|---|---|
| `--model_path` | Hugging Face model identifier or local checkpoint path |
| `--model_name` | Name used for output folders and result files |
| `--tokenizer_mode` | Tokenizer mode passed to vLLM |
| `--dtype` | Model dtype, for example `auto`, `float16`, or `bfloat16` |
| `--trust_remote_code` | Enables custom model code when required |
| `--max_tokens` | Maximum number of generated tokens |
| `--temperature` | Sampling temperature |

For the reported direct VLM baselines, the output should be deterministic:

```bash
--temperature 0
--max_tokens 1
```

The tokenizer mode is model- and vLLM-version-dependent. Start with:

```bash
--tokenizer_mode auto
```

If Qwen2-VL does not load correctly in the local vLLM version, retry with:

```bash
--tokenizer_mode qwen_vl
```

For MedGemma, start with:

```bash
--tokenizer_mode auto
```

If the selected model requires a custom processor, chat template, image formatting, tokenizer mode, or model-specific vLLM options, adapt the model-loading section inside:

```text
code/vlm_baselines/run_vlm_baseline.py
```

## Run Qwen2-VL baseline

Example using a Hugging Face model identifier:

```bash
python code/vlm_baselines/run_vlm_baseline.py \
  --model_path Qwen/Qwen2-VL-7B-Instruct \
  --image_dir "$IMAGE_DIR" \
  --qa_file "$QA_FILE" \
  --output_root "$OUTPUT_DIR" \
  --model_name qwen2vl_direct \
  --tokenizer_mode auto \
  --temperature 0 \
  --max_tokens 1 \
  --dtype auto \
  --trust_remote_code
```

If this does not load correctly with the installed vLLM version, retry with:

```bash
--tokenizer_mode qwen_vl
```

Example using a local checkpoint:

```bash
python code/vlm_baselines/run_vlm_baseline.py \
  --model_path /path/to/Qwen2-VL-7B-Instruct \
  --image_dir "$IMAGE_DIR" \
  --qa_file "$QA_FILE" \
  --output_root "$OUTPUT_DIR" \
  --model_name qwen2vl_direct \
  --tokenizer_mode auto \
  --temperature 0 \
  --max_tokens 1 \
  --dtype auto \
  --trust_remote_code
```

## Run MedGemma baseline

Example using a Hugging Face model identifier:

```bash
python code/vlm_baselines/run_vlm_baseline.py \
  --model_path google/medgemma-4b-it \
  --image_dir "$IMAGE_DIR" \
  --qa_file "$QA_FILE" \
  --output_root "$OUTPUT_DIR" \
  --model_name medgemma_direct \
  --tokenizer_mode auto \
  --temperature 0 \
  --max_tokens 1 \
  --dtype auto \
  --trust_remote_code
```

Example using a local checkpoint:

```bash
python code/vlm_baselines/run_vlm_baseline.py \
  --model_path /path/to/medgemma-4b-it \
  --image_dir "$IMAGE_DIR" \
  --qa_file "$QA_FILE" \
  --output_root "$OUTPUT_DIR" \
  --model_name medgemma_direct \
  --tokenizer_mode auto \
  --temperature 0 \
  --max_tokens 1 \
  --dtype auto \
  --trust_remote_code
```

## Changing models

To evaluate another VLM, change at least:

```bash
--model_path /path/to/model-or-huggingface-id
--model_name custom_model_name
```

Depending on the model, also check whether the following arguments need to be changed:

```bash
--tokenizer_mode auto
--dtype auto
--trust_remote_code
```

Some models may require additional changes inside:

```text
code/vlm_baselines/run_vlm_baseline.py
```

Typical model-specific changes include:

- tokenizer mode,
- chat template,
- image input format,
- processor behavior,
- maximum context length,
- vLLM backend arguments.

## Evaluate VLM outputs

After inference, evaluate the generated prediction files:

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

Generated predictions and evaluation outputs are not included in this repository.
