# Modular Medical Imaging Agent for Spatial Reasoning in CT Scans

Code for the MICCAI Agent Workshop submission:

**A Modular Medical Imaging Agent for Reliable and Auditable Spatial Reasoning in CT Scans**

This repository implements a modular medical imaging agent for binary spatial relation verification in axial CT slices.

Instead of asking a vision-language model (VLM) to directly answer spatial questions, the system decomposes the task into explicit and auditable stages:

1. language parsing,
2. anatomical object detection,
3. deterministic geometric verification,
4. stage-wise audit logging.

Given a CT slice and a question such as:

```text
Is the liver left of the spleen?
```

the agent extracts a structured relation tuple, localizes the queried anatomical structures with a YOLO-based detector, and computes the final binary answer from object center coordinates.

The final spatial decision is deterministic and does not rely on a language model directly predicting the spatial truth value.

---

## Overview

The repository supports three main evaluation settings:

| Setting | Description |
|---|---|
| Direct VLM baseline | MedGemma or Qwen2-VL receives the image and prompt directly and outputs `0` or `1`. |
| Hybrid agent | The same external prompt is passed to the agent. The VLM is used for language/query handling, while YOLO + geometry computes the final answer. |
| YOLO + geometry | Gold structured queries are provided directly to evaluate the perception-and-geometry core. |

All VLM-based systems use the same external prompt:

```text
This is a 2D axial CT slice.

Question: {question}

Answer the question with exactly one character:
1 if the statement is true.
0 if the statement is false.

Do not output any explanation.
```

The key idea is that the external interface remains comparable to a conventional VLM, while the hybrid agent internally performs explicit tool-based reasoning.

---

## Pipeline

The spatial verification pathway consists of four stages:

### 1. Question parsing

The input question is converted into a structured relation tuple.

Example:

```text
Is the liver left of the spleen?
```

becomes:

```json
{
  "entity_a": "liver",
  "relation": "left of",
  "entity_b": "spleen"
}
```

### 2. Ontology matching

Extracted entity names are mapped to the anatomical class names used by the detector.

### 3. Anatomical localization

A YOLO-based detector localizes the queried anatomical structures in the CT slice.

For each queried class, the highest-confidence detection is selected.

### 4. Deterministic geometric verification

Spatial relations are evaluated from object center coordinates.

For example:

```text
A is left of B  ->  center_x(A) < center_x(B)
A is above B    ->  center_y(A) < center_y(B)
```

The final answer is computed deterministically from these coordinates.

---

## Installation

We recommend Python 3.10+ and a CUDA-capable GPU.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

If no `requirements.txt` is available yet, install the main dependencies manually:

```bash
pip install torch torchvision ultralytics transformers vllm langchain spacy tqdm numpy opencv-python pyyaml
```

---

## Data

The evaluation uses the held-out MIRP spatial QA benchmark.

Expected data layout:

```text
data/
  images/
    test/
      *.png
  annotations/
    organs_gt.json
  yolo/
    data.yaml
```

Segmentation masks are used only to derive bounding-box supervision and reference locations.  
They are not used during inference.

---

## Pretrained Models

The following model files are required:

| Component | File / model |
|---|---|
| YOLO detector | `models/yolo/best.pt` |
| MedGemma | local or Hugging Face checkpoint |
| Qwen2-VL | local or Hugging Face checkpoint |

Place detector weights under:

```text
models/yolo/best.pt
```

### MedGemma vLLM compatibility note

For MedGemma, the local `config.json` may require the missing sliding-window configuration field expected by the vLLM backend.

This affects only backend compatibility.  
Model weights, prompts, decoding settings, and evaluation data remain unchanged.

Example:

```json
"sliding_window_pattern": 6,
"_sliding_window_pattern": 6
```

---

## Running the Hybrid Agent

Run a single spatial query:

```bash
python hybrid_agent/main.py \
  --image data/images/test/example.png \
  --question "Is the liver left of the spleen?" \
  --yolo_weights models/yolo/best.pt
```

Example output:

```json
{
  "answer": "1",
  "parsed_query": {
    "entity_a": "liver",
    "relation": "left of",
    "entity_b": "spleen"
  },
  "audit": {
    "question_extraction": "success",
    "routing": "spatial",
    "ontology_matching": "success",
    "detection": "success",
    "geometry": "center_x(liver) < center_x(spleen)"
  }
}
```

---

## Evaluation

### 1. Direct VLM baselines

```bash
python eval/run_vlm_baseline.py \
  --model_path /path/to/model \
  --data_path data/ \
  --output_root results/vlm_baselines \
  --batch_size 4
```

This evaluates a VLM end-to-end:

```text
CT image + prompt -> 0/1 answer
```

### 2. Hybrid agent evaluation

```bash
python eval/run_hybrid_agent.py \
  --data_path data/ \
  --yolo_weights models/yolo/best.pt \
  --language_model /path/to/model \
  --output_root results/hybrid_agent
```

This evaluates the full agent:

```text
CT image + same prompt -> parsing -> YOLO -> geometry -> 0/1 answer
```

### 3. YOLO + geometry core evaluation

```bash
python eval/run_yolo_geometry.py \
  --data_path data/ \
  --yolo_weights models/yolo/best.pt \
  --output_root results/yolo_geometry
```

This evaluates the perception-and-geometry core using gold structured queries.

---

## Results

Main results under the unified prompt protocol:

| System | Accuracy | F1 | Invalid |
|---|---:|---:|---:|
| MedGemma direct | `<fill>` | `<fill>` | `<fill>` |
| MedGemma + hybrid agent | `<fill>` | `<fill>` | `<fill>` |
| Qwen2-VL direct | `<fill>` | `<fill>` | `<fill>` |
| Qwen2-VL + hybrid agent | `<fill>` | `<fill>` | `<fill>` |
| YOLO + geometry | `<fill>` | `<fill>` | -- |

Strict evaluation counts invalid, failed, or non-binary outputs as incorrect.

---

## Failure Attribution

The hybrid agent stores intermediate information for each case, enabling stage-wise error analysis.

| Failure stage | Description |
|---|---|
| Question extraction | The full prompt could not be converted into the core spatial question. |
| Routing | The query was routed to the wrong task pathway. |
| Parsing | The wrong entity or relation was extracted. |
| Ontology matching | An extracted entity could not be mapped to a detector class. |
| Missing detection | At least one queried anatomical structure was not detected. |
| Localization | The detected center was too imprecise for the relation. |
| Geometry | The spatial configuration was borderline or ambiguous. |
| Formatting/runtime | The system produced an invalid output or failed during execution. |

This allows failures to be localized to individual modules instead of being hidden inside a black-box model response.

---

## Training the YOLO Detector

To train the detector from bounding-box annotations:

```bash
python train_yolo.py \
  --data data/yolo/data.yaml \
  --model yolov8m.pt \
  --epochs 500 \
  --imgsz 512 \
  --batch 128 \
  --freeze 2
```

The detector is trained only on bounding boxes derived from anatomical segmentation masks.

No spatial-relation labels are used for detector training.

---

## Reproducibility Notes

- All VLM baselines use the same external prompt.
- Decoding is deterministic with `temperature=0`.
- Invalid or non-binary outputs are counted as incorrect under strict evaluation.
- The hybrid agent receives the same user-facing prompt as the direct VLM baselines.
- The final spatial decision in the hybrid agent is computed from YOLO detections and deterministic geometry.
- Segmentation masks are not used during inference.
- Patient-level separation is enforced between detector training data and the held-out test set.
- The isolated YOLO + geometry setting is reported separately to distinguish perception-and-geometry performance from full-agent integration errors.

---

## Repository Structure

```text
hybrid_agent/
  agent/
    controller.py
    router.py

  language_parsing/
    ...
    
  perception/
    yolo_model.py
    deterministic_relations.py

  tasks/
    spatial_task.py
    presence_task.py

eval/
  run_vlm_baseline.py
  run_hybrid_agent.py
  run_yolo_geometry.py

models/
  yolo/
    best.pt

data/
  images/
  annotations/
  yolo/

docs/
  method_details.md
  data_preparation.md
  training.md
  evaluation.md

train_yolo.py
requirements.txt
README.md
```

---

## Additional Documentation

Detailed documentation can be placed in:

| File | Content |
|---|---|
| `docs/method_details.md` | Full method description and design motivation |
| `docs/data_preparation.md` | Dataset generation, preprocessing, and patient-wise splitting |
| `docs/training.md` | YOLO training details and hyperparameters |
| `docs/evaluation.md` | Evaluation protocol and metric definitions |
| `docs/legacy_experiments.md` | Additional exploratory experiments not used in the main paper |
| `docs/hpc.md` | SLURM and cluster-specific execution notes |

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{anonymous2026modular,
  title={A Modular Medical Imaging Agent for Reliable and Auditable Spatial Reasoning in CT Scans},
  author={Anonymous},
  booktitle={MICCAI Agent Workshop},
  year={2026}
}
```

---

## License

This repository is released for research use.  
Please check the licenses of all external datasets and pretrained models before use.

---

## Acknowledgements

This work builds on open-source tools including YOLOv8, vLLM, Transformers, LangChain, and spaCy.
