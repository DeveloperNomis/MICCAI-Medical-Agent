#!/usr/bin/env python3
"""
Direct VLM baseline inference for binary spatial QA on CT slices.

This script evaluates a vision-language model directly on image-question pairs.
The model receives the CT slice and the spatial question and must answer with
exactly one binary character:

    1 = Yes
    0 = No

No detector, tool use, or deterministic geometry is used in this baseline.

Supported input layouts
-----------------------

1. Flat benchmark layout:

    --image_dir /path/to/images
    --qa_file /path/to/qa.json

2. MIRP-style experiment layout:

    --data_path /path/to/benchmark
    --experiments RQ1 RQ2 RQ3

Expected QA format
------------------

The QA JSON file is expected to contain entries of the form:

[
  {
    "filename": "example.png",
    "question_answer": [
      {
        "question": "Is the liver left of the spleen?",
        "answer": 1
      }
    ]
  }
]

Outputs are written to --output_root and are not meant to be committed to Git.
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from vllm import LLM
from vllm.sampling_params import SamplingParams


def register_optional_mistral_aliases() -> None:
    """
    Register optional Mistral-family config aliases used by some VLM checkpoints.

    This is kept for compatibility with models whose Hugging Face configuration
    uses one of these aliases. It has no effect if the aliases are already known.
    """
    try:
        from transformers import MistralConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    except Exception:
        return

    for name in ("ministral", "ministral3", "mistral3"):
        if name not in CONFIG_MAPPING:
            CONFIG_MAPPING.register(name, MistralConfig)


def encode_image_to_base64(image: Image.Image) -> str:
    """
    Encode a PIL image as a base64 PNG string.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def normalize_16bit_to_8bit(image: Image.Image) -> Image.Image:
    """
    Convert a 16-bit grayscale image to 8-bit grayscale.
    """
    return image.point(lambda x: x / 256).convert("L")


def ensure_rgb(image: Image.Image) -> Image.Image:
    """
    Convert an input image to RGB.

    CT slices are often stored as grayscale images. Most VLM chat interfaces
    expect RGB images, so grayscale and 16-bit images are converted here.
    """
    if image.mode == "I;16":
        return normalize_16bit_to_8bit(image).convert("RGB")
    if image.mode in {"L", "LA", "RGBA"}:
        return image.convert("RGB")
    if image.mode == "RGB":
        return image.copy()

    raise ValueError(f"Unsupported image mode: {image.mode}")


def load_image_as_base64(image_path: Path) -> str:
    """
    Load an image from disk and return a base64-encoded PNG representation.
    """
    with Image.open(image_path) as image:
        rgb_image = ensure_rgb(image)
    return encode_image_to_base64(rgb_image)


def load_qa_entries(qa_file: Path) -> List[Dict[str, Any]]:
    """
    Load QA entries from a JSON file.
    """
    with qa_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of QA entries in {qa_file}")

    return data


def get_questions_for_image(
    image_filename: str,
    qa_entries: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Return all question-answer pairs associated with one image filename.
    """
    for entry in qa_entries:
        if entry.get("filename") == image_filename:
            qa_pairs = entry.get("question_answer", [])
            return [
                {
                    "question": qa.get("question"),
                    "answer": qa.get("answer"),
                }
                for qa in qa_pairs
                if qa.get("question") is not None
            ]

    return []


def build_binary_prompt(question: str) -> str:
    """
    Build the strict binary prompt used for direct VLM baselines.
    """
    return (
        "This is a 2D axial CT slice.\n\n"
        f"Question: {question}\n\n"
        "Answer the question with exactly one character:\n"
        "1 if the statement is true.\n"
        "0 if the statement is false.\n\n"
        "Do not output any explanation."
    )


def parse_binary_answer(text: str) -> Optional[str]:
    """
    Parse a strict binary answer.

    Returns "0" or "1" only if the stripped output is exactly binary.
    Otherwise returns None.
    """
    cleaned = str(text).strip()
    if cleaned in {"0", "1"}:
        return cleaned
    return None


def make_model_call(
    llm: LLM,
    sampling_params: SamplingParams,
    question_data: Dict[str, Any],
    base64_image: str,
    log_vision_check: bool = False,
) -> Dict[str, Any]:
    """
    Run one VLM call for one image-question pair.
    """
    prompt = build_binary_prompt(str(question_data["question"]))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    },
                },
            ],
        }
    ]

    outputs = llm.chat(messages, sampling_params=sampling_params)
    response = outputs[0]
    raw_answer = response.outputs[0].text
    parsed_answer = parse_binary_answer(raw_answer)

    if log_vision_check:
        total_tokens = len(getattr(response, "prompt_token_ids", []) or [])
        vision_tokens = [
            token
            for token in getattr(response, "prompt_tokens", []) or []
            if isinstance(token, dict) and token.get("modality") == "vision"
        ]

        print(f"Model: {llm.llm_engine.model_config.model}")
        print(f"Prompt tokens: {total_tokens}")

        if vision_tokens:
            print(f"Vision-token check: found {len(vision_tokens)} explicit vision tokens.")
        elif total_tokens > 200:
            print("Vision-token check: prompt token count indicates image input was included.")
        else:
            print("Warning: no vision tokens detected. The model may have received text only.")

    return {
        "question": question_data["question"],
        "model_answer": raw_answer,
        "parsed_answer": parsed_answer,
        "expected_answer": question_data.get("answer"),
        "entire_prompt": prompt,
    }


def get_experiment_plan(experiment: str) -> Dict[str, Dict[str, str]]:
    """
    Return the MIRP-style image/QA layout for an experiment name.
    """
    if experiment == "RQ1":
        return {
            "sub_experiment_1": {
                "img": "images",
                "qa": "qa.json",
            }
        }

    if experiment in {"RQ2", "RQ3"}:
        return {
            "sub_experiment_1": {
                "img": "images_numbers",
                "qa": "qa_numbers.json",
            },
            "sub_experiment_2": {
                "img": "images_letters",
                "qa": "qa_letters.json",
            },
            "sub_experiment_3": {
                "img": "images_dots",
                "qa": "qa_dots.json",
            },
        }

    raise ValueError(f"Unknown experiment: {experiment}")


def collect_flat_job(image_dir: Path, qa_file: Path) -> List[Tuple[str, Path, Path]]:
    """
    Create one inference job from a flat image directory and one QA file.
    """
    return [("flat", image_dir, qa_file)]


def collect_mirp_jobs(
    data_path: Path,
    experiments: Sequence[str],
) -> List[Tuple[str, Path, Path]]:
    """
    Create inference jobs from a MIRP-style benchmark directory.
    """
    jobs: List[Tuple[str, Path, Path]] = []

    for experiment in experiments:
        experiment_dir = data_path / experiment
        experiment_plan = get_experiment_plan(experiment)

        for sub_experiment, layout in experiment_plan.items():
            image_dir = experiment_dir / layout["img"]
            qa_file = experiment_dir / layout["qa"]
            job_name = f"{experiment}/{sub_experiment}"
            jobs.append((job_name, image_dir, qa_file))

    return jobs


def select_image_filenames(
    qa_entries: Sequence[Dict[str, Any]],
    sample_size: Optional[int],
    seed: int,
) -> Tuple[List[str], str]:
    """
    Select image filenames from the QA file.

    If sample_size is None, all images are used.
    """
    image_filenames = [
        entry["filename"]
        for entry in qa_entries
        if isinstance(entry, dict) and "filename" in entry
    ]

    if sample_size is None or sample_size >= len(image_filenames):
        print(f"Using all {len(image_filenames)} images.")
        return image_filenames, "all_images"

    if sample_size <= 0:
        raise ValueError("--sample_size must be positive when provided.")

    print(f"Using a random sample of {sample_size} images from {len(image_filenames)} images.")

    rng = random.Random(seed)
    sampled = rng.sample(image_filenames, sample_size)
    return sampled, f"random_sample_{sample_size}_images"


def run_inference_job(
    llm: LLM,
    sampling_params: SamplingParams,
    model_name: str,
    job_name: str,
    image_dir: Path,
    qa_file: Path,
    output_root: Path,
    sample_size: Optional[int],
    num_runs: int,
    seed: int,
    log_vision_check: bool,
) -> None:
    """
    Run direct VLM inference for one image directory and one QA file.
    """
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not qa_file.exists():
        raise FileNotFoundError(f"QA file not found: {qa_file}")

    qa_entries = load_qa_entries(qa_file)
    image_filenames, sample_tag = select_image_filenames(qa_entries, sample_size, seed)

    safe_job_name = job_name.replace("/", "_")
    target_dir = output_root / model_name / safe_job_name
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Job: {job_name}")
    print(f"Images: {image_dir}")
    print(f"QA file: {qa_file}")
    print(f"Output directory: {target_dir}")

    for run_idx in range(num_runs):
        start_time = time.time()
        dataset_results = []

        for image_filename in image_filenames:
            image_path = image_dir / image_filename
            if not image_path.exists():
                print(f"Warning: image not found, skipping: {image_path}")
                continue

            qa_pairs = get_questions_for_image(image_filename, qa_entries)
            if not qa_pairs:
                print(f"Warning: no QA pairs found, skipping: {image_filename}")
                continue

            base64_image = load_image_as_base64(image_path)

            for qa in qa_pairs:
                result = make_model_call(
                    llm=llm,
                    sampling_params=sampling_params,
                    question_data=qa,
                    base64_image=base64_image,
                    log_vision_check=log_vision_check,
                )

                dataset_results.append(
                    {
                        "model": model_name,
                        "file_name": image_filename,
                        "results_call": [result],
                    }
                )

        output_file = target_dir / (
            f"{model_name}__{qa_file.stem}_{sample_tag}_run_{run_idx}.json"
        )

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(dataset_results, f, indent=2, ensure_ascii=False)

        elapsed_time = time.time() - start_time
        print(f"Saved {len(dataset_results)} predictions to: {output_file}")
        print(f"Runtime for run {run_idx}: {elapsed_time:.2f} seconds")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run direct VLM baseline inference for binary spatial QA."
    )

    parser.add_argument(
        "--model_path",
        required=True,
        help="Path or Hugging Face identifier of the VLM checkpoint.",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Name used in output paths. Defaults to the last part of --model_path.",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--data_path",
        help="Root directory of a MIRP-style benchmark layout.",
    )
    input_group.add_argument(
        "--image_dir",
        help="Directory containing images for a flat benchmark layout.",
    )

    parser.add_argument(
        "--qa_file",
        default=None,
        help="QA JSON file for the flat benchmark layout. Required with --image_dir.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["RQ1"],
        help="MIRP-style experiments to run when --data_path is used.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Directory where prediction JSON files are written.",
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Optional number of images to sample. By default, all images are used.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of repeated runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for optional image sampling.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Model dtype passed to vLLM.",
    )
    parser.add_argument(
        "--tokenizer_mode",
        default="auto",
        help="Tokenizer mode passed to vLLM.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=32000,
        help="Maximum model context length passed to vLLM.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization passed to vLLM.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Enable trust_remote_code for model loading.",
    )
    parser.add_argument(
        "--register_mistral_aliases",
        action="store_true",
        help="Register optional Mistral-family config aliases for compatibility.",
    )
    parser.add_argument(
        "--log_vision_check",
        action="store_true",
        help="Print a lightweight check indicating whether image tokens were included.",
    )

    args = parser.parse_args()

    if args.image_dir and not args.qa_file:
        parser.error("--qa_file is required when --image_dir is used.")

    return args


def main() -> None:
    """
    Entry point.
    """
    args = parse_args()

    if args.register_mistral_aliases:
        register_optional_mistral_aliases()

    model_path = Path(args.model_path)
    model_name = args.model_name or model_path.name or "direct_vlm"
    output_root = Path(args.output_root)

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print("Loading model...")
    print(f"Model path: {args.model_path}")
    print(f"Model name: {model_name}")

    llm = LLM(
        model=args.model_path,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        limit_mm_per_prompt={"image": 1},
    )

    if args.image_dir:
        jobs = collect_flat_job(
            image_dir=Path(args.image_dir),
            qa_file=Path(args.qa_file),
        )
    else:
        jobs = collect_mirp_jobs(
            data_path=Path(args.data_path),
            experiments=args.experiments,
        )

    print(f"Number of inference jobs: {len(jobs)}")

    for job_name, image_dir, qa_file in jobs:
        run_inference_job(
            llm=llm,
            sampling_params=sampling_params,
            model_name=model_name,
            job_name=job_name,
            image_dir=image_dir,
            qa_file=qa_file,
            output_root=output_root,
            sample_size=args.sample_size,
            num_runs=args.num_runs,
            seed=args.seed,
            log_vision_check=args.log_vision_check,
        )

    print("Direct VLM inference finished.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)
