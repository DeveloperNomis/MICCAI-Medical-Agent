#!/usr/bin/env python3
"""
Evaluate direct VLM baseline prediction files.

This script evaluates JSON or JSONL prediction files produced by direct VLM
baseline inference. It is designed for binary spatial relation verification
experiments where each model response should resolve to:

    1 = yes / true
    0 = no / false

Unparseable responses are counted as invalid and are treated as incorrect under
strict evaluation by assigning the opposite label of the target.

Expected JSON format:
[
    {
        "model": "model_name",
        "file_name": "image_001.png",
        "results_call": [
            {
                "question": "Is the liver left of the spleen?",
                "model_answer": "1",
                "expected_answer": 1,
                "entire_prompt": "..."
            }
        ]
    }
]

The script writes CSV, Excel, and JSON summaries to the chosen output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openpyxl import Workbook
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


###############################################################################
# Answer parsing
###############################################################################


def parse_spatial_relation(question: str, answer: str) -> Optional[int]:
    """
    Legacy fallback for short directional answers.

    If the model answers with a short phrase such as "left" or "below" rather
    than a binary digit, this function maps it to 1 or 0 when the direction can
    be unambiguously compared to the relation mentioned in the question.
    """
    q_lower = question.lower()
    a_lower = answer.lower()

    directions = [
        ("above", "below"),
        ("below", "above"),
        ("left", "right"),
        ("right", "left"),
    ]

    for queried_direction, opposite_direction in directions:
        if queried_direction in q_lower:
            if queried_direction in a_lower:
                return 1
            if opposite_direction in a_lower:
                return 0

    return None


def parse_model_answer(
    model_answer: str,
    question: str,
    entire_prompt: str,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Parse a model response into a binary prediction.

    Returns:
        A tuple `(prediction, cleaned_text)` where prediction is 0, 1, or None.
        `cleaned_text` is populated only when prompt-removal fallback was used.
    """

    def single_digit_ignoring_punctuation(text: str) -> Optional[str]:
        pattern = re.compile(r"^[\(\[\{\'\"\.\s]*(0|1)[\)\]\}\'\"\.\s]*$")
        match = pattern.match(text.strip())
        return match.group(1) if match else None

    def single_yes_no_ignoring_punctuation(text: str) -> Optional[str]:
        pattern = re.compile(r"^[\(\[\{\'\"\.\s]*(yes|no)[\)\]\}\'\"\.\s]*$", re.IGNORECASE)
        match = pattern.match(text.strip())
        return match.group(1).lower() if match else None

    def starts_or_ends_with_binary_token(text: str) -> Optional[str]:
        text = text.strip()

        start_pattern = re.compile(r"^[\(\[\{\'\"\.\s]*(0|1|yes|no)", re.IGNORECASE)
        start_match = start_pattern.match(text)
        if start_match:
            return start_match.group(1).lower()

        end_pattern = re.compile(r"(0|1|yes|no)[\)\]\}\'\"\.\s]*$", re.IGNORECASE)
        end_match = end_pattern.search(text)
        if end_match:
            return end_match.group(1).lower()

        return None

    def is_single_short_sentence(text: str) -> bool:
        text = text.strip()
        if "\n" in text:
            return False
        if len(re.findall(r"[.!?]", text)) > 1:
            return False
        return len(text) <= 150

    text = str(model_answer or "").strip()

    maybe_digit = single_digit_ignoring_punctuation(text)
    if maybe_digit in {"0", "1"}:
        return int(maybe_digit), None

    maybe_yes_no = single_yes_no_ignoring_punctuation(text)
    if maybe_yes_no == "yes":
        return 1, None
    if maybe_yes_no == "no":
        return 0, None

    boundary_token = starts_or_ends_with_binary_token(text)
    if boundary_token in {"1", "yes"}:
        return 1, None
    if boundary_token in {"0", "no"}:
        return 0, None

    if is_single_short_sentence(text):
        spatial_prediction = parse_spatial_relation(question, text)
        if spatial_prediction is not None:
            return spatial_prediction, None

    cleaned_text = text
    for prompt_line in str(entire_prompt or "").splitlines():
        prompt_line = prompt_line.strip()
        if prompt_line:
            while prompt_line in cleaned_text:
                cleaned_text = cleaned_text.replace(prompt_line, "")

    first_sentence_match = re.search(r"[^.!?]+[.!?]?", cleaned_text)
    if first_sentence_match:
        first_sentence = first_sentence_match.group(0).strip()
        spatial_prediction = parse_spatial_relation(question, first_sentence)
        if spatial_prediction is not None:
            return spatial_prediction, cleaned_text

    answer_pattern = re.compile(
        r"""
        (?:
            \banswer\b
            | \bcorrect\s+answer\b
            | \bfinal\s+answer\b
            | \bsolution\b
            | \bresponse\b
            | \bthe\s+answer\b
            | \bthe\s+correct\s+answer\b
            | \bthe\s+final\s+answer\b
            | \btherefore\s+the\s+answer\b
            | \bhence\s+the\s+answer\b
        )
        (?:\s+is\s*|\s*:\s*|\s+)(?:["']?)
        ([10])
        (?:["']?\b)
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    match = answer_pattern.search(cleaned_text)
    if match:
        return int(match.group(1)), cleaned_text

    return None, cleaned_text


###############################################################################
# Data loading and evaluation
###############################################################################


def normalize_binary_label(value: Any) -> int:
    """
    Convert a target label to integer 0 or 1.
    """
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)) and value in {0, 1}:
        return int(value)

    text = str(value).strip().lower()
    if text in {"1", "yes", "true", "t"}:
        return 1
    if text in {"0", "no", "false", "f"}:
        return 0

    raise ValueError(f"Expected a binary label, got: {value!r}")


def load_prediction_file(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSON or JSONL prediction file.
    """
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if "predictions" in data and isinstance(data["predictions"], list):
            return data["predictions"]
        if "results" in data and isinstance(data["results"], list):
            return data["results"]

    raise ValueError(f"Unsupported prediction file format: {path}")


def iter_result_calls(entry: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Yield result-call dictionaries from supported prediction entry formats.
    """
    if "results_call" in entry and isinstance(entry["results_call"], list):
        yield from entry["results_call"]
        return

    # Support a flatter record layout if the inference script writes one case per line.
    if {"question", "model_answer", "expected_answer"}.issubset(entry.keys()):
        yield entry


def evaluate_prediction_file(path: Path) -> Dict[str, Any]:
    """
    Evaluate one prediction file and return metrics plus invalid cases.
    """
    data = load_prediction_file(path)

    all_predictions: List[int] = []
    all_targets: List[int] = []

    left_right_predictions: List[int] = []
    left_right_targets: List[int] = []
    above_below_predictions: List[int] = []
    above_below_targets: List[int] = []

    correct_count = 0
    incorrect_count = 0
    invalid_count = 0
    invalid_cases = []

    for entry in data:
        file_name = entry.get("file_name", entry.get("filename", ""))

        for result in iter_result_calls(entry):
            question = str(result.get("question", ""))
            model_answer = str(result.get("model_answer", result.get("prediction", "")))
            entire_prompt = str(result.get("entire_prompt", ""))
            expected = normalize_binary_label(result.get("expected_answer", result.get("answer")))

            parsed, cleaned_text = parse_model_answer(model_answer, question, entire_prompt)

            if parsed is None:
                prediction = 1 - expected
                invalid_count += 1
                invalid_case = {
                    "file_name": file_name,
                    "question": question,
                    "expected_answer": expected,
                    "model_answer": model_answer,
                }
                if cleaned_text:
                    invalid_case["cleaned_text"] = cleaned_text
                invalid_cases.append(invalid_case)
            else:
                prediction = parsed
                if cleaned_text:
                    result["cleaned_text"] = cleaned_text

            all_predictions.append(prediction)
            all_targets.append(expected)

            question_lower = question.lower()
            if "left" in question_lower or "right" in question_lower:
                left_right_predictions.append(prediction)
                left_right_targets.append(expected)

            if "above" in question_lower or "below" in question_lower:
                above_below_predictions.append(prediction)
                above_below_targets.append(expected)

            if prediction == expected:
                correct_count += 1
            else:
                incorrect_count += 1

    if not all_targets:
        raise ValueError(f"No evaluable predictions found in: {path}")

    return {
        "file_path": str(path),
        "num_cases": len(all_targets),
        "accuracy": accuracy_score(all_targets, all_predictions),
        "precision": precision_score(all_targets, all_predictions, zero_division=0),
        "recall": recall_score(all_targets, all_predictions, zero_division=0),
        "f1_score": f1_score(all_targets, all_predictions, zero_division=0),
        "accuracy_left_right": metric_or_nan(accuracy_score, left_right_targets, left_right_predictions),
        "f1_left_right": metric_or_nan(f1_score, left_right_targets, left_right_predictions),
        "accuracy_above_below": metric_or_nan(accuracy_score, above_below_targets, above_below_predictions),
        "f1_above_below": metric_or_nan(f1_score, above_below_targets, above_below_predictions),
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "invalid_count": invalid_count,
        "invalid_cases": invalid_cases,
        "all_predictions": all_predictions,
        "all_targets": all_targets,
    }


def metric_or_nan(metric_fn, targets: List[int], predictions: List[int]) -> float:
    """
    Return a metric value, or NaN if the subset is empty.
    """
    if not targets:
        return float("nan")
    return metric_fn(targets, predictions, zero_division=0) if metric_fn is f1_score else metric_fn(targets, predictions)


###############################################################################
# Aggregation and file collection
###############################################################################


def safe_stdev(values: List[float]) -> float:
    """
    Return the sample standard deviation, or 0 for a single value.
    """
    values = [v for v in values if not math.isnan(v)]
    if len(values) > 1:
        return stdev(values)
    return 0.0


def safe_mean(values: List[float]) -> float:
    """
    Return the mean of non-NaN values, or NaN if no valid values exist.
    """
    values = [v for v in values if not math.isnan(v)]
    return mean(values) if values else float("nan")


def extract_run_index(path: Path) -> int:
    """
    Extract a run index from file names ending in `_run_<N>.json` or `_run_<N>.jsonl`.
    """
    match = re.search(r"_run_(\d+)\.(json|jsonl)$", path.name, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def strip_run_suffix(path: Path) -> str:
    """
    Remove a trailing `_run_<N>` suffix from a prediction file name.
    """
    stem = path.stem
    return re.sub(r"_run_\d+$", "", stem, flags=re.IGNORECASE)


def should_skip_json_file(path: Path, output_dir: Path) -> bool:
    """
    Exclude generated evaluation summaries and invalid-case files from recursion.
    """
    name = path.name.lower()
    parts = {part.lower() for part in path.parts}

    if output_dir in path.parents or path == output_dir:
        return True
    if "unsure_cases" in parts or "invalid_cases" in parts or "evaluation" in parts:
        return True
    if name.startswith("result_"):
        return True
    if name.startswith("direct_vlm_evaluation_summary"):
        return True
    if name in {"results_images.json", "results_images.csv", "results_images.xlsx"}:
        return True

    return False


def collect_prediction_files(results_path: Path, output_dir: Path) -> List[Path]:
    """
    Recursively collect candidate JSON and JSONL prediction files.
    """
    files = []
    for pattern in ("*.json", "*.jsonl"):
        for path in results_path.rglob(pattern):
            if not should_skip_json_file(path, output_dir):
                files.append(path)
    return sorted(files)


def group_prediction_files(files: List[Path], results_path: Path) -> Dict[Tuple[str, str], List[Path]]:
    """
    Group prediction files by relative directory and base file name.
    """
    grouped: Dict[Tuple[str, str], List[Path]] = {}
    for path in files:
        relative_parent = str(path.parent.relative_to(results_path))
        if relative_parent == ".":
            relative_parent = "root"
        base_name = strip_run_suffix(path)
        grouped.setdefault((relative_parent, base_name), []).append(path)
    return grouped


def summarize_group(relative_parent: str, base_name: str, files: List[Path]) -> Dict[str, Any]:
    """
    Evaluate and aggregate all runs belonging to one grouped prediction set.
    """
    run_metrics = []
    for path in files:
        metrics = evaluate_prediction_file(path)
        metrics["run_idx"] = extract_run_index(path)
        run_metrics.append(metrics)

    run_metrics.sort(key=lambda item: item["run_idx"])

    def collect(key: str) -> List[float]:
        return [float(metrics[key]) for metrics in run_metrics]

    aggregated = {
        "num_runs": len(run_metrics),
        "num_cases_mean": safe_mean(collect("num_cases")),
        "accuracy_mean": safe_mean(collect("accuracy")),
        "accuracy_std": safe_stdev(collect("accuracy")),
        "precision_mean": safe_mean(collect("precision")),
        "precision_std": safe_stdev(collect("precision")),
        "recall_mean": safe_mean(collect("recall")),
        "recall_std": safe_stdev(collect("recall")),
        "f1_mean": safe_mean(collect("f1_score")),
        "f1_std": safe_stdev(collect("f1_score")),
        "accuracy_left_right_mean": safe_mean(collect("accuracy_left_right")),
        "accuracy_left_right_std": safe_stdev(collect("accuracy_left_right")),
        "f1_left_right_mean": safe_mean(collect("f1_left_right")),
        "f1_left_right_std": safe_stdev(collect("f1_left_right")),
        "accuracy_above_below_mean": safe_mean(collect("accuracy_above_below")),
        "accuracy_above_below_std": safe_stdev(collect("accuracy_above_below")),
        "f1_above_below_mean": safe_mean(collect("f1_above_below")),
        "f1_above_below_std": safe_stdev(collect("f1_above_below")),
        "correct_mean": safe_mean(collect("correct_count")),
        "incorrect_mean": safe_mean(collect("incorrect_count")),
        "invalid_mean": safe_mean(collect("invalid_count")),
    }

    invalid_cases = []
    for metrics in run_metrics:
        for case in metrics["invalid_cases"]:
            invalid_cases.append({
                "run_idx": metrics["run_idx"],
                "source_file": metrics["file_path"],
                **case,
            })

    compact_run_metrics = []
    for metrics in run_metrics:
        compact_run_metrics.append({
            key: value
            for key, value in metrics.items()
            if key not in {"all_predictions", "all_targets", "invalid_cases"}
        })

    return {
        "relative_parent": relative_parent,
        "base_name": base_name,
        "run_metrics": compact_run_metrics,
        "aggregated": aggregated,
        "invalid_cases": invalid_cases,
    }


###############################################################################
# Output writers
###############################################################################


SUMMARY_COLUMNS = [
    "Relative Directory",
    "Base Name",
    "Num Runs",
    "Num Cases Mean",
    "Accuracy Mean",
    "Accuracy Std",
    "Precision Mean",
    "Precision Std",
    "Recall Mean",
    "Recall Std",
    "F1 Mean",
    "F1 Std",
    "Accuracy Left/Right Mean",
    "Accuracy Left/Right Std",
    "F1 Left/Right Mean",
    "F1 Left/Right Std",
    "Accuracy Above/Below Mean",
    "Accuracy Above/Below Std",
    "F1 Above/Below Mean",
    "F1 Above/Below Std",
    "Correct Mean",
    "Incorrect Mean",
    "Invalid Mean",
]


def summary_row(summary: Dict[str, Any]) -> List[Any]:
    """
    Convert one grouped summary to a flat table row.
    """
    agg = summary["aggregated"]
    return [
        summary["relative_parent"],
        summary["base_name"],
        agg.get("num_runs"),
        agg.get("num_cases_mean"),
        agg.get("accuracy_mean"),
        agg.get("accuracy_std"),
        agg.get("precision_mean"),
        agg.get("precision_std"),
        agg.get("recall_mean"),
        agg.get("recall_std"),
        agg.get("f1_mean"),
        agg.get("f1_std"),
        agg.get("accuracy_left_right_mean"),
        agg.get("accuracy_left_right_std"),
        agg.get("f1_left_right_mean"),
        agg.get("f1_left_right_std"),
        agg.get("accuracy_above_below_mean"),
        agg.get("accuracy_above_below_std"),
        agg.get("f1_above_below_mean"),
        agg.get("f1_above_below_std"),
        agg.get("correct_mean"),
        agg.get("incorrect_mean"),
        agg.get("invalid_mean"),
    ]


def write_csv_summary(summaries: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Write the aggregated summary table as CSV.
    """
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_COLUMNS)
        for summary in summaries:
            writer.writerow(summary_row(summary))


def write_excel_summary(summaries: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Write the aggregated summary table as Excel workbook.
    """
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Results"
    sheet.append(SUMMARY_COLUMNS)

    for summary in summaries:
        sheet.append(summary_row(summary))

    workbook.save(output_path)


def write_json_summary(summaries: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Write all grouped metrics as JSON.
    """
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)


def write_invalid_cases(summaries: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Write all invalid or unparseable cases as JSON.
    """
    invalid_cases = []
    for summary in summaries:
        for case in summary.get("invalid_cases", []):
            invalid_cases.append({
                "relative_parent": summary["relative_parent"],
                "base_name": summary["base_name"],
                **case,
            })

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(invalid_cases, f, indent=2, ensure_ascii=False)


###############################################################################
# Command-line interface
###############################################################################


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate direct VLM baseline prediction files."
    )
    parser.add_argument(
        "--results_path",
        required=True,
        help="Directory containing prediction JSON or JSONL files.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for evaluation outputs. Defaults to <results_path>/evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point.
    """
    args = parse_args()

    results_path = Path(args.results_path).resolve()
    if not results_path.exists():
        raise FileNotFoundError(f"Results path does not exist: {results_path}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else results_path / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_files = collect_prediction_files(results_path, output_dir)
    print(f"Found {len(prediction_files)} prediction file(s) under: {results_path}")

    if not prediction_files:
        raise RuntimeError("No prediction JSON or JSONL files found.")

    grouped = group_prediction_files(prediction_files, results_path)
    summaries = []

    for (relative_parent, base_name), files in sorted(grouped.items()):
        print(f"Evaluating {relative_parent}/{base_name} with {len(files)} run file(s).")
        summaries.append(summarize_group(relative_parent, base_name, files))

    json_path = output_dir / "direct_vlm_evaluation_summary.json"
    csv_path = output_dir / "direct_vlm_evaluation_summary.csv"
    excel_path = output_dir / "direct_vlm_evaluation_summary.xlsx"
    invalid_cases_path = output_dir / "direct_vlm_invalid_cases.json"

    write_json_summary(summaries, json_path)
    write_csv_summary(summaries, csv_path)
    write_excel_summary(summaries, excel_path)
    write_invalid_cases(summaries, invalid_cases_path)

    print(f"JSON summary saved to: {json_path}")
    print(f"CSV summary saved to: {csv_path}")
    print(f"Excel summary saved to: {excel_path}")
    print(f"Invalid cases saved to: {invalid_cases_path}")


if __name__ == "__main__":
    main()
