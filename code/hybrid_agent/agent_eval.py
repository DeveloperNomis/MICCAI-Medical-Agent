import json
import os
import traceback
from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Any, Optional

from agent.controller import AgentController


# ============================================================
# CONFIG
# ============================================================

# Optional run identifier. On Slurm, this defaults to SLURM_JOB_ID.
JOB_ID = os.environ.get("JOB_ID", os.environ.get("SLURM_JOB_ID", "run"))

# Repository/project root. Can be overridden via environment variable.
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()

# Input files.
ORG_GT_PATH = Path(
    os.environ.get(
        "ORG_GT_PATH",
        PROJECT_ROOT / "data/relations/organs_gt.json",
    )
)

IMG_DIR = Path(
    os.environ.get(
        "IMG_DIR",
        PROJECT_ROOT / "data/images/test",
    )
)

# Output directory.
OUTPUT_DIR = Path(
    os.environ.get(
        "OUTPUT_DIR",
        PROJECT_ROOT / "results/inference/agent_eval",
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JSONL_OUTPUT = OUTPUT_DIR / f"agent_spatial_predictions_{JOB_ID}.jsonl"
JSON_OUTPUT = OUTPUT_DIR / f"agent_spatial_predictions_{JOB_ID}.json"
SUMMARY_OUTPUT = OUTPUT_DIR / f"agent_spatial_summary_{JOB_ID}.json"
ERROR_OUTPUT = OUTPUT_DIR / f"agent_spatial_errors_{JOB_ID}.json"


# ============================================================
# HELPERS
# ============================================================

def normalize_answer(value: Any) -> Optional[int]:
    """
    Normalize predictions or ground-truth answers to 0/1.
    Returns None if the value cannot be interpreted as a binary answer.
    """
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        if value in (0, 1):
            return value

    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes"}:
            return 1
        if v in {"0", "false", "no"}:
            return 0

    return None


def safe_asdict(obj: Any) -> Any:
    """
    Convert dataclasses recursively if possible.
    Fall back to string conversion if the object is not JSON-serializable.
    """
    if is_dataclass(obj):
        return asdict(obj)

    if isinstance(obj, dict):
        return {k: safe_asdict(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [safe_asdict(v) for v in obj]

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    return str(obj)


def extract_agent_prediction(agent_result: Any) -> Any:
    """
    Extract the final prediction from an AgentResult-like object.
    Adjust this function if the contract field is named differently.
    """
    if hasattr(agent_result, "result"):
        return agent_result.result

    if isinstance(agent_result, dict) and "result" in agent_result:
        return agent_result["result"]

    return None


def extract_query(agent_result: Any) -> Any:
    if hasattr(agent_result, "query"):
        return safe_asdict(agent_result.query)

    if isinstance(agent_result, dict) and "query" in agent_result:
        return safe_asdict(agent_result["query"])

    return None


def extract_audit_log(agent_result: Any) -> Any:
    if hasattr(agent_result, "audit_log"):
        return safe_asdict(agent_result.audit_log)

    if isinstance(agent_result, dict) and "audit_log" in agent_result:
        return safe_asdict(agent_result["audit_log"])

    return None


def extract_explanation(agent_result: Any) -> str:
    if hasattr(agent_result, "explanation"):
        return str(agent_result.explanation)

    if isinstance(agent_result, dict) and "explanation" in agent_result:
        return str(agent_result["explanation"])

    return ""


def load_test_entries(org_gt_path: Path, img_dir: Path):
    """
    Load organs_gt.json and keep only entries whose filename exists in the test image directory.

    Returns a flat list of QA cases:
    {
        filename,
        image_path,
        question,
        ground_truth,
        object1_name,
        object2_name,
        object1_gray,
        object2_gray
    }
    """
    with open(org_gt_path, "r", encoding="utf-8") as f:
        org_gt_data = json.load(f)

    test_names = {p.name for p in Path(img_dir).glob("*.png")}

    flat_cases = []

    for entry in org_gt_data:
        fname = Path(entry["filename"]).name

        if fname not in test_names:
            continue

        image_path = str(Path(img_dir) / fname)

        for qa in entry.get("question_answer", []):
            gt = normalize_answer(qa.get("answer"))

            if gt is None:
                continue

            flat_cases.append({
                "filename": fname,
                "image_path": image_path,
                "question": qa["question"],
                "ground_truth": gt,
                "object1_name": qa.get("object1_name"),
                "object2_name": qa.get("object2_name"),
                "object1_gray": qa.get("object1_gray"),
                "object2_gray": qa.get("object2_gray"),
                "object1_center_x": qa.get("object1_center_x"),
                "object1_center_y": qa.get("object1_center_y"),
                "object2_center_x": qa.get("object2_center_x"),
                "object2_center_y": qa.get("object2_center_y"),
            })

    return flat_cases


def write_jsonl_line(path: Path, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ============================================================
# MAIN
# ============================================================

def main():
    print("=== Loading test cases ===")
    cases = load_test_entries(ORG_GT_PATH, IMG_DIR)
    print(f"Loaded {len(cases)} QA cases from test images.")

    if not cases:
        print("No test cases found. Aborting.")
        return

    # Fresh run: remove old JSONL output if it exists.
    if JSONL_OUTPUT.exists():
        JSONL_OUTPUT.unlink()

    print("\n=== Initializing AgentController ===")
    print("The LLM is initialized once and reused for all cases.")
    controller = AgentController(use_llm=True, use_router=True)

    total = 0
    correct = 0
    invalid = 0

    tp = fp = tn = fn = 0
    strict_tp = strict_fp = strict_tn = strict_fn = 0

    all_results = []
    errors = []

    print("\n=== Running agent on test set ===")

    for idx, case in enumerate(cases, start=1):
        filename = case["filename"]
        image_path = case["image_path"]
        core_question = case["question"]
        ground_truth = case["ground_truth"]

        full_prompt = (
            "This is a 2D axial CT slice.\n\n"
            f"Question: {core_question}\n\n"
            "Answer the question with exactly one character:\n"
            "1 if the statement is true.\n"
            "0 if the statement is false.\n\n"
            "Do not output any explanation."
        )

        print(f"[{idx}/{len(cases)}] {filename}")

        result_entry = {
            "filename": filename,
            "image_path": image_path,
            "question": core_question,
            "raw_prompt": full_prompt,
            "ground_truth": ground_truth,
            "object1_name": case.get("object1_name"),
            "object2_name": case.get("object2_name"),
            "object1_gray": case.get("object1_gray"),
            "object2_gray": case.get("object2_gray"),
            "status": "ok",
        }

        try:
            agent_result = controller.run(
                question=full_prompt,
                image=image_path,
            )

            raw_prediction = extract_agent_prediction(agent_result)
            normalized_prediction = normalize_answer(raw_prediction)

            explanation = extract_explanation(agent_result)
            query = extract_query(agent_result)
            audit_log = extract_audit_log(agent_result)

            result_entry.update({
                "agent_result_raw": safe_asdict(agent_result),
                "prediction_raw": raw_prediction,
                "prediction": normalized_prediction,
                "query": query,
                "audit_log": audit_log,
                "explanation": explanation,
            })

            if normalized_prediction is None:
                result_entry["status"] = "invalid_prediction"
                result_entry["correct"] = False
                invalid += 1

                if ground_truth == 1:
                    strict_fn += 1
                else:
                    strict_fp += 1

            else:
                is_correct = normalized_prediction == ground_truth
                result_entry["correct"] = is_correct

                if is_correct:
                    correct += 1

                if normalized_prediction == 1 and ground_truth == 1:
                    tp += 1
                    strict_tp += 1
                elif normalized_prediction == 1 and ground_truth == 0:
                    fp += 1
                    strict_fp += 1
                elif normalized_prediction == 0 and ground_truth == 0:
                    tn += 1
                    strict_tn += 1
                elif normalized_prediction == 0 and ground_truth == 1:
                    fn += 1
                    strict_fn += 1

        except Exception as e:
            result_entry.update({
                "status": "exception",
                "correct": False,
                "prediction_raw": None,
                "prediction": None,
                "query": None,
                "audit_log": None,
                "explanation": "",
                "error": str(e),
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "traceback": traceback.format_exc(),
            })

            invalid += 1

            if ground_truth == 1:
                strict_fn += 1
            else:
                strict_fp += 1

        total += 1

        # Append the current result immediately to make long runs recoverable.
        write_jsonl_line(JSONL_OUTPUT, result_entry)
        all_results.append(result_entry)

        if result_entry["status"] != "ok":
            errors.append(result_entry)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    valid_predictions = total - invalid
    accuracy_strict = correct / total if total else 0.0
    accuracy_valid_only = correct / valid_predictions if valid_predictions else 0.0

    precision_valid = tp / (tp + fp) if (tp + fp) else 0.0
    recall_valid = tp / (tp + fn) if (tp + fn) else 0.0
    f1_valid = (
        2 * precision_valid * recall_valid / (precision_valid + recall_valid)
        if (precision_valid + recall_valid)
        else 0.0
    )

    precision_strict = strict_tp / (strict_tp + strict_fp) if (strict_tp + strict_fp) else 0.0
    recall_strict = strict_tp / (strict_tp + strict_fn) if (strict_tp + strict_fn) else 0.0
    f1_strict = (
        2 * precision_strict * recall_strict / (precision_strict + recall_strict)
        if (precision_strict + recall_strict)
        else 0.0
    )

    summary = {
        "num_cases": total,
        "correct": correct,
        "incorrect_or_invalid": total - correct,
        "invalid_predictions": invalid,
        "valid_predictions": valid_predictions,
        "accuracy_strict": accuracy_strict,
        "accuracy_valid_only": accuracy_valid_only,

        "tp_valid": tp,
        "fp_valid": fp,
        "tn_valid": tn,
        "fn_valid": fn,
        "precision_valid": precision_valid,
        "recall_valid": recall_valid,
        "f1_valid": f1_valid,

        "tp_strict": strict_tp,
        "fp_strict": strict_fp,
        "tn_strict": strict_tn,
        "fn_strict": strict_fn,
        "precision_strict": precision_strict,
        "recall_strict": recall_strict,
        "f1_strict": f1_strict,

        "jsonl_output": str(JSONL_OUTPUT),
        "json_output": str(JSON_OUTPUT),
        "errors_output": str(ERROR_OUTPUT),
        "summary_output": str(SUMMARY_OUTPUT),
    }

    # Save the full JSON list.
    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Save the summary.
    with open(SUMMARY_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save problematic cases.
    with open(ERROR_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(
            [r for r in all_results if r.get("status") != "ok" or not r.get("correct", False)],
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n===== Agent Spatial Evaluation Summary =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nSaved files:")
    print(" -", JSONL_OUTPUT)
    print(" -", JSON_OUTPUT)
    print(" -", SUMMARY_OUTPUT)
    print(" -", ERROR_OUTPUT)


if __name__ == "__main__":
    main()