import json
import math
import re
import os
from pathlib import Path
from typing import Any, Optional
from collections import Counter


# ============================================================
# CONFIG
# ============================================================

JOB_ID = os.environ.get("JOB_ID", "run")

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

PREDICTIONS_JSONL = Path(
    os.environ.get(
        "PREDICTIONS_JSONL",
        PROJECT_ROOT / f"results/inference/agent_eval/agent_spatial_predictions_{JOB_ID}.jsonl",
    )
)

PREDICTIONS_JSON = Path(
    os.environ.get(
        "PREDICTIONS_JSON",
        PROJECT_ROOT / f"results/inference/agent_eval/agent_spatial_predictions_{JOB_ID}.json",
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

CHECK_SUMMARY_OUT = OUTPUT_DIR / f"agent_eval_check_summary_{JOB_ID}.json"
MISSING_CASES_OUT = OUTPUT_DIR / f"agent_eval_missing_cases_{JOB_ID}.json"
DUPLICATE_CASES_OUT = OUTPUT_DIR / f"agent_eval_duplicate_cases_{JOB_ID}.json"
EVAL_SUMMARY_OUT = OUTPUT_DIR / f"agent_eval_metrics_{JOB_ID}.json"
WRONG_CASES_OUT = OUTPUT_DIR / f"agent_eval_wrong_cases_{JOB_ID}.json"

FAILURE_ATTRIBUTION_OUT = OUTPUT_DIR / f"agent_failure_attribution_{JOB_ID}.json"
WRONG_CASES_ENRICHED_OUT = OUTPUT_DIR / f"agent_eval_wrong_cases_enriched_{JOB_ID}.json"
LATEX_FAILURE_TABLE_OUT = OUTPUT_DIR / f"agent_failure_attribution_latex_{JOB_ID}.txt"

LOCALIZATION_THRESHOLD_PX = 30.0


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_answer(value: Any) -> Optional[int]:
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


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def normalize_entity(text: Any) -> str:
    text = str(text or "").replace("_", " ")
    return normalize_space(text)


def normalize_relation(text: Any) -> str:
    r = normalize_space(text)

    if r in {"left", "left of", "to the left of"}:
        return "left of"
    if r in {"right", "right of", "to the right of"}:
        return "right of"
    if r in {"above", "over"}:
        return "above"
    if r in {"below", "under"}:
        return "below"

    return r


def extract_question_from_prompt(text: str) -> str:
    """
    Makes matching robust if a prediction accidentally stores the full prompt
    instead of only the core question.
    """
    if text is None:
        return ""

    match = re.search(
        r"Question:\s*(.*?)(?:\n\s*\n|$)",
        str(text),
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    return str(text).strip()


def infer_relation_from_question(question: str) -> str:
    q = normalize_space(question)

    if "left of" in q or "to the left of" in q:
        return "left of"
    if "right of" in q or "to the right of" in q:
        return "right of"
    if "above" in q:
        return "above"
    if "below" in q:
        return "below"

    return "unknown"


def make_key(entry: dict):
    fname = Path(entry["filename"]).name
    question = extract_question_from_prompt(entry["question"])
    return (fname, question)


def point_from_xy(x, y):
    if x is None or y is None:
        return None

    try:
        return [float(x), float(y)]
    except Exception:
        return None


def point_from_detection(det: Optional[dict]):
    if not det:
        return None

    center = det.get("center")

    if isinstance(center, (list, tuple)) and len(center) >= 2:
        try:
            return [float(center[0]), float(center[1])]
        except Exception:
            return None

    if "center_x" in det and "center_y" in det:
        return point_from_xy(det.get("center_x"), det.get("center_y"))

    return None


def euclidean_distance(p1, p2):
    if p1 is None or p2 is None:
        return None

    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def relation_answer_from_centers(center_a, center_b, relation: str) -> Optional[int]:
    if center_a is None or center_b is None:
        return None

    ax, ay = center_a
    bx, by = center_b

    relation = normalize_relation(relation)

    if relation == "left of":
        return int(ax < bx)
    if relation == "right of":
        return int(ax > bx)
    if relation == "above":
        return int(ay < by)
    if relation == "below":
        return int(ay > by)

    return None


# ============================================================
# LOADING
# ============================================================

def load_expected_cases(org_gt_path: Path, img_dir: Path):
    """
    Builds the expected spatial QA cases from organs_gt.json,
    filtered to images present in IMG_DIR.
    Uses (filename, question) as unique case key.
    """
    with open(org_gt_path, "r", encoding="utf-8") as f:
        org_gt_data = json.load(f)

    test_names = {p.name for p in Path(img_dir).glob("*.png")}

    expected_cases = []

    for entry in org_gt_data:
        fname = Path(entry["filename"]).name

        if fname not in test_names:
            continue

        for qa in entry.get("question_answer", []):
            gt = normalize_answer(qa.get("answer"))

            if gt is None:
                continue

            expected_cases.append({
                "filename": fname,
                "question": qa["question"],
                "ground_truth": gt,

                "object1_name": qa.get("object1_name"),
                "object2_name": qa.get("object2_name"),
                "object1_gray": qa.get("object1_gray"),
                "object2_gray": qa.get("object2_gray"),

                # Needed for localization failure attribution
                "object1_center_x": qa.get("object1_center_x"),
                "object1_center_y": qa.get("object1_center_y"),
                "object2_center_x": qa.get("object2_center_x"),
                "object2_center_y": qa.get("object2_center_y"),
            })

    return expected_cases


def load_predictions(predictions_jsonl: Path = None, predictions_json: Path = None):
    """
    Loads prediction entries from JSONL if available, otherwise JSON.
    """
    if predictions_jsonl and Path(predictions_jsonl).exists():
        rows = []

        with open(predictions_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        return rows

    if predictions_json and Path(predictions_json).exists():
        with open(predictions_json, "r", encoding="utf-8") as f:
            return json.load(f)

    raise FileNotFoundError("Neither predictions_jsonl nor predictions_json could be loaded.")


# ============================================================
# EXTRACTION FROM AGENT OUTPUT
# ============================================================

def get_agent_raw(row: dict) -> dict:
    raw = row.get("agent_result_raw")
    return raw if isinstance(raw, dict) else {}


def get_audit_log(row: dict) -> dict:
    audit = row.get("audit_log")

    if isinstance(audit, dict):
        return audit

    raw = get_agent_raw(row)
    audit = raw.get("audit_log")

    return audit if isinstance(audit, dict) else {}


def get_task_audit(row: dict) -> dict:
    audit = get_audit_log(row)

    task_audit = audit.get("task_audit")
    if isinstance(task_audit, dict):
        return task_audit

    return {}


def get_query_from_row(row: dict) -> dict:
    """
    Main query stored by agent_eval.py.
    """
    query = row.get("query")

    if isinstance(query, dict):
        return query

    raw = get_agent_raw(row)
    query = raw.get("query")

    return query if isinstance(query, dict) else {}


def get_raw_query(row: dict) -> dict:
    """
    Raw query before ontology matching, if available.
    Falls back to final query.
    """
    task_audit = get_task_audit(row)

    for key in ["raw_query", "parsed_query", "parser_output"]:
        value = task_audit.get(key)
        if isinstance(value, dict):
            return value

    return get_query_from_row(row)


def get_validated_query(row: dict) -> dict:
    """
    Query after ontology matching / validation, if available.
    Falls back to final query.
    """
    task_audit = get_task_audit(row)

    for key in ["validated_query", "normalized_query", "query"]:
        value = task_audit.get(key)
        if isinstance(value, dict):
            return value

    return get_query_from_row(row)


def get_geometry_debug(row: dict) -> dict:
    task_audit = get_task_audit(row)

    for key in ["geometry_debug", "geometry", "relation_debug"]:
        value = task_audit.get(key)
        if isinstance(value, dict):
            return value

    return {}


def get_all_detections(row: dict):
    detections = []

    raw = get_agent_raw(row)

    if isinstance(raw.get("detections"), list):
        detections.extend(raw["detections"])

    if isinstance(row.get("detections"), list):
        detections.extend(row["detections"])

    return detections


def select_detection_for_class(row: dict, class_name: str, geometry_slot: str = None):
    """
    Prefer the object used by the geometry module if available.
    Otherwise, select highest-confidence detection for the requested class.
    """
    class_name_n = normalize_entity(class_name)

    geometry_debug = get_geometry_debug(row)

    if geometry_slot:
        obj = geometry_debug.get(geometry_slot)
        if isinstance(obj, dict) and normalize_entity(obj.get("class_name")) == class_name_n:
            return obj

    for key in ["obj_a", "object_a", "entity_a_detection"]:
        obj = geometry_debug.get(key)
        if isinstance(obj, dict) and normalize_entity(obj.get("class_name")) == class_name_n:
            return obj

    for key in ["obj_b", "object_b", "entity_b_detection"]:
        obj = geometry_debug.get(key)
        if isinstance(obj, dict) and normalize_entity(obj.get("class_name")) == class_name_n:
            return obj

    candidates = []

    for det in get_all_detections(row):
        if not isinstance(det, dict):
            continue

        if normalize_entity(det.get("class_name")) == class_name_n:
            candidates.append(det)

    if not candidates:
        return None

    return max(candidates, key=lambda d: float(d.get("confidence", 0.0)))


# ============================================================
# FAILURE ATTRIBUTION
# ============================================================

def assign_failure_stage(sample: dict):
    """
    Assigns each failed sample to the earliest failing stage.
    The order matches the paper table.
    """
    if sample.get("correct") is True:
        return "correct", "correct_prediction"

    if not sample.get("valid_output", False):
        return "Formatting/runtime", "invalid_or_missing_prediction"

    if not sample.get("question_extraction_success", True):
        return "Question extraction", "failed_to_extract_core_question"

    if not sample.get("routing_correct", True):
        return "Routing", "wrong_task_route"

    if not sample.get("parser_correct", False):
        return "Parsing", "wrong_entity_or_relation"

    if not sample.get("ontology_match_success", False):
        return "Ontology matching", "entity_not_mapped_to_detector_class"

    if not sample.get("entity_a_found", False) and not sample.get("entity_b_found", False):
        return "Missing detection", "both_entities_missing"

    if not sample.get("entity_a_found", False):
        return "Missing detection", "entity_a_missing"

    if not sample.get("entity_b_found", False):
        return "Missing detection", "entity_b_missing"

    center_error_a = sample.get("center_error_a")
    center_error_b = sample.get("center_error_b")

    if center_error_a is not None and center_error_a > LOCALIZATION_THRESHOLD_PX:
        return "Imprecise localization", "entity_a_center_error_above_threshold"

    if center_error_b is not None and center_error_b > LOCALIZATION_THRESHOLD_PX:
        return "Imprecise localization", "entity_b_center_error_above_threshold"

    if sample.get("geometry_correct") is True and sample.get("correct") is False:
        return "Formatting/runtime", "final_answer_inconsistent_with_geometry"

    if not sample.get("geometry_correct", False):
        return "Geometry ambiguity", "wrong_relation_despite_available_detections"

    return "Geometry ambiguity", "unclassified_after_geometry"


def build_stage_audit(pred_row: dict, gt_row: dict, pred: Optional[int], gt: int, is_correct: bool):
    """
    Creates the per-sample audit object used for failure attribution.
    """

    # -------------------------
    # Gold information
    # -------------------------
    gold_entity_a = gt_row.get("object1_name")
    gold_entity_b = gt_row.get("object2_name")
    gold_relation = infer_relation_from_question(gt_row.get("question", ""))

    gold_center_a = point_from_xy(
        gt_row.get("object1_center_x"),
        gt_row.get("object1_center_y"),
    )
    gold_center_b = point_from_xy(
        gt_row.get("object2_center_x"),
        gt_row.get("object2_center_y"),
    )

    # -------------------------
    # Prompt / routing
    # -------------------------
    audit_log = get_audit_log(pred_row)

    extracted_question = audit_log.get("extracted_question")
    expected_question = gt_row.get("question")

    if extracted_question is None:
        question_extraction_success = True
    else:
        question_extraction_success = (
            normalize_space(extracted_question) == normalize_space(expected_question)
        )

    task = audit_log.get("task")

    if task is None:
        routing_correct = True
    else:
        routing_correct = normalize_space(task) == "spatial"

    # -------------------------
    # Query / parsing / ontology
    # -------------------------
    raw_query = get_raw_query(pred_row)
    validated_query = get_validated_query(pred_row)
    final_query = get_query_from_row(pred_row)

    raw_entity_a = raw_query.get("entity_a")
    raw_relation = raw_query.get("relation")
    raw_entity_b = raw_query.get("entity_b")

    val_entity_a = validated_query.get("entity_a")
    val_relation = validated_query.get("relation")
    val_entity_b = validated_query.get("entity_b")

    parser_correct = (
        normalize_entity(raw_entity_a) == normalize_entity(gold_entity_a)
        and normalize_relation(raw_relation) == normalize_relation(gold_relation)
        and normalize_entity(raw_entity_b) == normalize_entity(gold_entity_b)
    )

    ontology_match_success = (
        normalize_entity(val_entity_a) == normalize_entity(gold_entity_a)
        and normalize_relation(val_relation) == normalize_relation(gold_relation)
        and normalize_entity(val_entity_b) == normalize_entity(gold_entity_b)
    )

    # -------------------------
    # Detection / localization
    # -------------------------
    det_a = select_detection_for_class(pred_row, val_entity_a, geometry_slot="obj_a")
    det_b = select_detection_for_class(pred_row, val_entity_b, geometry_slot="obj_b")

    entity_a_found = det_a is not None
    entity_b_found = det_b is not None

    pred_center_a = point_from_detection(det_a)
    pred_center_b = point_from_detection(det_b)

    center_error_a = euclidean_distance(pred_center_a, gold_center_a)
    center_error_b = euclidean_distance(pred_center_b, gold_center_b)

    # -------------------------
    # Geometry
    # -------------------------
    geometry_debug = get_geometry_debug(pred_row)

    pred_geometry_answer = relation_answer_from_centers(
        pred_center_a,
        pred_center_b,
        gold_relation,
    )

    geometry_correct = (
        pred_geometry_answer is not None
        and pred_geometry_answer == gt
    )

    sample = {
        "filename": pred_row.get("filename"),
        "question": pred_row.get("question"),
        "raw_prompt": pred_row.get("raw_prompt"),

        "ground_truth": gt,
        "prediction": pred,
        "prediction_raw": pred_row.get("prediction_raw"),
        "status": pred_row.get("status"),

        "correct": is_correct,
        "valid_output": pred is not None,

        "question_extraction_success": question_extraction_success,
        "routing_correct": routing_correct,

        "parser_correct": parser_correct,
        "ontology_match_success": ontology_match_success,

        "entity_a_found": entity_a_found,
        "entity_b_found": entity_b_found,
        "center_error_a": center_error_a,
        "center_error_b": center_error_b,
        "geometry_correct": geometry_correct,

        "gold": {
            "entity_a": gold_entity_a,
            "relation": gold_relation,
            "entity_b": gold_entity_b,
            "answer": gt,
            "center_a": gold_center_a,
            "center_b": gold_center_b,
        },

        "parsed_query": final_query,
        "raw_query": raw_query,
        "validated_query": validated_query,

        "ontology_matching": {
            "entity_a_raw": raw_entity_a,
            "entity_a_matched_class": val_entity_a,
            "entity_a_gold_class": gold_entity_a,
            "entity_a_correct": normalize_entity(val_entity_a) == normalize_entity(gold_entity_a),

            "entity_b_raw": raw_entity_b,
            "entity_b_matched_class": val_entity_b,
            "entity_b_gold_class": gold_entity_b,
            "entity_b_correct": normalize_entity(val_entity_b) == normalize_entity(gold_entity_b),

            "success": ontology_match_success,
            "correct": ontology_match_success,
        },

        "detections": {
            "entity_a_found": entity_a_found,
            "entity_b_found": entity_b_found,
            "entity_a_detection": det_a,
            "entity_b_detection": det_b,
            "entity_a_center": pred_center_a,
            "entity_b_center": pred_center_b,
            "entity_a_center_error": center_error_a,
            "entity_b_center_error": center_error_b,
        },

        "geometry": {
            "predicted_relation_answer": pred_geometry_answer,
            "gold_geometry_answer": gt,
            "correct": geometry_correct,
            "dx": geometry_debug.get("dx"),
            "dy": geometry_debug.get("dy"),
        },

        "audit_log": audit_log,
    }

    failure_stage, failure_reason = assign_failure_stage(sample)

    sample["failure_stage"] = failure_stage
    sample["failure_reason"] = failure_reason

    return sample


def build_failure_summary(wrong_cases_enriched, total_cases: int):
    stages = [
        "Question extraction",
        "Routing",
        "Parsing",
        "Ontology matching",
        "Missing detection",
        "Imprecise localization",
        "Geometry ambiguity",
        "Formatting/runtime",
    ]

    counter = Counter(case["failure_stage"] for case in wrong_cases_enriched)
    num_failures = len(wrong_cases_enriched)

    summary = {}

    for stage in stages:
        count = counter.get(stage, 0)

        summary[stage] = {
            "count": count,
            "percent_of_failures": (count / num_failures * 100.0) if num_failures else 0.0,
            "percent_of_all_cases": (count / total_cases * 100.0) if total_cases else 0.0,
        }

    unknown_count = counter.get("unknown", 0)
    if unknown_count:
        summary["unknown"] = {
            "count": unknown_count,
            "percent_of_failures": (unknown_count / num_failures * 100.0) if num_failures else 0.0,
            "percent_of_all_cases": (unknown_count / total_cases * 100.0) if total_cases else 0.0,
        }

    return summary


def make_latex_failure_table_rows(failure_summary: dict):
    rows = [
        ("Question extraction", "Question extraction", "malformed prompt"),
        ("Routing", "Routing", "wrong pathway"),
        ("Parsing", "Parsing / query extraction", "wrong entity or relation"),
        ("Ontology matching", "Ontology matching", "detector-class mapping failure"),
        ("Missing detection", "Missing detection", "organ not detected"),
        ("Imprecise localization", "Imprecise localization", "center error changes relation"),
        ("Geometry ambiguity", "Geometry ambiguity", "borderline relation"),
        ("Formatting/runtime", "Formatting/runtime", "invalid output or exception"),
    ]

    lines = []

    for key, display_name, reason in rows:
        item = failure_summary.get(key, {})
        count = item.get("count", 0)
        pct = item.get("percent_of_failures", 0.0)

        lines.append(
            f"{display_name} & {count} & {pct:.1f} & {reason} \\\\"
        )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=== Loading expected cases ===")
    expected_cases = load_expected_cases(ORG_GT_PATH, IMG_DIR)
    expected_keys = [make_key(x) for x in expected_cases]
    expected_key_set = set(expected_keys)

    print(f"Expected total cases: {len(expected_cases)}")
    print(f"Expected unique keys: {len(expected_key_set)}")

    print("\n=== Loading prediction results ===")
    predictions = load_predictions(PREDICTIONS_JSONL, PREDICTIONS_JSON)
    pred_keys = [make_key(x) for x in predictions if "filename" in x and "question" in x]

    print(f"Prediction rows loaded: {len(predictions)}")
    print(f"Prediction keys found: {len(pred_keys)}")

    # --------------------------------------------------------
    # 1) Check completeness
    # --------------------------------------------------------
    seen = {}
    duplicates = []

    for row in predictions:
        if "filename" not in row or "question" not in row:
            continue

        key = make_key(row)

        if key in seen:
            duplicates.append(row)
        else:
            seen[key] = row

    pred_key_set = set(seen.keys())

    missing_keys = expected_key_set - pred_key_set
    extra_keys = pred_key_set - expected_key_set

    missing_cases = [x for x in expected_cases if make_key(x) in missing_keys]

    check_summary = {
        "expected_total_cases": len(expected_cases),
        "expected_unique_cases": len(expected_key_set),
        "prediction_rows_loaded": len(predictions),
        "unique_prediction_cases": len(pred_key_set),
        "missing_cases": len(missing_keys),
        "duplicate_cases": len(duplicates),
        "unexpected_extra_cases": len(extra_keys),
        "all_cases_processed": len(missing_keys) == 0,
    }

    with open(CHECK_SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(check_summary, f, indent=2, ensure_ascii=False)

    with open(MISSING_CASES_OUT, "w", encoding="utf-8") as f:
        json.dump(missing_cases, f, indent=2, ensure_ascii=False)

    with open(DUPLICATE_CASES_OUT, "w", encoding="utf-8") as f:
        json.dump(duplicates, f, indent=2, ensure_ascii=False)

    print("\n=== Completeness Check ===")
    for k, v in check_summary.items():
        print(f"{k}: {v}")

    if extra_keys:
        print("\nWarning: found prediction cases not present in expected set.")
        print("Example extra keys:", list(extra_keys)[:5])

    # --------------------------------------------------------
    # 2) Evaluation + failure attribution
    # --------------------------------------------------------
    print("\n=== Evaluating predictions ===")

    total_expected = len(expected_cases)

    correct = 0
    invalid = 0
    processed = 0

    tp = fp = tn = fn = 0
    strict_tp = strict_fp = strict_tn = strict_fn = 0

    wrong_cases = []
    wrong_cases_enriched = []

    expected_lookup = {make_key(x): x for x in expected_cases}

    for key, pred_row in seen.items():
        gt_row = expected_lookup.get(key)

        if gt_row is None:
            continue

        gt = normalize_answer(gt_row["ground_truth"])
        pred = normalize_answer(pred_row.get("prediction"))

        if gt is None:
            continue

        processed += 1

        if pred is None:
            invalid += 1
            is_correct = False

            if gt == 1:
                strict_fn += 1
            else:
                strict_fp += 1

        else:
            is_correct = (pred == gt)

            if is_correct:
                correct += 1

            if pred == 1 and gt == 1:
                tp += 1
                strict_tp += 1
            elif pred == 1 and gt == 0:
                fp += 1
                strict_fp += 1
            elif pred == 0 and gt == 0:
                tn += 1
                strict_tn += 1
            elif pred == 0 and gt == 1:
                fn += 1
                strict_fn += 1

        if not is_correct:
            enriched = build_stage_audit(
                pred_row=pred_row,
                gt_row=gt_row,
                pred=pred,
                gt=gt,
                is_correct=is_correct,
            )

            wrong_cases_enriched.append(enriched)

            wrong_cases.append({
                "filename": pred_row.get("filename"),
                "question": pred_row.get("question"),
                "ground_truth": gt,
                "prediction": pred,
                "prediction_raw": pred_row.get("prediction_raw"),
                "status": pred_row.get("status"),
                "reason": "invalid_prediction" if pred is None else "wrong_prediction",
                "failure_stage": enriched["failure_stage"],
                "failure_reason": enriched["failure_reason"],
                "query": pred_row.get("query"),
                "audit_log": pred_row.get("audit_log"),
                "explanation": pred_row.get("explanation"),
            })

    # --------------------------------------------------------
    # Missing cases count as strict failures
    # --------------------------------------------------------
    for missing in missing_cases:
        gt = normalize_answer(missing.get("ground_truth"))

        if gt == 1:
            strict_fn += 1
        elif gt == 0:
            strict_fp += 1

        wrong_cases_enriched.append({
            "filename": missing.get("filename"),
            "question": missing.get("question"),
            "ground_truth": gt,
            "prediction": None,
            "correct": False,
            "valid_output": False,
            "failure_stage": "Formatting/runtime",
            "failure_reason": "missing_prediction_case",
            "gold": {
                "entity_a": missing.get("object1_name"),
                "relation": infer_relation_from_question(missing.get("question", "")),
                "entity_b": missing.get("object2_name"),
                "answer": gt,
                "center_a": point_from_xy(missing.get("object1_center_x"), missing.get("object1_center_y")),
                "center_b": point_from_xy(missing.get("object2_center_x"), missing.get("object2_center_y")),
            },
        })

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    valid_predictions = processed - invalid

    accuracy_strict = correct / total_expected if total_expected else 0.0
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

    failure_attribution = build_failure_summary(
        wrong_cases_enriched=wrong_cases_enriched,
        total_cases=total_expected,
    )

    latex_rows = make_latex_failure_table_rows(failure_attribution)

    eval_summary = {
        "expected_total_cases": total_expected,
        "processed_unique_cases": len(pred_key_set),
        "processed_matching_cases": processed,
        "correct": correct,
        "incorrect_or_invalid": total_expected - correct,
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

        "localization_threshold_px": LOCALIZATION_THRESHOLD_PX,
        "failure_attribution": failure_attribution,

        "check_summary_output": str(CHECK_SUMMARY_OUT),
        "wrong_cases_output": str(WRONG_CASES_OUT),
        "wrong_cases_enriched_output": str(WRONG_CASES_ENRICHED_OUT),
        "failure_attribution_output": str(FAILURE_ATTRIBUTION_OUT),
        "latex_failure_table_output": str(LATEX_FAILURE_TABLE_OUT),
    }

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    with open(EVAL_SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)

    with open(WRONG_CASES_OUT, "w", encoding="utf-8") as f:
        json.dump(wrong_cases, f, indent=2, ensure_ascii=False)

    with open(WRONG_CASES_ENRICHED_OUT, "w", encoding="utf-8") as f:
        json.dump(wrong_cases_enriched, f, indent=2, ensure_ascii=False)

    with open(FAILURE_ATTRIBUTION_OUT, "w", encoding="utf-8") as f:
        json.dump(failure_attribution, f, indent=2, ensure_ascii=False)

    with open(LATEX_FAILURE_TABLE_OUT, "w", encoding="utf-8") as f:
        f.write(latex_rows)

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------
    print("\n=== Evaluation Summary ===")
    for k, v in eval_summary.items():
        if k != "failure_attribution":
            print(f"{k}: {v}")

    print("\n=== Failure Attribution ===")
    for stage, item in failure_attribution.items():
        print(
            f"{stage}: "
            f"{item['count']} "
            f"({item['percent_of_failures']:.1f}% of failures, "
            f"{item['percent_of_all_cases']:.1f}% of all cases)"
        )

    print("\n=== LaTeX table rows ===")
    print(latex_rows)

    print("\nSaved files:")
    print(" -", CHECK_SUMMARY_OUT)
    print(" -", MISSING_CASES_OUT)
    print(" -", DUPLICATE_CASES_OUT)
    print(" -", EVAL_SUMMARY_OUT)
    print(" -", WRONG_CASES_OUT)
    print(" -", WRONG_CASES_ENRICHED_OUT)
    print(" -", FAILURE_ATTRIBUTION_OUT)
    print(" -", LATEX_FAILURE_TABLE_OUT)


if __name__ == "__main__":
    main()