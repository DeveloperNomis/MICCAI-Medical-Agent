from tasks.base_task import BaseTask
from tasks.spatial.deterministic_spatial import run_detection
from tasks.spatial.guards_spatial import validate_query
from language_parsing.language_parser import extract_triplet
from perception.deterministic_relations import check_relation


def query_to_dict(query):
    """
    Convert a structured relation query to a serializable dictionary.
    """
    if query is None:
        return None

    return {
        "entity_a": query.entity_a,
        "relation": query.relation,
        "entity_b": query.entity_b,
    }


class SpatialTask(BaseTask):
    """
    Spatial verification task.

    This task parses a natural-language spatial question, runs anatomical
    object detection, and evaluates the requested relation using deterministic
    geometry.
    """

    def __init__(self, yolo_model, llm_advisor=None, class_names=None):
        self.yolo = yolo_model
        self.llm_advisor = llm_advisor
        self.class_names = class_names or []

    def execute(self, question, image):
        parse_source = "deterministic"

        # 1. Deterministic parsing.
        raw_query = extract_triplet(question, self.class_names)
        validated_query = validate_query(raw_query)

        # 2. LLM fallback only if deterministic parsing fails.
        if not validated_query and self.llm_advisor:
            suggestion = self.llm_advisor.suggest_spatial(question)
            validated_query = validate_query(suggestion)

            if validated_query:
                parse_source = "llm_fallback"
            else:
                parse_source = "failed"

        print("Validated query:", validated_query)

        if not validated_query:
            return {
                "query": None,
                "detections": [],
                "result": False,
                "detection_feedback": None,
                "parse_source": parse_source,
                "error": "language_parsing_failed",
                "audit": {
                    "question": question,
                    "raw_query": query_to_dict(raw_query),
                    "validated_query": None,
                    "geometry_debug": None,
                },
            }

        # Language-only mode.
        if self.yolo is None:
            print("YOLO model is not loaded.")
            return {
                "query": validated_query,
                "detections": [],
                "result": False,
                "detection_feedback": None,
                "parse_source": parse_source,
                "error": "yolo_not_loaded",
                "audit": {
                    "question": question,
                    "raw_query": query_to_dict(raw_query),
                    "validated_query": query_to_dict(validated_query),
                    "geometry_debug": None,
                },
            }

        # 3. Object detection.
        detections = run_detection(self.yolo, image)

        detection_feedback = None

        # Detection review is intentionally disabled for the reported benchmark.
        # If enabled, this would ask the LLM advisor to inspect detections, but
        # the final benchmark keeps perception and deterministic geometry fixed.
        # if self.llm_advisor:
        #     detection_feedback = self.llm_advisor.review_detections(detections)

        # 4. Deterministic geometric verification.
        result, geometry_debug = check_relation(
            validated_query,
            detections,
            return_debug=True,
        )

        return {
            "query": validated_query,
            "detections": detections,
            "result": bool(result),
            "detection_feedback": detection_feedback,
            "parse_source": parse_source,
            "error": None,
            "audit": {
                "question": question,
                "raw_query": query_to_dict(raw_query),
                "validated_query": query_to_dict(validated_query),
                "geometry_debug": geometry_debug,
            },
        }