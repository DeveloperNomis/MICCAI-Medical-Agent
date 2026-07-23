# tasks/presence/presence_task.py

from tasks.base_task import BaseTask
from language_parsing.language_parser import extract_presence_entity


class PresenceTask(BaseTask):
    """
    Presence verification task.

    This task checks whether a queried anatomical structure is detected
    in the input image. It is implemented as a secondary task pathway but
    is not the focus of the spatial QA experiments reported in the paper.
    """

    def __init__(self, yolo_model, llm_advisor=None, class_names=None):
        self.yolo = yolo_model
        self.llm_advisor = llm_advisor
        self.class_names = class_names or []

    def execute(self, question, image):
        parse_source = "deterministic"

        # Extract the queried anatomical structure.
        entity = extract_presence_entity(question, self.class_names)

        # Use the LLM fallback only if deterministic extraction fails.
        if not entity and self.llm_advisor:
            entity = self.llm_advisor.suggest_presence(question)
            parse_source = "llm_fallback"

        if not entity:
            return {
                "query": None,
                "detections": [],
                "result": False,
                "detection_feedback": None,
                "parse_source": "failed",
                "error": "language_parsing_failed",
            }

        if self.yolo is None:
            return {
                "query": entity,
                "detections": [],
                "result": False,
                "detection_feedback": None,
                "parse_source": parse_source,
                "error": "yolo_not_loaded",
            }

        # Run object detection.
        detections = self.yolo.detect(image)

        detection_feedback = None

        # Optional detection review is not required for the reported benchmark.
        # if self.llm_advisor:
        #     detection_feedback = self.llm_advisor.review_detections(detections)

        # Check whether the queried structure was detected.
        result = any(
            det.get("class_name") == entity
            for det in detections
            if isinstance(det, dict)
        )

        return {
            "query": entity,
            "detections": detections,
            "result": bool(result),
            "detection_feedback": detection_feedback,
            "parse_source": parse_source,
            "error": None,
        }