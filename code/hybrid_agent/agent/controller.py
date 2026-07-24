import os
import re
from pathlib import Path

from tasks.spatial.spatial_task import SpatialTask
from tasks.measurement.measurement_task import MeasurementTask
from tasks.presence.presence_task import PresenceTask

from language_parsing.llm_advisor import LLMAdvisor
from agent.contracts import AgentResult
from agent.router import TaskRouter

from perception.yolo_model import YoloModel
from utils.class_loader import load_class_names_from_yaml
from utils.timer import Timer


class AgentController:
    """
    Main controller for the hybrid medical imaging agent.

    The controller handles:
    - class-name loading,
    - YOLO model initialization,
    - optional LLM advisor setup,
    - task routing,
    - task execution,
    - standardized AgentResult creation.
    """

    def __init__(
        self,
        yolo_model=None,
        use_llm=True,
        use_router=True,
        data_yaml_path=None,
        yolo_weights_path=None,
    ):
        project_root = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()

        data_yaml_path = Path(
            data_yaml_path
            or os.environ.get(
                "DATA_YAML_PATH",
                project_root / "data/data.example.yaml",
            )
        )

        yolo_weights_path = Path(
            yolo_weights_path
            or os.environ.get(
                "YOLO_WEIGHTS_PATH",
                project_root / "models/yolo/best.pt",
            )
        )

        self.class_names = load_class_names_from_yaml(str(data_yaml_path))

        if yolo_model is not None:
            self.yolo = yolo_model
        else:
            self.yolo = YoloModel(str(yolo_weights_path))

        yolo_class_names = self.yolo.get_class_names()

        if {
            c.strip().lower()
            for c in self.class_names
        } != {
            c.strip().lower()
            for c in yolo_class_names
        }:
            raise ValueError(
                "Class mismatch between data.example.yaml and YOLO model."
            )

        # Optional LLM advisor. The backend model is loaded lazily.
        self.llm_advisor = LLMAdvisor(
            use_llm=use_llm,
            known_classes=self.class_names,
        )

        # Optional task router.
        self.router = TaskRouter(self.llm_advisor) if use_router else None

        self.tasks = {
            "spatial": SpatialTask(
                yolo_model=self.yolo,
                llm_advisor=self.llm_advisor,
                class_names=self.class_names,
            ),
            "presence": PresenceTask(
                yolo_model=self.yolo,
                llm_advisor=self.llm_advisor,
                class_names=self.class_names,
            ),
            "measurement": MeasurementTask(),
        }

    @staticmethod
    def extract_question_from_prompt(prompt: str) -> str:
        """
        Extract the actual question from the unified external prompt.

        If the input does not contain a 'Question:' field, the original input
        is returned.
        """
        if prompt is None:
            return ""

        match = re.search(
            r"Question:\s*(.*?)(?:\n\s*\n|$)",
            prompt,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if match:
            return match.group(1).strip()

        return prompt.strip()

    def run(self, question: str, image=None):
        timer = Timer()

        timer.start("total")

        # ------------------------------------------------------------
        # External prompt handling
        # ------------------------------------------------------------
        raw_prompt = question
        core_question = self.extract_question_from_prompt(raw_prompt)

        # ------------------------------------------------------------
        # Task routing
        # ------------------------------------------------------------
        timer.start("routing")

        if self.router:
            task_type = self.router.route(core_question)
        else:
            task_type = "spatial"

        print("Task type:", task_type)

        timer.stop("routing")

        if task_type not in self.tasks:
            timer.stop("total")

            return AgentResult(
                query=None,
                result=False,
                explanation="Unsupported question type.",
                detections=None,
                confidence=0.0,
                audit_log={
                    "task": "unknown",
                    "raw_prompt": raw_prompt,
                    "extracted_question": core_question,
                },
            )

        # ------------------------------------------------------------
        # Task execution
        # ------------------------------------------------------------
        timer.start("task_execution")

        task = self.tasks[task_type]
        core_output = task.execute(core_question, image)

        timer.stop("task_execution")

        if core_output.get("error") is not None:
            timer.stop("total")

            return AgentResult(
                query=core_output.get("query"),
                result=False,
                explanation=core_output.get("error", "Task execution failed."),
                detections=core_output.get("detections"),
                confidence=0.0,
                audit_log={
                    "task": task_type,
                    "raw_prompt": raw_prompt,
                    "extracted_question": core_question,
                    "parse_source": core_output.get("parse_source"),
                    "task_audit": core_output.get("audit"),
                    "error": core_output.get("error"),
                },
            )

        # ------------------------------------------------------------
        # Optional report generation
        # ------------------------------------------------------------
        timer.start("report_generation")

        explanation = ""

        # Report generation is disabled for the reported benchmark.
        # if self.llm_advisor:
        #     explanation = self.llm_advisor.generate_report(**core_output)

        timer.stop("report_generation")
        timer.stop("total")

        return AgentResult(
            query=core_output["query"],
            result=core_output["result"],
            explanation=explanation,
            detections=core_output["detections"],
            confidence=1.0,
            audit_log={
                "task": task_type,
                "raw_prompt": raw_prompt,
                "extracted_question": core_question,
                "parse_source": core_output.get("parse_source"),
                "task_audit": core_output.get("audit"),
                "error": core_output.get("error"),
            },
        )