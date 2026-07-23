import os

import torch
import torch.nn.functional as F
from langchain.prompts import PromptTemplate


class TaskRouter:
    """
    Routes a natural-language question to one of the supported task pathways:
    spatial verification, measurement, or presence verification.

    Deterministic keyword routing is attempted first. If that fails and the LLM
    advisor is enabled, the router falls back to LLM-based label scoring.
    """

    def __init__(self, llm_advisor, threshold=0.5):
        self.llm_advisor = llm_advisor
        self.threshold = threshold

        self.labels = ["A", "B", "C"]

        self.label_map = {
            "A": "spatial",
            "B": "measurement",
            "C": "presence",
        }

        self.debug = os.environ.get("DEBUG_ROUTER", "0") == "1"

        self.prompt = PromptTemplate(
            input_variables=["question"],
            template="""
You are a medical task classifier.

Classify the following question into exactly one of these categories:

- A: spatial
- B: measurement
- C: presence

Respond with ONLY A, B, or C.
Do not explain.
Do not add examples.
Do not add text.

Question: {question}

Answer:
""",
        )

    def _deterministic_route(self, question: str):
        """
        Route simple questions using deterministic keyword rules.
        """
        question_lower = question.lower()

        if any(
            word in question_lower
            for word in ["left", "right", "above", "below", "lateral", "medial"]
        ):
            return "spatial"

        if any(
            word in question_lower
            for word in ["size", "length", "diameter", "measure"]
        ):
            return "measurement"

        if any(
            word in question_lower
            for word in ["present", "visible", "exist"]
        ):
            return "presence"

        return None

    def route(self, question: str):
        """
        Return the predicted task type for a question.
        """
        task = self._deterministic_route(question)

        if task is not None:
            return task

        if not self.llm_advisor.use_llm:
            raise ValueError(
                "Task could not be determined: LLM is disabled and deterministic routing failed."
            )

        self.llm_advisor._ensure_model_loaded()

        formatted_prompt = self.prompt.format(question=question)

        scores = self.llm_advisor.llm.score_next_token_options(
            formatted_prompt,
            self.labels,
        )

        # score_next_token_options returns log probabilities.
        # Normalize over A/B/C to obtain routing probabilities.
        label_scores = torch.tensor(
            [scores[label] for label in self.labels],
            dtype=torch.float32,
        )

        probs = F.softmax(label_scores, dim=0)

        confidence, idx = torch.max(probs, dim=0)

        chosen_label = self.labels[idx.item()]
        predicted_task = self.label_map[chosen_label]
        confidence = confidence.item()

        if self.debug:
            print("Routing probabilities:", {
                self.labels[i]: round(probs[i].item(), 3)
                for i in range(len(self.labels))
            })

            print(
                "Chosen task:",
                predicted_task,
                "Label:",
                chosen_label,
                "Confidence:",
                round(confidence, 3),
            )

        if confidence < self.threshold:
            raise ValueError(
                f"Low routing confidence ({confidence:.2f})."
            )

        return predicted_task