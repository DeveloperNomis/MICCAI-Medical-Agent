# llm_advisor.py

import json
import os
import re
from typing import List, Optional

from langchain.prompts import PromptTemplate

from agent.contracts import RelationQuery
from agent.llm_model import LocalLLM
from agent.langchain_adapter import LocalLLMAdapter
from utils.timer import Timer


class LLMAdvisor:
    """
    Optional LLM-based advisor for fallback parsing and report generation.

    The LLM is loaded lazily and is not used to directly decide spatial
    relations from the image. Spatial decisions remain deterministic and are
    computed from object detections and geometric rules.
    """

    def __init__(self, use_llm=True, known_classes: Optional[List[str]] = None):
        self.use_llm = use_llm
        self.known_classes = [
            c.strip().lower()
            for c in (known_classes or [])
        ]

        # Lazy-loaded backend model.
        self.llm = None
        self.langchain_llm = None

        # Lazy-created LangChain chains.
        self.parse_chain = None
        self.presence_chain = None
        self.measurement_chain = None
        self.report_chain = None

        self.debug = os.environ.get("DEBUG_LLM_ADVISOR", "0") == "1"

    def _ensure_model_loaded(self):
        """
        Load the local LLM backend only when it is needed.
        """
        if not self.use_llm:
            return

        if self.llm is not None:
            return

        timer = Timer()
        timer.start("LLM loading")

        print("Loading local LLM backend...")
        self.llm = LocalLLM()
        self.langchain_llm = LocalLLMAdapter(self.llm)

        timer.stop("LLM loading")

    def _ensure_spatial_chain(self):
        """
        Create the spatial parsing chain if it does not exist yet.
        """
        if self.parse_chain is not None:
            return

        self._ensure_model_loaded()

        class_list = "\n".join(f"- {c}" for c in self.known_classes)

        template = f"""
Extract the spatial relation from the question.

You must choose entity_a and entity_b strictly from this list:

{class_list}

You must choose one relation strictly from this list:

- above
- below
- left of
- right of
- lateral to
- medial to

Return strictly valid JSON in this format:
{{
  "entity_a": "...",
  "relation": "...",
  "entity_b": "..."
}}

Question: {{question}}
"""

        parse_prompt = PromptTemplate(
            input_variables=["question"],
            template=template,
        )

        self.parse_chain = parse_prompt | self.langchain_llm

    def _ensure_presence_chain(self):
        """
        Create the presence parsing chain if it does not exist yet.
        """
        if self.presence_chain is not None:
            return

        self._ensure_model_loaded()

        class_list = "\n".join(f"- {c}" for c in self.known_classes)

        template = f"""
Extract the anatomical structure mentioned in the question.

You must choose strictly one structure from this list:

{class_list}

Return ONLY the exact name.
Do not explain.

Question: {{question}}
"""

        presence_prompt = PromptTemplate(
            input_variables=["question"],
            template=template,
        )

        self.presence_chain = presence_prompt | self.langchain_llm

    def _ensure_report_chain(self):
        """
        Create the report generation chain if it does not exist yet.
        """
        if self.report_chain is not None:
            return

        self._ensure_model_loaded()

        report_prompt = PromptTemplate(
            input_variables=[
                "query",
                "result",
                "detections",
                "confidence",
                "audit_log",
                "parse_source",
            ],
            template="""
You are a radiological reporting assistant.

Generate a concise, evidence-grounded report from the structured agent output.
Do not invent findings. Use only the provided structured information.

Structured Agent Output:
- Query: {query}
- Result: {result}
- Detections: {detections}
- Confidence: {confidence}
- Parsing Source: {parse_source}
- Audit Log: {audit_log}

Write the report with these sections:
1. Finding
2. Image Evidence
3. Geometric Verification
4. Conclusion
5. Limitations, if any
""",
        )

        self.report_chain = report_prompt | self.langchain_llm

    @staticmethod
    def _response_to_text(response) -> str:
        """
        Convert a LangChain/model response to plain text.
        """
        if response is None:
            return ""

        if hasattr(response, "content"):
            return str(response.content)

        return str(response)

    def extract_json_from_response(self, response):
        """
        Extract a JSON object from an LLM response.

        Markdown code fences are removed if present.
        """
        response_text = self._response_to_text(response)

        # Remove markdown fences.
        response_text = re.sub(r"```json", "", response_text)
        response_text = re.sub(r"```", "", response_text)

        # Extract JSON block.
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None

    def suggest_spatial(self, question: str):
        """
        Use the LLM as a fallback parser for spatial relation questions.
        """
        if not self.use_llm:
            return None

        self._ensure_spatial_chain()
        response = self.parse_chain.invoke({"question": question})

        if self.debug:
            print("LLM raw spatial response:\n", response)

        data = self.extract_json_from_response(response)

        if not data:
            return None

        entity_a = data.get("entity_a", "").strip().lower()
        relation = data.get("relation", "").strip().lower()
        entity_b = data.get("entity_b", "").strip().lower()

        if entity_a not in self.known_classes:
            return None

        if entity_b not in self.known_classes:
            return None

        allowed_relations = [
            "above",
            "below",
            "left of",
            "right of",
            "lateral to",
            "medial to",
        ]

        if relation not in allowed_relations:
            return None

        if self.debug:
            print("LLM parsed spatial query:", entity_a, relation, entity_b)

        return RelationQuery(
            entity_a=entity_a,
            relation=relation,
            entity_b=entity_b,
        )

    def suggest_presence(self, question: str):
        """
        Use the LLM as a fallback parser for presence questions.
        """
        if not self.use_llm:
            return None

        self._ensure_presence_chain()
        response = self.presence_chain.invoke({"question": question})

        entity = self._response_to_text(response).strip().lower()

        if entity not in self.known_classes:
            return None

        return entity

    def review_detections(self, detections):
        """
        Optional qualitative detection review.

        This is not required for the reported benchmark and should not be used
        for computing deterministic spatial decisions.
        """
        if not self.use_llm:
            return None

        self._ensure_model_loaded()

        prompt = f"""
Review object detections for plausibility:
{detections}
"""

        return self.langchain_llm.invoke(prompt)

    def generate_report(
        self,
        query,
        detections,
        result,
        parse_source=None,
        confidence=None,
        audit_log=None,
    ):
        """
        Generate a concise textual report from structured agent outputs.
        """
        if not self.use_llm:
            return None

        self._ensure_report_chain()

        return self.report_chain.invoke({
            "query": json.dumps(query, indent=2, ensure_ascii=False, default=str),
            "result": json.dumps(result, indent=2, ensure_ascii=False, default=str),
            "detections": json.dumps(detections, indent=2, ensure_ascii=False, default=str),
            "confidence": confidence if confidence is not None else "not provided",
            "audit_log": (
                json.dumps(audit_log, indent=2, ensure_ascii=False, default=str)
                if audit_log is not None
                else "not provided"
            ),
            "parse_source": parse_source if parse_source is not None else "not provided",
        })