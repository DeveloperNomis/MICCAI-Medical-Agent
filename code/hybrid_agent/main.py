from agent.controller import AgentController

import json
import os
from dataclasses import asdict
from pathlib import Path


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

# Example image and question for single-case agent inference.
# Replace these paths with local data paths before running.
IMAGE_PATH = os.environ.get(
    "IMAGE_PATH",
    "/path/to/example_image.png",
)

QUESTION = os.environ.get(
    "QUESTION",
    "Is the liver left of the spleen?",
)

OUTPUT_DIR = Path(
    os.environ.get(
        "OUTPUT_DIR",
        "../../results/inference/agent",
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    print("=== Running hybrid-agent single-case inference ===")

    controller = AgentController(use_llm=True, use_router=True)

    result = controller.run(
        question=QUESTION,
        image=IMAGE_PATH,
    )

    print("\n=== Agent result ===")
    print(result)

    print("\n=== Parsed query ===")
    print("Query:", result.query)

    print("\n=== Audit log ===")
    print("Audit log:", result.audit_log)

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    output_file = OUTPUT_DIR / f"output_{job_id}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=4, ensure_ascii=False)

    print("\nSaved result to:", output_file)


if __name__ == "__main__":
    main()