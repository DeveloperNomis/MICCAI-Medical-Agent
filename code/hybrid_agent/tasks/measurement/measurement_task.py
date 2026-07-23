# tasks/measurement/measurement_task.py

from tasks.base_task import BaseTask


class MeasurementTask(BaseTask):
    """
    Placeholder for future measurement-oriented reasoning.

    Measurement reasoning is part of the modular task architecture but is not
    used in the experiments reported in the paper.
    """

    def execute(self, question, image):
        raise NotImplementedError(
            "MeasurementTask is a placeholder and is not used in the reported experiments."
        )