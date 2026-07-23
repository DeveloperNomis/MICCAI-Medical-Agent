# deterministic_spatial.py

"""
Thin wrapper functions for the deterministic spatial verification pathway.

These helpers keep detection and geometric verification separate:
1. run object detection,
2. verify the requested spatial relation using deterministic geometry.
"""


def run_detection(yolo_model, image):
    """
    Run the object detector on an input image.
    """
    return yolo_model.detect(image)


def run_geometric_verification(geometry_engine, query, detections):
    """
    Verify a structured spatial query using deterministic geometry.
    """
    return geometry_engine.verify(query, detections)