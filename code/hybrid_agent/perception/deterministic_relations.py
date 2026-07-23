def normalize_relation(relation):
    """
    Normalize supported spatial relation phrases.
    """
    relation = str(relation or "").strip().lower()

    if relation in {"left", "left of", "to the left of"}:
        return "left of"
    if relation in {"right", "right of", "to the right of"}:
        return "right of"
    if relation in {"above", "over", "superior to"}:
        return "above"
    if relation in {"below", "under", "inferior to"}:
        return "below"

    return relation


def check_relation(query, detections, return_debug=False):
    """
    Check a spatial relation deterministically using detected object centers.

    Args:
        query:
            RelationQuery-like object with entity_a, relation, and entity_b.
        detections:
            List of detection dictionaries containing at least:
            - class_name
            - confidence
            - center
        return_debug:
            If True, also return intermediate debug information.

    Returns:
        bool, or (bool, debug) if return_debug is True.
    """

    entity_a = str(query.entity_a).strip().lower()
    entity_b = str(query.entity_b).strip().lower()
    relation = normalize_relation(query.relation)

    # Build top-1 detection per class name.
    top1 = {}

    for det in detections:
        if not isinstance(det, dict):
            continue

        class_name = str(det.get("class_name", "")).strip().lower()
        if not class_name:
            continue

        conf = float(det.get("confidence", 0.0))

        if class_name not in top1 or conf > float(top1[class_name].get("confidence", 0.0)):
            top1[class_name] = det

    # Retrieve queried structures.
    obj_a = top1.get(entity_a)
    obj_b = top1.get(entity_b)

    debug = {
        "entity_a": entity_a,
        "entity_b": entity_b,
        "relation": relation,
        "top1_keys": sorted(list(top1.keys())),
        "obj_a_found": obj_a is not None,
        "obj_b_found": obj_b is not None,
        "obj_a": obj_a,
        "obj_b": obj_b,
        "dx": None,
        "dy": None,
        "raw_result": False,
    }

    if obj_a is None or obj_b is None:
        return (False, debug) if return_debug else False

    center_a = obj_a.get("center")
    center_b = obj_b.get("center")

    if not center_a or not center_b or len(center_a) < 2 or len(center_b) < 2:
        return (False, debug) if return_debug else False

    x1, y1 = center_a[:2]
    x2, y2 = center_b[:2]

    dx = float(x1) - float(x2)
    dy = float(y1) - float(y2)

    debug["dx"] = dx
    debug["dy"] = dy

    # Deterministic relation logic.
    if relation == "left of":
        result = dx < 0
    elif relation == "right of":
        result = dx > 0
    elif relation == "above":
        result = dy < 0
    elif relation == "below":
        result = dy > 0
    else:
        result = False

    debug["raw_result"] = bool(result)

    return (bool(result), debug) if return_debug else bool(result)