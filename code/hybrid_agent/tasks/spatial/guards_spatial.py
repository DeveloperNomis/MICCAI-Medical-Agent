# guards_spatial.py

"""
Validation helpers for spatial relation queries.
"""

VALID_RELATIONS = {
    "above",
    "over",
    "superior to",
    "below",
    "under",
    "inferior to",
    "left of",
    "right of",
    "to the left of",
    "to the right of",
    "lateral to",
    "medial to",
}


def normalize_relation(relation):
    """
    Normalize relation strings for validation.
    """
    return str(relation or "").strip().lower()


def validate_query(query):
    """
    Validate that a spatial query contains two entities and a supported relation.

    Returns the query if valid, otherwise None.
    """
    if not query:
        return None

    relation = normalize_relation(getattr(query, "relation", None))
    entity_a = getattr(query, "entity_a", None)
    entity_b = getattr(query, "entity_b", None)

    if relation not in VALID_RELATIONS:
        return None

    if not entity_a or not entity_b:
        return None

    return query