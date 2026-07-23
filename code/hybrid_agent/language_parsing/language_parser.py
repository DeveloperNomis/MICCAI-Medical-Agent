import re
from typing import List, Optional

import spacy

from agent.contracts import RelationQuery


nlp = spacy.load("en_core_web_sm")


RELATIONS = {
    "above": ["above", "over", "superior to"],
    "below": ["below", "under", "inferior to"],
    "left of": ["left of", "to the left of"],
    "right of": ["right of", "to the right of"],
    "lateral to": ["lateral to"],
    "medial to": ["medial to"],
}


def clean_input(text: str) -> str:
    """
    Normalize whitespace and minor punctuation artifacts.
    """
    text = re.sub(r"\s+", " ", text.strip())
    text = text.replace(" ?", "?")
    return text


def normalize_question(question: str) -> str:
    """
    Ensure a non-empty question starts with an uppercase character.
    """
    return question[0].upper() + question[1:] if question else question


def clean_phrase(phrase: str) -> str:
    """
    Remove leading articles from extracted noun phrases.
    """
    return re.sub(r"^(the|a|an)\s+", "", phrase.strip(), flags=re.IGNORECASE)


def resolve_entity(
    candidate: str,
    known_classes: List[str],
    detected_classes: Optional[List[str]] = None,
):
    """
    Resolve an extracted entity phrase against known detector classes.

    Matching is performed using exact matching first and unique substring
    matching as a fallback.
    """
    candidate_norm = candidate.strip().lower()
    if not candidate_norm:
        return None

    known_classes_norm = [k.strip().lower() for k in known_classes]
    detected_classes_norm = (
        [d.strip().lower() for d in detected_classes]
        if detected_classes
        else []
    )

    # 1. Exact match in known classes.
    if candidate_norm in known_classes_norm:
        return candidate_norm

    # 2. Exact match in detected classes.
    if candidate_norm in detected_classes_norm:
        return candidate_norm

    # 3. Unique substring match in known classes.
    known_hits = [
        k for k in known_classes_norm
        if candidate_norm in k or k in candidate_norm
    ]
    if len(known_hits) == 1:
        return known_hits[0]

    # 4. Unique substring match in detected classes.
    detected_hits = [
        d for d in detected_classes_norm
        if candidate_norm in d or d in candidate_norm
    ]
    if len(detected_hits) == 1:
        return detected_hits[0]

    return None


def extract_triplet(question: str, known_classes: List[str]):
    """
    Deterministically extract a spatial relation triplet from a question.

    Returns:
        RelationQuery(entity_a, relation, entity_b), or None if extraction fails.
    """
    question = clean_input(question)
    question_lower = question.lower()

    doc = nlp(normalize_question(question_lower))

    # 1. Find relation.
    relation = ""
    for canonical_relation, variants in RELATIONS.items():
        for variant in variants:
            if re.search(rf"\b{re.escape(variant)}\b", question_lower):
                relation = canonical_relation
                break
        if relation:
            break

    # 2. Extract noun phrases, optionally including preceding adjectives.
    noun_phrases = []

    for chunk in doc.noun_chunks:
        start = chunk.start
        phrase = chunk.text

        if (
            start > 0
            and doc[start - 1].pos_ == "ADJ"
            and not any(x in phrase for x in ["left", "right"])
        ):
            phrase = doc[start - 1].text + " " + phrase

        noun_phrases.append(clean_phrase(phrase))

    if len(noun_phrases) >= 2:
        object1 = noun_phrases[0]
        object2 = noun_phrases[-1]
    else:
        # Fallback if fewer than two noun phrases were found.
        nouns = [
            clean_phrase(token.text)
            for token in doc
            if token.pos_ == "NOUN"
        ]
        object1, object2 = (nouns + ["", ""])[:2]

    object1 = object1.strip()
    object2 = object2.strip()
    relation = relation.strip()

    if not object1 or not object2 or not relation:
        return None

    # Resolve extracted entities against the known detector classes.
    resolved_object1 = resolve_entity(object1, known_classes)
    resolved_object2 = resolve_entity(object2, known_classes)

    if not resolved_object1 or not resolved_object2:
        return None

    return RelationQuery(resolved_object1, relation, resolved_object2)


def extract_presence_entity(question: str, known_classes: List[str]):
    """
    Deterministically extract the queried anatomical entity for presence checks.

    Returns:
        The resolved class name, or None if extraction fails.
    """
    question = clean_input(question)
    doc = nlp(normalize_question(question.lower()))

    # Prefer noun chunks first.
    noun_phrases = []

    for chunk in doc.noun_chunks:
        start = chunk.start
        phrase = chunk.text

        if (
            start > 0
            and doc[start - 1].pos_ == "ADJ"
            and not any(x in phrase for x in ["left", "right"])
        ):
            phrase = doc[start - 1].text + " " + phrase

        noun_phrases.append(clean_phrase(phrase).strip())

    for candidate in noun_phrases:
        resolved = resolve_entity(candidate, known_classes)
        if resolved:
            return resolved

    # Fallback to single nouns.
    for token in doc:
        if token.pos_ == "NOUN":
            candidate = clean_phrase(token.text).strip()
            resolved = resolve_entity(candidate, known_classes)
            if resolved:
                return resolved

    return None