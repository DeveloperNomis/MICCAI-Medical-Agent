# Evaluation

This document describes the evaluation protocol used for the reported results.

## Task

Each sample consists of a CT slice and a binary spatial question.

Example:

```text
Is the liver left of the spleen?
```

The expected answer is:

```text
1 = Yes
0 = No
```

## Strict binary evaluation

All systems are evaluated under a strict binary protocol.

Valid outputs are:

```text
0
1
```

Invalid, missing, failed, or non-binary outputs are counted as incorrect.

## Metrics

The reported metrics are:

- accuracy,
- precision,
- recall,
- F1,
- invalid output rate.

Accuracy and F1 are reported in percent in the paper.

## Failure attribution

For hybrid-agent predictions, incorrect or invalid outputs are assigned to the earliest identifiable failing stage.

Failure stages:

| Stage | Meaning |
|---|---|
| Question extraction | The full prompt could not be converted into the core spatial question. |
| Routing | The query was routed to the wrong task pathway. |
| Parsing / query extraction | The wrong entity or relation was extracted. |
| Ontology matching | An extracted entity could not be mapped to a detector class. |
| Missing detection | At least one queried anatomical structure was not detected. |
| Imprecise localization | A detected center was too imprecise for the relation. |
| Geometry ambiguity | The spatial configuration was borderline or ambiguous. |
| Formatting/runtime | The system produced an invalid output or failed during execution. |

For LLM-fallback cases, failures resulting in an incorrect detector-aligned query are conservatively attributed to parsing/query extraction.

## Localization threshold

Imprecise localization is assigned when both queried structures are detected but the selected detection center deviates by more than 30 pixels from the corresponding reference center.

This threshold is used only for failure attribution and does not affect the reported accuracy or F1 scores.