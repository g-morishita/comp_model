from __future__ import annotations

import json
from typing import Any, Mapping

from .block import BlockPlan, StudyPlan


def study_plan_from_dict(raw: Mapping[str, Any]) -> StudyPlan:
    """
    Construct StudyPlan from a Python mapping.

    Expected shape:
    {
      "subjects": { "s1": [ {...blockplan...}, ... ], "s2": [...] },
      "metadata": {...}   # optional
    }
    """
    if "subjects" not in raw or not isinstance(raw["subjects"], dict):
        raise ValueError("Input must contain key 'subjects' as an object mapping subject_id -> block list.")

    subjects: dict[str, list[BlockPlan]] = {}
    for sid, blocks in raw["subjects"].items():
        if not isinstance(blocks, list):
            raise ValueError(f"subjects[{sid}] must be a list.")
        subjects[str(sid)] = [BlockPlan(**b) for b in blocks]

    metadata = raw.get("metadata", {})
    return StudyPlan(subjects=subjects, metadata=metadata if isinstance(metadata, dict) else {})


def load_study_plan_json(path: str) -> StudyPlan:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return study_plan_from_dict(raw)


def load_study_plan_yaml(path: str) -> StudyPlan:
    """
    Load StudyPlan from YAML.

    Requires dependency: PyYAML (pip install pyyaml)
    Uses safe_load to avoid executing arbitrary YAML tags.
    """
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError(
            "PyYAML is required to load YAML plans. Install with: pip install pyyaml"
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("YAML root must be a mapping/object.")
    return study_plan_from_dict(raw)
