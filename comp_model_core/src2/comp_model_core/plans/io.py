"""
I/O helpers for :mod:`comp_model_core.plans`.

The functions in this module load plan files from disk and validate their basic shape
before constructing :class:`~comp_model_core.plans.block.BlockPlan` and
:class:`~comp_model_core.plans.block.StudyPlan` objects.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from .block import BlockPlan, StudyPlan


def study_plan_from_dict(raw: Mapping[str, Any]) -> StudyPlan:
    """
    Construct a :class:`~comp_model_core.plans.block.StudyPlan` from a mapping.

    Parameters
    ----------
    raw : Mapping[str, Any]
        Mapping with keys ``"subjects"`` and (optionally) ``"metadata"``.

        Expected structure::

            {
              "subjects": {
                "s1": [ { ...blockplan... }, ... ],
                "s2": [ ... ]
              },
              "metadata": { ... }  # optional
            }

    Returns
    -------
    StudyPlan
        Parsed study plan.

    Raises
    ------
    ValueError
        If the structure is invalid or required keys are missing.
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
    """
    Load a study plan from a JSON file.

    Parameters
    ----------
    path : str
        Path to a JSON file.

    Returns
    -------
    StudyPlan
        Parsed plan.

    Raises
    ------
    OSError
        If the file cannot be read.
    ValueError
        If the file content is not valid JSON or does not match the expected structure.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return study_plan_from_dict(raw)


def load_study_plan_yaml(path: str) -> StudyPlan:
    """
    Load a study plan from a YAML file.

    This function uses ``yaml.safe_load`` to avoid executing arbitrary YAML tags.

    Parameters
    ----------
    path : str
        Path to a YAML file.

    Returns
    -------
    StudyPlan
        Parsed plan.

    Raises
    ------
    ImportError
        If PyYAML is not installed.
    OSError
        If the file cannot be read.
    ValueError
        If the YAML root is not a mapping or if the plan structure is invalid.
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
