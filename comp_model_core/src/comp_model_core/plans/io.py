"""
comp_model_core.plans.io

I/O helpers for :mod:`comp_model_core.plans`.

This module loads JSON/YAML plans and normalizes the per-trial interface schedule.

Trial spec schedule (no defaults)
---------------------------------
Each :class:`~comp_model_core.plans.block.BlockPlan` must contain a fully-explicit
``trial_specs`` list with length ``n_trials``.

For convenience, plan files may instead specify:

- ``trial_spec_template``: a single trial-spec dict used as the base for every trial.
- ``trial_spec_overrides``: a length-``n_trials`` list of per-trial partial overrides
  (use ``null`` to keep the template).

These are expanded at load time into the canonical ``trial_specs`` list. No backward
compatibility shims are provided.

Subject expansion (optional)
----------------------------
To avoid repeating identical block structures across subjects, plans may specify
``subjects`` as a list of subject IDs and provide a block template at the top level:

- ``blocks_template``: list of block mappings used for every subject, **or**
- ``subject_template``: mapping with a ``blocks`` list.

Block ``block_id`` strings may include ``{subject}`` or ``{sid}`` placeholders,
which are formatted with the subject ID during expansion.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping, List
from pathlib import Path

from .block import BlockPlan, StudyPlan


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """
    Deep-merge a mapping into a base dictionary.

    The merge is recursive for ``dict`` values:

    - If both ``base[k]`` and ``override[k]`` are mappings, they are merged
      recursively.
    - Otherwise, ``override[k]`` replaces ``base[k]``.

    Lists (and other non-mapping values) are **replaced**, not concatenated.

    Parameters
    ----------
    base
        Base dictionary to merge into. This dictionary is not mutated.
    override
        Mapping of override values.

    Returns
    -------
    dict[str, Any]
        A new dictionary with ``override`` applied to ``base``.

    Notes
    -----
    This helper is used to apply per-trial ``trial_spec_overrides`` on top of a
    ``trial_spec_template`` when expanding plans.

    Examples
    --------
    >>> _deep_merge({"a": {"x": 1}, "b": [1, 2]}, {"a": {"y": 2}, "b": [9]})
    {'a': {'x': 1, 'y': 2}, 'b': [9]}
    """
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, Mapping):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _expand_trial_specs(block: Mapping[str, Any]) -> list[dict[str, Any]]:
    """
    Expand a block mapping into the canonical per-trial schedule.

    A block must provide *either*:

    1) ``trial_specs``: a list of length ``n_trials`` containing full trial-spec
       dictionaries, **or**
    2) ``trial_spec_template`` (+ optional ``trial_spec_overrides``): where the
       template is copied for each trial and then deep-merged with each override.

    Parameters
    ----------
    block
        Raw block mapping (typically parsed from JSON/YAML).

    Returns
    -------
    list[dict[str, Any]]
        The canonical ``trial_specs`` list of length ``n_trials``.

    Raises
    ------
    ValueError
        If ``n_trials`` is missing/invalid, if the schedule is missing, if both
        schedule styles are provided, or if list lengths/types do not match
        expectations.

    Notes
    -----
    - ``trial_spec_overrides`` must be a list of length ``n_trials``.
    - Each override entry may be ``None`` (YAML ``null``) to keep the template.
    - Overrides are applied using :func:`~comp_model_core.plans.io._deep_merge`.

    Examples
    --------
    Compact/original YAML block input:

    .. code-block:: yaml

       n_trials: 3
       trial_spec_template:
         self_outcome: {kind: VERIDICAL}
         available_actions: [0, 1]
       trial_spec_overrides:
         - null
         - {available_actions: [1]}
         - null

    Transformed/canonical YAML trial schedule:

    .. code-block:: yaml

       trial_specs:
         - self_outcome: {kind: VERIDICAL}
           available_actions: [0, 1]
         - self_outcome: {kind: VERIDICAL}
           available_actions: [1]
         - self_outcome: {kind: VERIDICAL}
           available_actions: [0, 1]
    """
    if "n_trials" not in block:
        raise ValueError("Block is missing required key 'n_trials'.")
    n_trials = int(block["n_trials"])
    if n_trials <= 0:
        raise ValueError("n_trials must be > 0")

    raw = block.get("trial_specs", None)
    template = block.get("trial_spec_template", None)
    overrides = block.get("trial_spec_overrides", None)

    if raw is not None:
        if template is not None or overrides is not None:
            raise ValueError(
                "Specify either 'trial_specs' OR ('trial_spec_template'/'trial_spec_overrides'), not both."
            )
        if not isinstance(raw, list):
            raise ValueError("trial_specs must be a list")
        if len(raw) != n_trials:
            raise ValueError("trial_specs length must equal n_trials")
        for i, ts in enumerate(raw):
            if not isinstance(ts, dict):
                raise ValueError(f"trial_specs[{i}] must be a dict")
        return raw

    # Template path
    if template is None:
        raise ValueError(
            "Missing required trial schedule. Provide either 'trial_specs' (length n_trials) "
            "or 'trial_spec_template' (+ optional 'trial_spec_overrides')."
        )
    if not isinstance(template, Mapping):
        raise ValueError("trial_spec_template must be a mapping/dict")

    specs = [deepcopy(dict(template)) for _ in range(n_trials)]

    if overrides is not None:
        if not isinstance(overrides, list):
            raise ValueError("trial_spec_overrides must be a list (use null entries to keep template)")
        if len(overrides) != n_trials:
            raise ValueError("trial_spec_overrides length must equal n_trials")
        for i, ov in enumerate(overrides):
            if ov is None:
                continue
            if not isinstance(ov, Mapping):
                raise ValueError(f"trial_spec_overrides[{i}] must be a mapping or null")
            specs[i] = _deep_merge(specs[i], ov)

    return specs


def _normalize_block_dict(b: Mapping[str, Any]) -> dict[str, Any]:
    """
    Normalize a raw block mapping into :class:`~comp_model_core.plans.block.BlockPlan` kwargs.

    This function expands any template-based trial schedule into the canonical
    ``trial_specs`` list and removes convenience keys so that
    ``BlockPlan(**out)`` succeeds.

    Parameters
    ----------
    b
        Raw block mapping.

    Returns
    -------
    dict[str, Any]
        A dictionary suitable for ``BlockPlan(**kwargs)``.

    Raises
    ------
    ValueError
        If ``b`` is not a mapping or contains an invalid trial schedule.
    """
    if not isinstance(b, Mapping):
        raise ValueError("Each block must be a mapping")

    out = dict(b)
    out["trial_specs"] = _expand_trial_specs(out)
    # Remove convenience keys so BlockPlan(**out) works.
    out.pop("trial_spec_template", None)
    out.pop("trial_spec_overrides", None)
    return out


def _expand_subjects(raw: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    subjects_raw = raw.get("subjects")
    if isinstance(subjects_raw, int):
        if subjects_raw <= 0:
            raise ValueError("subjects integer must be > 0.")
        width = max(2, len(str(subjects_raw)))
        subjects_raw = [f"s{i:0{width}d}" for i in range(1, subjects_raw + 1)]
    if isinstance(subjects_raw, dict):
        return subjects_raw

    if isinstance(subjects_raw, list):
        template = raw.get("blocks_template")
        if template is None:
            template = raw.get("subject_template")

        if template is None:
            raise ValueError("subjects list requires 'blocks_template' or 'subject_template'.")

        if isinstance(template, Mapping):
            template = template.get("blocks")

        if not isinstance(template, list):
            raise ValueError("Template must be a list of block mappings (or subject_template.blocks).")

        out: dict[str, list[Mapping[str, Any]]] = {}
        for sid in subjects_raw:
            if not isinstance(sid, (str, int)):
                raise ValueError("subjects list must contain only strings or integers.")
            sid_str = str(sid)
            blocks_out: list[Mapping[str, Any]] = []
            for b in template:
                if not isinstance(b, Mapping):
                    raise ValueError("Each template block must be a mapping/object.")
                repeat_raw = b.get("repeat", None)
                if repeat_raw is None:
                    repeat = 1
                else:
                    try:
                        repeat = int(repeat_raw)
                    except Exception as e:
                        raise ValueError("repeat must be an integer.") from e
                    if repeat <= 0:
                        raise ValueError("repeat must be > 0.")
                for rep in range(1, repeat + 1):
                    nb = deepcopy(b)
                    nb.pop("repeat", None)
                    bid = nb.get("block_id")
                    if isinstance(bid, str):
                        if "{subject}" in bid or "{sid}" in bid or "{rep}" in bid:
                            nb["block_id"] = bid.format(subject=sid_str, sid=sid_str, rep=rep)
                        elif repeat > 1:
                            nb["block_id"] = f"{bid}_r{rep}"
                    blocks_out.append(nb)
            out[sid_str] = blocks_out
        return out

    raise ValueError("Input must contain key 'subjects' as an object mapping or a list of subject IDs.")


def study_plan_from_dict(raw: Mapping[str, Any]) -> StudyPlan:
    """
    Construct a :class:`~comp_model_core.plans.block.StudyPlan` from a mapping.

    Parameters
    ----------
    raw
        Root plan mapping (typically parsed from JSON/YAML) with at least a
        ``subjects`` key mapping subject IDs to block lists.

    Returns
    -------
    StudyPlan
        The constructed study plan.

    Raises
    ------
    ValueError
        If ``subjects`` is missing or not a mapping, if a subject's block list is
        not a list, or if any block cannot be normalized/validated.

    Notes
    -----
    The returned plan will always have canonical block schedules: each
    :class:`~comp_model_core.plans.block.BlockPlan` will contain a fully-expanded
    ``trial_specs`` list of length ``n_trials``.

    Examples
    --------
    Compact/original YAML input:

    .. code-block:: yaml

       subjects: [S1]
       blocks_template:
         - block_id: "b_{subject}"
           condition: "A"
           n_trials: 3
           bandit_type: "BernoulliBanditEnv"
           bandit_config: {probs: [0.2, 0.8]}
           trial_spec_template:
             self_outcome: {kind: VERIDICAL}
             available_actions: [0, 1]
           trial_spec_overrides:
             - null
             - {available_actions: [1]}
             - null

    Transformed/canonical YAML shape after expansion:

    .. code-block:: yaml

       subjects:
         S1:
           - block_id: "b_S1"
             condition: "A"
             n_trials: 3
             bandit_type: "BernoulliBanditEnv"
             bandit_config: {probs: [0.2, 0.8]}
             trial_specs:
               - self_outcome: {kind: VERIDICAL}
                 available_actions: [0, 1]
               - self_outcome: {kind: VERIDICAL}
                 available_actions: [1]
               - self_outcome: {kind: VERIDICAL}
                 available_actions: [0, 1]

    Two-block compact YAML input:

    .. code-block:: yaml

       subjects: [S1]
       blocks_template:
         - block_id: "train_{subject}"
           condition: "A"
           n_trials: 2
           bandit_type: "BernoulliBanditEnv"
           bandit_config: {probs: [0.2, 0.8]}
           trial_spec_template:
             self_outcome: {kind: VERIDICAL}
             available_actions: [0, 1]

         - block_id: "test_{subject}"
           condition: "B"
           n_trials: 2
           bandit_type: "BernoulliBanditEnv"
           bandit_config: {probs: [0.6, 0.4]}
           trial_spec_template:
             self_outcome: {kind: VERIDICAL}
             available_actions: [0, 1]

    Two-block transformed/canonical YAML shape:

    .. code-block:: yaml

       subjects:
         S1:
           - block_id: "train_S1"
             condition: "A"
             n_trials: 2
             bandit_type: "BernoulliBanditEnv"
             bandit_config: {probs: [0.2, 0.8]}
             trial_specs:
               - self_outcome: {kind: VERIDICAL}
                 available_actions: [0, 1]
               - self_outcome: {kind: VERIDICAL}
                 available_actions: [0, 1]

           - block_id: "test_S1"
             condition: "B"
             n_trials: 2
             bandit_type: "BernoulliBanditEnv"
             bandit_config: {probs: [0.6, 0.4]}
             trial_specs:
               - self_outcome: {kind: VERIDICAL}
                 available_actions: [0, 1]
               - self_outcome: {kind: VERIDICAL}
                 available_actions: [0, 1]
    """
    if "subjects" not in raw:
        raise ValueError("Input must contain key 'subjects'.")

    subjects_raw = _expand_subjects(raw)
    subjects: dict[str, list[BlockPlan]] = {}
    for sid, blocks in subjects_raw.items():
        if not isinstance(blocks, list):
            raise ValueError(f"subjects[{sid}] must be a list.")
        subjects[str(sid)] = [BlockPlan(**_normalize_block_dict(b)) for b in blocks]

    metadata = raw.get("metadata", {})
    return StudyPlan(subjects=subjects, metadata=metadata if isinstance(metadata, dict) else {})


def load_study_plan_json(path: str | Path) -> StudyPlan:
    """
    Load a study plan from a JSON file.

    Parameters
    ----------
    path
        Path to the JSON plan file.

    Returns
    -------
    StudyPlan
        The loaded study plan.

    Raises
    ------
    OSError
        If the file cannot be opened.
    json.JSONDecodeError
        If the file is not valid JSON.
    ValueError
        If the loaded object is not a valid plan mapping.
    """
    with open(str(path), "r", encoding="utf-8") as f:
        raw = json.load(f)
    return study_plan_from_dict(raw)


def load_study_plan_yaml(path: str | Path) -> StudyPlan:
    """
    Load a study plan from a YAML file.

    This function uses ``yaml.safe_load`` (PyYAML).

    Parameters
    ----------
    path
        Path to the YAML plan file.

    Returns
    -------
    StudyPlan
        The loaded study plan.

    Raises
    ------
    ImportError
        If PyYAML is not installed.
    OSError
        If the file cannot be opened.
    ValueError
        If the YAML root is not a mapping/object or is not a valid plan mapping.

    Notes
    -----
    Install PyYAML with::

        pip install pyyaml
    """
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError("PyYAML is required to load YAML plans. Install with: pip install pyyaml") from e

    with open(str(path), "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("YAML root must be a mapping/object.")
    return study_plan_from_dict(raw)


def load_study_plan(path: str | Path) -> StudyPlan:
    """
    Load a study plan from a JSON or YAML file.

    Dispatches by file extension (case-insensitive):
    ``.json`` -> :func:`load_study_plan_json`
    ``.yaml``/``.yml`` -> :func:`load_study_plan_yaml`.

    Parameters
    ----------
    path : str | Path
        Path to the study plan file.

    Returns
    -------
    StudyPlan
        The loaded study plan.

    Raises
    ------
    ValueError
        If ``path`` does not end with ``.json``, ``.yaml``, or ``.yml``.
    OSError
        If the file cannot be opened.
    ImportError
        If loading YAML but PyYAML is not installed.
    json.JSONDecodeError
        If a JSON file is not valid JSON.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    # Load plan
    if suffix in {".yaml", ".yml"}:
        return load_study_plan_yaml(str(p))
    if suffix == ".json":
        return load_study_plan_json(str(p))

    raise ValueError(f"path must end with .yaml/.yml or .json, got: {p.suffix}")
    

def infer_load_conditions(plan_path: str | Path) -> List[str]:
    """
    Load StudyPlan and infer unique condition labels in order of first appearance.

    No-default philosophy:
    - every block must have a non-empty condition string
    - raises ValueError if any is missing/blank
    """
    p = Path(plan_path).expanduser().resolve()
    plan = load_study_plan(p)

    conditions: list[str] = []
    seen: set[str] = set()

    # StudyPlan.subjects is typically: dict[subj_id, list[BlockPlan]]
    for _, blocks in plan.subjects.items():
        for b in blocks:
            cond = getattr(b, "condition", None)
            if cond is None or (isinstance(cond, str) and cond.strip() == ""):
                raise ValueError(
                    f"Block is missing required condition (no-default philosophy). "
                    f"Offending block: {b}"
                )
            cond = str(cond)
            if cond not in seen:
                seen.add(cond)
                conditions.append(cond)

    if not conditions:
        raise ValueError("No conditions found in plan (unexpected).")

    return conditions
