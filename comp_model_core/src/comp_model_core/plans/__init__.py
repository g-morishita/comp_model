"""
Simulation plan schemas and I/O helpers.

A *plan* specifies what to simulate (which tasks/bandits, how many trials, and
optional demonstrators) without hard-coding those choices in Python scripts.

Typical workflow
----------------
1. Write a plan in JSON or YAML.
2. Load it with :func:`comp_model_core.plans.io.load_study_plan_json` or
   :func:`comp_model_core.plans.io.load_study_plan_yaml`.
3. Pass it to a :class:`comp_model_core.interfaces.generator.Generator`.

See Also
--------
comp_model_core.plans.block.BlockPlan
comp_model_core.plans.block.StudyPlan
"""

from .block import BlockPlan, StudyPlan
from .io import load_study_plan_json, load_study_plan_yaml, study_plan_from_dict, infer_load_conditions

__all__ = [
    "BlockPlan",
    "StudyPlan",
    "load_study_plan_json",
    "load_study_plan_yaml",
    "study_plan_from_dict",
    "infer_load_conditions"
]
