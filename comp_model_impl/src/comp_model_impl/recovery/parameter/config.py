"""Configuration for parameter recovery experiments.

This module defines dataclasses used to configure parameter recovery runs and
utilities for loading them from YAML/JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import json


@dataclass(frozen=True, slots=True)
class DistSpec:
    """
    Distribution specification (scipy.stats).

    Parameters
    ----------
    name : str
        Name of the scipy.stats distribution (e.g., ``"norm"``, ``"beta"``) or
        ``"constant"`` for a fixed value draw.
    args : dict[str, Any], optional
        Keyword arguments passed to the distribution constructor.

    Examples
    --------
    >>> DistSpec(name="beta", args={"a": 2.0, "b": 2.0})
    DistSpec(name='beta', args={'a': 2.0, 'b': 2.0})
    >>> DistSpec(name="lognorm", args={"s": 0.4, "scale": 4.0})
    DistSpec(name='lognorm', args={'s': 0.4, 'scale': 4.0})
    """
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConditionSamplingSpec:
    """
    Per-condition sampling overrides (base-model parameter space).

    Parameters
    ----------
    individual : dict[str, DistSpec]
        Per-parameter distributions for independent sampling.
    population : dict[str, DistSpec]
        Per-parameter distributions for hierarchical population centers.
    individual_sd : dict[str, float]
        Per-parameter standard deviations used for hierarchical subject draws.
    fixed : dict[str, float]
        Fixed parameter values (used when ``mode="fixed"``).
    """

    individual: dict[str, DistSpec] = field(default_factory=dict)
    population: dict[str, DistSpec] = field(default_factory=dict)
    individual_sd: dict[str, float] = field(default_factory=dict)
    fixed: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SamplingSpec:
    """
    How to sample true parameters.

    Parameters
    ----------
    mode : {"independent", "hierarchical", "fixed"}
        Sampling mode.
    space : {"param", "z"}
        Sampling space. ``"param"`` draws in constrained parameter space,
        ``"z"`` draws in unconstrained space and maps via the model schema.
    individual : dict[str, DistSpec]
        Per-parameter distributions for independent sampling.
    population : dict[str, DistSpec]
        Per-parameter distributions for hierarchical population centers.
    individual_sd : dict[str, float]
        Per-parameter standard deviations used for hierarchical subject draws.
    fixed : dict[str, float]
        Fixed parameter values (used when ``mode="fixed"``).
    by_condition : dict[str, ConditionSamplingSpec]
        Optional per-condition overrides for sampling. These are interpreted
        in the *base model* parameter space and are only supported for
        within-subject shared+delta models.
    clip_to_bounds : bool
        Whether to clip sampled values to parameter bounds (if available).

    Notes
    -----
    - ``space="z"`` requires ``model.param_schema`` with ``params_from_z`` and
      consistent schema ordering.
    - ``by_condition`` is applied by condition label, and any missing values
      fall back to the top-level distributions for that parameter.

    Examples
    --------
    >>> SamplingSpec(mode="fixed", fixed={"alpha": 0.2, "beta": 3.0})
    SamplingSpec(mode='fixed', space='param', individual={}, population={}, individual_sd={}, fixed={'alpha': 0.2, 'beta': 3.0}, by_condition={}, clip_to_bounds=True)
    """
    mode: str = "independent"   # "independent" | "hierarchical" | "fixed"
    space: str = "param"        # "param" | "z"

    individual: dict[str, DistSpec] = field(default_factory=dict)

    population: dict[str, DistSpec] = field(default_factory=dict)
    individual_sd: dict[str, float] = field(default_factory=dict)

    fixed: dict[str, float] = field(default_factory=dict)

    by_condition: dict[str, ConditionSamplingSpec] = field(default_factory=dict)

    clip_to_bounds: bool = True


@dataclass(frozen=True, slots=True)
class OutputSpec:

    """
    Output settings for parameter recovery.

    Attributes
    ----------
    out_dir : str
        Directory to write outputs to.
    save_format : {"csv", "parquet"}
        Table serialization format.
    save_config : bool
        Whether to write the config alongside results.
    save_fit_diagnostics : bool
        Whether to write estimator diagnostics (if available).
    save_simulated_study : bool
        Whether to save pickled simulated StudyData (can be large).
    """

    out_dir: str = "recovery_out"
    save_format: str = "csv"         # "csv" | "parquet"
    save_config: bool = True
    save_fit_diagnostics: bool = True
    save_simulated_study: bool = False  # can be large; saves pickled StudyData per rep


@dataclass(frozen=True, slots=True)
class ParameterRecoveryConfig:
    """
    Parameter recovery configuration.

    Parameters
    ----------
    plan_path : str
        YAML/JSON study plan used for simulation.
    n_reps : int
        Number of replications.
    seed : int
        RNG seed.
    n_jobs : int
        Number of parallel worker processes for replications.
        Use ``1`` for sequential execution.
    sampling : SamplingSpec
        Sampling configuration.
    output : OutputSpec
        Output configuration.

    Examples
    --------
    >>> ParameterRecoveryConfig(plan_path="study_plan.yaml", n_reps=10, seed=123, n_jobs=1)
    ParameterRecoveryConfig(plan_path='study_plan.yaml', n_reps=10, seed=123, n_jobs=1, sampling=SamplingSpec(mode='independent', space='param', individual={}, population={}, individual_sd={}, fixed={}, by_condition={}, clip_to_bounds=True), output=OutputSpec(out_dir='recovery_out', save_format='csv', save_config=True, save_fit_diagnostics=True, save_simulated_study=False))
    """
    plan_path: str
    n_reps: int = 50
    seed: int = 0
    n_jobs: int = 1

    sampling: SamplingSpec = field(default_factory=SamplingSpec)
    output: OutputSpec = field(default_factory=OutputSpec)


def _parse_dists(d: Any) -> dict[str, DistSpec]:
    """Parse a mapping of distribution specs into DistSpec objects.

    Parameters
    ----------
    d : Any
        Raw mapping from parameter name to dicts of the form
        ``{"name": <dist>, "args": {...}}``.

    Returns
    -------
    dict[str, DistSpec]
        Parsed distribution specs.

    Raises
    ------
    ValueError
        If the input is not a mapping or if a spec is missing ``name``.
    """
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise ValueError("Expected mapping of param->dist spec.")
    out: dict[str, DistSpec] = {}
    for k, v in d.items():
        if not isinstance(v, dict) or "name" not in v:
            raise ValueError(f"DistSpec for {k} must be dict with at least 'name'.")
        out[str(k)] = DistSpec(name=str(v["name"]), args=dict(v.get("args", {})))
    return out


def _parse_condition_sampling(d: Any) -> dict[str, ConditionSamplingSpec]:
    """Parse per-condition sampling overrides.

    Parameters
    ----------
    d : Any
        Raw mapping from condition label to sampling overrides.

    Returns
    -------
    dict[str, ConditionSamplingSpec]
        Parsed per-condition sampling specs.

    Raises
    ------
    ValueError
        If the input is not a mapping or if a condition entry is invalid.
    """
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise ValueError("sampling.by_condition must be a mapping of condition -> spec.")
    out: dict[str, ConditionSamplingSpec] = {}
    for k, v in d.items():
        if v is None:
            out[str(k)] = ConditionSamplingSpec()
            continue
        if not isinstance(v, dict):
            raise ValueError(f"sampling.by_condition[{k}] must be a mapping.")
        out[str(k)] = ConditionSamplingSpec(
            individual=_parse_dists(v.get("individual")),
            population=_parse_dists(v.get("population")),
            individual_sd={str(p): float(val) for p, val in (v.get("individual_sd", {}) or {}).items()},
            fixed={str(p): float(val) for p, val in (v.get("fixed", {}) or {}).items()},
        )
    return out


def load_parameter_recovery_config(path: str | Path) -> ParameterRecoveryConfig:

    """
    Load a :class:`ParameterRecoveryConfig` from YAML or JSON.
    
    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``.yaml``/``.yml`` or ``.json`` config file.
    
    Returns
    -------
    ParameterRecoveryConfig
        Parsed configuration.

    Examples
    --------
    >>> # cfg = load_parameter_recovery_config("recovery_config.yaml")  # doctest: +SKIP
    """

    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError("PyYAML required to load YAML. Install: pip install pyyaml") from e
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raise ValueError("Config must be .yaml/.yml or .json")

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping/object.")

    samp_raw = raw.get("sampling", {}) or {}
    sampling = SamplingSpec(
        mode=str(samp_raw.get("mode", "independent")),
        space=str(samp_raw.get("space", "param")),
        individual=_parse_dists(samp_raw.get("individual")),
        population=_parse_dists(samp_raw.get("population")),
        individual_sd={str(k): float(v) for k, v in (samp_raw.get("individual_sd", {}) or {}).items()},
        fixed={str(k): float(v) for k, v in (samp_raw.get("fixed", {}) or {}).items()},
        by_condition=_parse_condition_sampling(samp_raw.get("by_condition")),
        clip_to_bounds=bool(samp_raw.get("clip_to_bounds", True)),
    )

    out_raw = raw.get("output", {}) or {}
    output = OutputSpec(
        out_dir=str(out_raw.get("out_dir", "recovery_out")),
        save_format=str(out_raw.get("save_format", "csv")),
        save_config=bool(out_raw.get("save_config", True)),
        save_fit_diagnostics=bool(out_raw.get("save_fit_diagnostics", True)),
        save_simulated_study=bool(out_raw.get("save_simulated_study", False)),
    )

    if "plan_path" not in raw:
        raise ValueError("Missing required field: plan_path")

    return ParameterRecoveryConfig(
        plan_path=str(raw["plan_path"]),
        n_reps=int(raw.get("n_reps", 50)),
        seed=int(raw.get("seed", 0)),
        n_jobs=max(1, int(raw.get("n_jobs", 1))),
        sampling=sampling,
        output=output,
    )


def config_to_json(cfg: ParameterRecoveryConfig) -> str:

    """
    Serialize a :class:`ParameterRecoveryConfig` to JSON.
    
    Parameters
    ----------
    cfg : ParameterRecoveryConfig
        Configuration to serialize.
    
    Returns
    -------
    str
        Pretty-printed JSON string.

    Examples
    --------
    >>> cfg = ParameterRecoveryConfig(plan_path="study_plan.yaml")
    >>> s = config_to_json(cfg)
    >>> '"plan_path"' in s
    True
    """

    return json.dumps(asdict(cfg), indent=2)
