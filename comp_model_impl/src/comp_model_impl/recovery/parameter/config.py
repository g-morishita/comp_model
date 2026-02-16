"""Configuration for parameter recovery experiments.

This module defines dataclasses used to configure parameter recovery runs and
utilities for loading them from YAML/JSON.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import json
import importlib.util


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
class ComponentSpec:
    """Registry component reference and constructor kwargs.

    Parameters
    ----------
    name : str
        Registry key of the component.
    kwargs : dict[str, Any], optional
        Keyword arguments passed to the component constructor.
    """

    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParameterRecoveryComponents:
    """Component references used by registry-based parameter recovery.

    Parameters
    ----------
    generator : ComponentSpec
        Generator component reference.
    generating_model : ComponentSpec
        Model used to simulate data.
    fitting_model : ComponentSpec
        Model used by the estimator when fitting.
    estimator : ComponentSpec
        Estimator component reference.
    """

    generator: ComponentSpec
    generating_model: ComponentSpec
    fitting_model: ComponentSpec
    estimator: ComponentSpec


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
    components : ParameterRecoveryComponents
        Registry component references for generator/model/estimator.
    sampling : SamplingSpec
        Sampling configuration.
    output : OutputSpec
        Output configuration.
    """
    plan_path: str
    n_reps: int
    seed: int
    n_jobs: int
    components: ParameterRecoveryComponents

    sampling: SamplingSpec = field(default_factory=SamplingSpec)
    output: OutputSpec = field(default_factory=OutputSpec)


def _parse_dists(d: Any) -> dict[str, DistSpec]:
    """Parse a mapping of distribution specs into DistSpec objects.

    Parameters
    ----------
    d : Any
        Expected to be ``None`` or a mapping with the shape::

            {
              "<param_name>": {"name": "<scipy_dist_name>", "args": {...}},
              ...
            }

        where ``args`` is optional (defaults to ``{}``).

        Example::

            {
              "alpha": {"name": "beta", "args": {"a": 2.0, "b": 2.0}},
              "beta": {"name": "lognorm", "args": {"s": 0.4, "scale": 4.0}},
            }

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


def _require(raw: Mapping[str, Any], key: str) -> Any:
    if key not in raw:
        raise ValueError(f"Missing required field: {key}")
    return raw[key]


def _require_int(raw: Mapping[str, Any], key: str, *, min_value: int | None = None) -> int:
    """Require an integer field and optionally validate a minimum value."""
    value = _require(raw, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer.")
    out = int(value)
    if min_value is not None and out < min_value:
        raise ValueError(f"{key} must be >= {min_value}.")
    return out


def _as_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    """Require a mapping/object value for a named field."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping/object.")
    return value


def _require_non_empty_string(raw: Mapping[str, Any], key: str, *, field_name: str) -> str:
    """Require a non-empty string field from a mapping."""
    value = _require(raw, key)
    if not isinstance(value, str):
        raise ValueError(f"{field_name}.{key} must be a non-empty string.")
    out = value.strip()
    if out == "":
        raise ValueError(f"{field_name}.{key} must be a non-empty string.")
    return out


def _parse_component_spec(raw: Any, *, field_name: str) -> ComponentSpec:
    """Parse one component spec from a mapping."""
    comp_raw = _as_mapping(raw, field_name=field_name)
    name = _require_non_empty_string(comp_raw, "name", field_name=field_name)
    kwargs_raw = comp_raw.get("kwargs", {})
    kwargs_mapping = _as_mapping(kwargs_raw, field_name=f"{field_name}.kwargs")
    kwargs = {str(k): v for k, v in kwargs_mapping.items()}
    return ComponentSpec(name=name, kwargs=kwargs)


def _parse_components_spec(raw: Any) -> ParameterRecoveryComponents:
    """Parse the required ``components`` section."""
    comp_root = _as_mapping(raw, field_name="components")
    return ParameterRecoveryComponents(
        generator=_parse_component_spec(_require(comp_root, "generator"), field_name="components.generator"),
        generating_model=_parse_component_spec(
            _require(comp_root, "generating_model"),
            field_name="components.generating_model",
        ),
        fitting_model=_parse_component_spec(
            _require(comp_root, "fitting_model"),
            field_name="components.fitting_model",
        ),
        estimator=_parse_component_spec(_require(comp_root, "estimator"), field_name="components.estimator"),
    )


def _validate_components_registered(components: ParameterRecoveryComponents) -> None:
    """Validate that configured component names exist in the default registry."""
    from comp_model_impl.register import make_registry

    r = make_registry()

    if components.generator.name not in set(r.generators.names()):
        available = ", ".join(r.generators.names()) or "<none>"
        raise ValueError(
            f"Unknown components.generator.name: {components.generator.name!r}. "
            f"Available generators: {available}"
        )

    if components.generating_model.name not in set(r.models.names()):
        available = ", ".join(r.models.names()) or "<none>"
        raise ValueError(
            f"Unknown components.generating_model.name: {components.generating_model.name!r}. "
            f"Available models: {available}"
        )

    if components.fitting_model.name not in set(r.models.names()):
        available = ", ".join(r.models.names()) or "<none>"
        raise ValueError(
            f"Unknown components.fitting_model.name: {components.fitting_model.name!r}. "
            f"Available models: {available}"
        )

    if components.estimator.name not in set(r.estimators.names()):
        available = ", ".join(r.estimators.names()) or "<none>"
        raise ValueError(
            f"Unknown components.estimator.name: {components.estimator.name!r}. "
            f"Available estimators: {available}"
        )


def _parse_sampling_spec_strict(samp_raw: Mapping[str, Any]) -> SamplingSpec:
    """
    Parse and validate the ``sampling`` section with no implicit defaults.

    Required keys for every sampling config are:
    ``mode``, ``space``, ``clip_to_bounds``, and ``by_condition``.

    Mode-specific required keys are:
    - ``fixed``: ``fixed``
    - ``independent``: ``individual``
    - ``hierarchical``: ``population`` and ``individual_sd``

    Parameters
    ----------
    samp_raw : Mapping[str, Any]
        Raw ``sampling`` object loaded from YAML/JSON.

    Returns
    -------
    SamplingSpec
        Parsed sampling specification.

    Raises
    ------
    ValueError
        If required fields are missing, malformed, or contain unknown enum
        values for mode/space.
    """
    if not isinstance(samp_raw, Mapping):
        raise ValueError("sampling must be a mapping/object.")

    mode = str(_require(samp_raw, "mode")).lower()
    space = str(_require(samp_raw, "space")).lower()

    if mode not in {"fixed", "independent", "hierarchical"}:
        raise ValueError(f"Unknown sampling.mode: {mode!r}")
    if space not in {"param", "z"}:
        raise ValueError(f"Unknown sampling.space: {space!r}")

    individual: dict[str, DistSpec] = {}
    population: dict[str, DistSpec] = {}
    individual_sd: dict[str, float] = {}
    fixed: dict[str, float] = {}

    if mode == "fixed":
        fixed_raw = _as_mapping(_require(samp_raw, "fixed"), field_name="sampling.fixed")
        fixed = {str(k): float(v) for k, v in fixed_raw.items()}
    elif mode == "independent":
        individual_raw = _as_mapping(_require(samp_raw, "individual"), field_name="sampling.individual")
        individual = _parse_dists(dict(individual_raw))
    else:  # hierarchical
        population_raw = _as_mapping(_require(samp_raw, "population"), field_name="sampling.population")
        individual_sd_raw = _as_mapping(
            _require(samp_raw, "individual_sd"),
            field_name="sampling.individual_sd",
        )
        population = _parse_dists(dict(population_raw))
        individual_sd = {str(k): float(v) for k, v in individual_sd_raw.items()}

    by_condition_raw = _require(samp_raw, "by_condition")
    clip_to_bounds_raw = _require(samp_raw, "clip_to_bounds")

    return SamplingSpec(
        mode=mode,
        space=space,
        individual=individual,
        population=population,
        individual_sd=individual_sd,
        fixed=fixed,
        by_condition=_parse_condition_sampling(by_condition_raw),
        clip_to_bounds=bool(clip_to_bounds_raw),
    )


def load_parameter_recovery_config(path: str | Path) -> ParameterRecoveryConfig:

    """
    Load a :class:`ParameterRecoveryConfig` from YAML or JSON.

    This loader follows a strict/no-default policy for core run controls and
    sampling settings. The following top-level fields are required:

    - ``plan_path``
    - ``n_reps`` (integer, ``>= 1``)
    - ``seed`` (integer)
    - ``n_jobs`` (integer, ``>= 1``)
    - ``components``
    - ``sampling``

    The ``components`` section must include:

    - ``generator`` (with ``name`` and optional ``kwargs``)
    - ``generating_model`` (with ``name`` and optional ``kwargs``)
    - ``fitting_model`` (with ``name`` and optional ``kwargs``)
    - ``estimator`` (with ``name`` and optional ``kwargs``)

    The ``sampling`` section is validated by
    :func:`_parse_sampling_spec_strict` and must include common keys
    (``mode``, ``space``, ``by_condition``, ``clip_to_bounds``) plus
    mode-specific keys:

    - ``mode="fixed"`` requires ``fixed``
    - ``mode="independent"`` requires ``individual``
    - ``mode="hierarchical"`` requires ``population`` and ``individual_sd``

    The ``output`` section is optional; when omitted, ``OutputSpec`` defaults
    are applied for output-related fields.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``.yaml``/``.yml`` or ``.json`` config file.

    Returns
    -------
    ParameterRecoveryConfig
        Parsed configuration.

    Raises
    ------
    ImportError
        If loading YAML but PyYAML is not installed.
    OSError
        If the file cannot be opened/read.
    json.JSONDecodeError
        If the JSON file is malformed.
    ValueError
        If the file extension is unsupported, the root object is not a
        mapping, required fields are missing, or typed/range checks fail.
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

    components = _parse_components_spec(_require(raw, "components"))
    _validate_components_registered(components)
    sampling = _parse_sampling_spec_strict(_require(raw, "sampling"))

    out_raw = raw.get("output", {}) or {}
    output = OutputSpec(
        out_dir=str(out_raw.get("out_dir", "recovery_out")),
        save_format=str(out_raw.get("save_format", "csv")),
        save_config=bool(out_raw.get("save_config", True)),
        save_fit_diagnostics=bool(out_raw.get("save_fit_diagnostics", True)),
        save_simulated_study=bool(out_raw.get("save_simulated_study", False)),
    )
    
    plan_path = _require(raw, "plan_path")

    n_reps = _require_int(raw, "n_reps", min_value=1)
    seed = _require_int(raw, "seed")
    n_jobs = _require_int(raw, "n_jobs", min_value=1)

    return ParameterRecoveryConfig(
        plan_path=str(plan_path),
        n_reps=n_reps,
        seed=seed,
        n_jobs=n_jobs,
        components=components,
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
    """

    return json.dumps(asdict(cfg), indent=2)


def config_to_yaml(cfg: ParameterRecoveryConfig) -> str:
    """
    Serialize a :class:`ParameterRecoveryConfig` to YAML.

    Parameters
    ----------
    cfg : ParameterRecoveryConfig
        Configuration to serialize.

    Returns
    -------
    str
        YAML string.
    """
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml") from e

    return yaml.safe_dump(asdict(cfg), sort_keys=False)


def save_config_auto(cfg: ParameterRecoveryConfig, out_dir: Path, stem: str) -> Path:
    """
    Save a parameter-recovery config using the best available text format.

    The function prefers YAML for readability when PyYAML is installed.
    If PyYAML is unavailable, it falls back to JSON.

    Parameters
    ----------
    cfg : ParameterRecoveryConfig
        Configuration object to serialize.
    out_dir : Path
        Directory where the config file will be written.
    stem : str, default="parameter_recovery_config"
        Output filename stem (extension is chosen automatically).

    Returns
    -------
    Path
        Path to the written config file (``.yaml`` or ``.json``).

    Raises
    ------
    OSError
        If writing the output file fails.

    Notes
    -----
    Uses :func:`config_to_yaml` when PyYAML is importable, otherwise
    :func:`config_to_json`.
    """
    if importlib.util.find_spec("yaml") is not None:
        text = config_to_yaml(cfg)
        path = out_dir / f"{stem}.yaml"
    else:
        text = config_to_json(cfg)
        path = out_dir / f"{stem}.json"

    path.write_text(text, encoding="utf-8")
    return path
