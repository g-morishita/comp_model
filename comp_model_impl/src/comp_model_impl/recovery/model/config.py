"""Configuration for model recovery experiments.

Model recovery simulates data from each generating model, fits all candidate
models to each simulated dataset, and selects a winner under a criterion
such as log-likelihood, AIC, BIC, or WAIC.

The sampling schema intentionally reuses parameter-recovery specs so the same
parameter generation definitions can be shared across both workflows.
"""

from __future__ import annotations

import json

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from comp_model_impl.recovery.parameter.config import (
    ConditionSamplingSpec,
    DistSpec,
    SamplingSpec,
)


def _require_non_empty_text(value: Any, *, field_name: str) -> str:
    """Validate and normalize a required non-empty string field."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value)}")
    out = value.strip()
    if out == "":
        raise ValueError(f"{field_name} must be a non-empty string")
    return out


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """Registry component reference and constructor kwargs."""

    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_text(self.name, field_name="components.*.name")
        if not isinstance(self.kwargs, Mapping):
            raise TypeError(f"components.*.kwargs must be a mapping, got {type(self.kwargs)}")


@dataclass(frozen=True, slots=True)
class ModelRecoveryComponents:
    """Registry-backed components used by model recovery."""

    generator: ComponentSpec

    def __post_init__(self) -> None:
        if not isinstance(self.generator, ComponentSpec):
            raise TypeError(
                "components.generator must be a ComponentSpec "
                f"(got {type(self.generator)})"
            )


@dataclass(frozen=True, slots=True)
class GeneratingModelSpec:
    """Specification for one generating model.

    Parameters
    ----------
    name : str
        Human-readable label used in output tables.
    model : str
        Model registry key resolved from
        ``comp_model_impl.register.make_registry().models``.
    model_kwargs : dict[str, Any], default={}
        Keyword arguments passed to the model constructor/factory.
    sampling : SamplingSpec, default=SamplingSpec()
        Parameter sampling configuration used when simulating data.
    """

    name: str
    model: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    sampling: SamplingSpec = field(default_factory=SamplingSpec)

    def __post_init__(self) -> None:
        _require_non_empty_text(self.name, field_name="generating[].name")
        _require_non_empty_text(self.model, field_name="generating[].model")


@dataclass(frozen=True, slots=True)
class CandidateModelSpec:
    """Specification for one candidate model.

    Parameters
    ----------
    name : str
        Human-readable label used in output tables.
    model : str
        Candidate model registry key resolved from
        ``comp_model_impl.register.make_registry().models``.
    estimator : str
        Estimator registry key resolved from
        ``comp_model_impl.register.make_registry().estimators``.
    model_kwargs : dict[str, Any], default={}
        Keyword arguments passed to the model constructor/factory.
    estimator_kwargs : dict[str, Any], default={}
        Keyword arguments passed to the estimator constructor/factory.
    fixed_params : dict[str, float], default={}
        Optional fixed parameter values forwarded to ``estimator.fit`` when that
        method accepts a ``fixed_params`` argument.
    """

    name: str
    model: str
    estimator: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    estimator_kwargs: dict[str, Any] = field(default_factory=dict)
    fixed_params: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_text(self.name, field_name="candidates[].name")
        _require_non_empty_text(self.model, field_name="candidates[].model")
        _require_non_empty_text(self.estimator, field_name="candidates[].estimator")


@dataclass(frozen=True, slots=True)
class SelectionSpec:
    """Model selection settings.

    Parameters
    ----------
    criterion : {"loglike", "aic", "bic", "waic"}, default="bic"
        Scoring criterion used to compare candidates.
    tie_break : {"first", "simpler"}, default="simpler"
        Tie-breaking strategy when scores are equal within ``atol``.
    atol : float, default=1e-9
        Absolute tolerance for tie handling.
    """

    criterion: str = "bic"
    tie_break: str = "simpler"
    atol: float = 1e-9


@dataclass(frozen=True, slots=True)
class OutputSpec:
    """Output options for model recovery.

    Parameters
    ----------
    out_dir : str, default="model_recovery_out"
        Base output directory; each run creates a unique subdirectory.
    save_format : {"csv", "parquet"}, default="csv"
        Serialization format for the fit table.
    save_config : bool, default=True
        Whether to write ``config.json`` and ``model_recovery_manifest.json``.
    save_fit_diagnostics : bool, default=True
        Whether to write per-fit diagnostics as JSONL.
    save_simulated_study : bool, default=False
        Whether to persist each simulated study artifact.
    """

    out_dir: str = "model_recovery_out"
    save_format: str = "csv"
    save_config: bool = True
    save_fit_diagnostics: bool = True
    save_simulated_study: bool = False


@dataclass(frozen=True, slots=True)
class ModelRecoveryConfig:
    """Top-level configuration for model recovery.

    Parameters
    ----------
    plan_path : str
        Path to the study plan YAML/JSON used for simulation and fitting.
    n_reps : int, default=50
        Number of replications per generating model.
    seed : int, default=0
        Base random seed.
    components : ModelRecoveryComponents or None, default=None
        Optional registry component references for instantiating the
        simulation generator from config.
    generating : list[GeneratingModelSpec], default=[]
        Generating model specifications.
    candidates : list[CandidateModelSpec], default=[]
        Candidate model specifications.
    selection : SelectionSpec, default=SelectionSpec()
        Model selection settings.
    output : OutputSpec, default=OutputSpec()
        Output settings.
    """

    plan_path: str
    n_reps: int = 50
    seed: int = 0

    components: ModelRecoveryComponents | None = None

    generating: list[GeneratingModelSpec] = field(default_factory=list)
    candidates: list[CandidateModelSpec] = field(default_factory=list)

    selection: SelectionSpec = field(default_factory=SelectionSpec)
    output: OutputSpec = field(default_factory=OutputSpec)

    def __post_init__(self) -> None:
        _require_non_empty_text(self.plan_path, field_name="plan_path")
        if self.components is not None and not isinstance(self.components, ModelRecoveryComponents):
            raise TypeError(
                "components must be a ModelRecoveryComponents or None "
                f"(got {type(self.components)})"
            )


def _coerce_mapping(raw: Any, *, field_name: str) -> dict[str, Any]:
    """Coerce an optional mapping to ``dict[str, Any]``.

    Parameters
    ----------
    raw : Any
        Raw value from configuration.
    field_name : str
        Field label used in error messages.

    Returns
    -------
    dict[str, Any]
        Parsed dictionary with string keys.

    Raises
    ------
    TypeError
        If ``raw`` is not ``None`` and not a mapping.
    """

    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"{field_name} must be a mapping, got {type(raw)}")
    return {str(k): v for k, v in raw.items()}


def _coerce_float_mapping(raw: Any, *, field_name: str) -> dict[str, float]:
    """Coerce an optional mapping to ``dict[str, float]``.

    Parameters
    ----------
    raw : Any
        Raw value from configuration.
    field_name : str
        Field label used in error messages.

    Returns
    -------
    dict[str, float]
        Parsed dictionary with float values.
    """

    out = _coerce_mapping(raw, field_name=field_name)
    return {k: float(v) for k, v in out.items()}


def _coerce_component_ref(raw: Any, *, field_name: str) -> str:
    """Parse a required component reference string.

    Parameters
    ----------
    raw : Any
        Component reference value.
    field_name : str
        Field label used in error messages.

    Returns
    -------
    str
        Parsed component reference.

    Raises
    ------
    ValueError
        If ``raw`` is missing.
    TypeError
        If ``raw`` is not a string.
    """

    if raw is None:
        raise ValueError(f"Missing required field: {field_name}")
    return _require_non_empty_text(raw, field_name=field_name)


def _coerce_component_spec(raw: Any, *, field_name: str) -> ComponentSpec:
    """Parse a component object containing ``name`` and optional ``kwargs``."""

    if not isinstance(raw, Mapping):
        raise TypeError(f"{field_name} must be a mapping, got {type(raw)}")

    name = _coerce_component_ref(raw.get("name"), field_name=f"{field_name}.name")
    kwargs = _coerce_mapping(raw.get("kwargs"), field_name=f"{field_name}.kwargs")
    return ComponentSpec(name=name, kwargs=kwargs)


def _coerce_components(raw: Any) -> ModelRecoveryComponents:
    """Parse the optional ``components`` section."""

    if not isinstance(raw, Mapping):
        raise TypeError(f"components must be a mapping, got {type(raw)}")
    if "generator" not in raw:
        raise ValueError("components missing required 'generator'")

    return ModelRecoveryComponents(
        generator=_coerce_component_spec(
            raw.get("generator"),
            field_name="components.generator",
        )
    )


def _coerce_sampling_spec(raw: Any) -> SamplingSpec:
    """Parse a sampling configuration object.

    Parameters
    ----------
    raw : Any
        ``SamplingSpec`` instance or raw mapping.

    Returns
    -------
    SamplingSpec
        Parsed sampling spec.
    """

    if isinstance(raw, SamplingSpec):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError(f"sampling must be a mapping, got {type(raw)}")

    mode = str(raw.get("mode", "independent"))
    space = str(raw.get("space", "param"))

    def _parse_dist_map(obj: Any) -> dict[str, DistSpec]:
        if obj is None:
            return {}
        if not isinstance(obj, Mapping):
            raise TypeError("distribution map must be a mapping")
        out: dict[str, DistSpec] = {}
        for k, v in obj.items():
            if isinstance(v, DistSpec):
                out[str(k)] = v
            elif isinstance(v, Mapping):
                out[str(k)] = DistSpec(
                    name=str(v.get("name")),
                    args=_coerce_mapping(v.get("args"), field_name=f"distribution[{k}].args"),
                )
            else:
                raise TypeError(f"Invalid DistSpec for {k}: {type(v)}")
        return out

    by_cond_raw = raw.get("by_condition", {}) or {}
    by_condition: dict[str, ConditionSamplingSpec] = {}
    if by_cond_raw:
        if not isinstance(by_cond_raw, Mapping):
            raise TypeError("by_condition must be a mapping")
        for cond, cs in by_cond_raw.items():
            if isinstance(cs, ConditionSamplingSpec):
                by_condition[str(cond)] = cs
                continue
            if not isinstance(cs, Mapping):
                raise TypeError("ConditionSamplingSpec must be a mapping")
            by_condition[str(cond)] = ConditionSamplingSpec(
                individual=_parse_dist_map(cs.get("individual")),
                population=_parse_dist_map(cs.get("population")),
                individual_sd=_coerce_float_mapping(cs.get("individual_sd"), field_name=f"by_condition[{cond}].individual_sd"),
                fixed=_coerce_float_mapping(cs.get("fixed"), field_name=f"by_condition[{cond}].fixed"),
            )

    return SamplingSpec(
        mode=mode,
        space=space,
        individual=_parse_dist_map(raw.get("individual")),
        population=_parse_dist_map(raw.get("population")),
        individual_sd=_coerce_float_mapping(raw.get("individual_sd"), field_name="sampling.individual_sd"),
        fixed=_coerce_float_mapping(raw.get("fixed"), field_name="sampling.fixed"),
        by_condition=by_condition,
    )


def config_from_raw_dict(raw: Mapping[str, Any]) -> ModelRecoveryConfig:
    """Parse a raw mapping into :class:`ModelRecoveryConfig`.

    Parameters
    ----------
    raw : Mapping[str, Any]
        Parsed configuration dictionary from YAML/JSON.

    Returns
    -------
    ModelRecoveryConfig
        Validated configuration object.
    """

    if "plan_path" not in raw:
        raise ValueError("Missing required field: plan_path")

    components = None
    if raw.get("components") is not None:
        components = _coerce_components(raw.get("components"))

    generating_raw = raw.get("generating", []) or []
    candidates_raw = raw.get("candidates", []) or []

    generating: list[GeneratingModelSpec] = []
    for i, g in enumerate(generating_raw):
        if not isinstance(g, Mapping):
            raise TypeError(f"generating[{i}] must be a mapping")
        if "model" not in g:
            raise ValueError(f"generating[{i}] missing required 'model'")
        model_ref = _coerce_component_ref(
            g["model"],
            field_name=f"generating[{i}].model",
        )
        model_kwargs = _coerce_mapping(
            g.get("model_kwargs"),
            field_name=f"generating[{i}].model_kwargs",
        )
        name = str(g.get("name") or model_ref or f"gen_{i}")
        sampling = _coerce_sampling_spec(g.get("sampling", {}))
        generating.append(
            GeneratingModelSpec(
                name=name,
                model=model_ref,
                model_kwargs=model_kwargs,
                sampling=sampling,
            )
        )

    candidates: list[CandidateModelSpec] = []
    for i, c in enumerate(candidates_raw):
        if not isinstance(c, Mapping):
            raise TypeError(f"candidates[{i}] must be a mapping")
        if "name" not in c:
            raise ValueError(f"candidates[{i}] missing required 'name'")
        if "model" not in c:
            raise ValueError(f"candidates[{i}] missing required 'model'")
        if "estimator" not in c:
            raise ValueError(f"candidates[{i}] missing required 'estimator'")

        model_ref = _coerce_component_ref(
            c["model"],
            field_name=f"candidates[{i}].model",
        )
        model_kwargs = _coerce_mapping(c.get("model_kwargs"), field_name=f"candidates[{i}].model_kwargs")

        estimator_ref = _coerce_component_ref(
            c["estimator"],
            field_name=f"candidates[{i}].estimator",
        )
        estimator_kwargs = _coerce_mapping(c.get("estimator_kwargs"), field_name=f"candidates[{i}].estimator_kwargs")

        candidates.append(
            CandidateModelSpec(
                name=str(c["name"]),
                model=model_ref,
                model_kwargs=model_kwargs,
                estimator=estimator_ref,
                estimator_kwargs=estimator_kwargs,
                fixed_params=_coerce_float_mapping(c.get("fixed_params"), field_name=f"candidates[{i}].fixed_params"),
            )
        )

    sel_raw = _coerce_mapping(raw.get("selection"), field_name="selection")
    selection = SelectionSpec(
        criterion=str(sel_raw.get("criterion", "bic")),
        tie_break=str(sel_raw.get("tie_break", "simpler")),
        atol=float(sel_raw.get("atol", 1e-9)),
    )

    out_raw = _coerce_mapping(raw.get("output"), field_name="output")
    output = OutputSpec(
        out_dir=str(out_raw.get("out_dir", "model_recovery_out")),
        save_format=str(out_raw.get("save_format", "csv")),
        save_config=bool(out_raw.get("save_config", True)),
        save_fit_diagnostics=bool(out_raw.get("save_fit_diagnostics", True)),
        save_simulated_study=bool(out_raw.get("save_simulated_study", False)),
    )

    return ModelRecoveryConfig(
        plan_path=str(raw["plan_path"]),
        n_reps=int(raw.get("n_reps", 50)),
        seed=int(raw.get("seed", 0)),
        components=components,
        generating=generating,
        candidates=candidates,
        selection=selection,
        output=output,
    )


def load_model_recovery_config(path: str | Path) -> ModelRecoveryConfig:
    """Load model recovery config from JSON/YAML.

    Parameters
    ----------
    path : str or pathlib.Path
        File path ending with ``.json``, ``.yaml``, or ``.yml``.

    Returns
    -------
    ModelRecoveryConfig
        Parsed config object.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml  # lazy import

        raw = yaml.safe_load(p.read_text())
    elif suffix == ".json":
        raw = json.loads(p.read_text())
    else:
        raise ValueError(f"Unsupported config extension: {p.suffix}")

    if not isinstance(raw, Mapping):
        raise TypeError("Config root must be a mapping")
    return config_from_raw_dict(raw)


def config_to_json(cfg: ModelRecoveryConfig) -> str:
    """Serialize :class:`ModelRecoveryConfig` to pretty JSON.

    Parameters
    ----------
    cfg : ModelRecoveryConfig
        Configuration object.

    Returns
    -------
    str
        JSON string representation.
    """

    return json.dumps(asdict(cfg), indent=2)
