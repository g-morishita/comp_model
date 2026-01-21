from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import json


@dataclass(frozen=True, slots=True)
class DistSpec:
    """
    Distribution specification (scipy.stats).

    Example:
      DistSpec(name="beta", args={"a": 2.0, "b": 2.0})
      DistSpec(name="lognorm", args={"s": 0.4, "scale": 4.0})
    """
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SamplingSpec:
    """
    How to sample true parameters.

    mode:
      - "independent": each subject sampled from `individual` distributions
      - "hierarchical": sample population center from `population`, then individuals around it
      - "fixed": all subjects use `fixed`

    space:
      - "param": distributions are on parameter values directly
      - "z": distributions are on unconstrained z; then mapped via model.param_schema transforms
             (requires schema.params_from_z and schema.params ordering)
    """
    mode: str = "independent"   # "independent" | "hierarchical" | "fixed"
    space: str = "param"        # "param" | "z"

    individual: dict[str, DistSpec] = field(default_factory=dict)

    population: dict[str, DistSpec] = field(default_factory=dict)
    individual_sd: dict[str, float] = field(default_factory=dict)

    fixed: dict[str, float] = field(default_factory=dict)

    clip_to_bounds: bool = True


@dataclass(frozen=True, slots=True)
class OutputSpec:
    out_dir: str = "recovery_out"
    save_format: str = "csv"         # "csv" | "parquet"
    save_config: bool = True
    save_fit_diagnostics: bool = True
    save_simulated_study: bool = False  # can be large; saves pickled StudyData per rep


@dataclass(frozen=True, slots=True)
class PlotSpec:
    make_plots: bool = True
    scatter_alpha: float = 0.6
    max_points: int = 50_000


@dataclass(frozen=True, slots=True)
class ParameterRecoveryConfig:
    """
    Parameter recovery configuration.

    plan_path: YAML/JSON study plan used for simulation
    n_reps: number of replications
    seed: RNG seed
    """
    plan_path: str
    n_reps: int = 50
    seed: int = 0

    sampling: SamplingSpec = field(default_factory=SamplingSpec)
    output: OutputSpec = field(default_factory=OutputSpec)
    plots: PlotSpec = field(default_factory=PlotSpec)


def _parse_dists(d: Any) -> dict[str, DistSpec]:
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


def load_parameter_recovery_config(path: str | Path) -> ParameterRecoveryConfig:
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

    plot_raw = raw.get("plots", {}) or {}
    plots = PlotSpec(
        make_plots=bool(plot_raw.get("make_plots", True)),
        scatter_alpha=float(plot_raw.get("scatter_alpha", 0.6)),
        max_points=int(plot_raw.get("max_points", 50_000)),
    )

    if "plan_path" not in raw:
        raise ValueError("Missing required field: plan_path")

    return ParameterRecoveryConfig(
        plan_path=str(raw["plan_path"]),
        n_reps=int(raw.get("n_reps", 50)),
        seed=int(raw.get("seed", 0)),
        sampling=sampling,
        output=output,
        plots=plots,
    )


def config_to_json(cfg: ParameterRecoveryConfig) -> str:
    return json.dumps(asdict(cfg), indent=2)
