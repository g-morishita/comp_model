"""Stan-based NUTS estimators.

This module provides two concrete estimators built on CmdStanPy:

* :class:`StanNUTSSubjectwiseEstimator` - fits each subject independently.
* :class:`StanHierarchicalNUTSEstimator` - fits a hierarchical model across subjects.

Both estimators rely on an event-log representation stored on blocks and on Stan
program templates distributed with this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from comp_model_core.data.types import StudyData
from comp_model_core.errors import CompatibilityError
from comp_model_core.interfaces.estimator import Estimator, FitResult
from comp_model_core.interfaces.model import ComputationalModel

from .adapters.registry import resolve_stan_adapter
from .cmdstan_utils import compile_cmdstan, load_stan_code
from .exporter import subject_to_stan_data, study_to_stan_data
from .priors import priors_to_stan_data


def _safe_summary_metric(summary: Any, candidates: list[str], *, agg: str) -> float | None:
    """Extract a scalar diagnostic from a CmdStanPy summary dataframe."""
    if summary is None:
        return None
    cols = {str(c).lower(): c for c in getattr(summary, "columns", [])}
    col = None
    for cand in candidates:
        if cand.lower() in cols:
            col = cols[cand.lower()]
            break
    if col is None:
        return None

    series = summary[col]
    if agg == "max":
        try:
            return float(series.max())
        except Exception:  # noqa: BLE001
            return None
    if agg == "min":
        try:
            return float(series.min())
        except Exception:  # noqa: BLE001
            return None
    raise ValueError(f"Unknown agg: {agg!r}")


@dataclass(slots=True)
class StanEstimator(Estimator):
    ...


@dataclass(slots=True)
class StanNUTSSubjectwiseEstimator(StanEstimator):
    """Independent (per-subject) Bayesian inference with NUTS."""

    model: ComputationalModel
    priors: Mapping[str, Any] | str

    chains: int = 4
    iter_warmup: int = 500
    iter_sampling: int = 1000
    adapt_delta: float = 0.9
    max_treedepth: int = 12

    forbid_extra_priors: bool = True
    show_progress: bool = False

    _compiled: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        adapter = resolve_stan_adapter(self.model)

        if isinstance(self.priors, str):
            try:
                import yaml  # type: ignore
            except ImportError as e:
                raise ImportError("PyYAML is required to load YAML plans. Install with: pip install pyyaml") from e
            with open(self.priors, "r", encoding="utf-8") as f:
                self.priors = yaml.safe_load(f)
                
        self.priors = priors_to_stan_data(
            priors_cfg=self.priors,
            adapter=adapter,
            family="indiv",
            forbid_extra=self.forbid_extra_priors,
        )

    def supports(self, study: StudyData) -> bool:
        for subj in study.subjects:
            for blk in subj.blocks:
                if blk.env_spec is None or not self.model.supports(blk.env_spec):
                    return False
                if blk.event_log is None:
                    return False
        return True

    def fit(self, *, study: StudyData, rng: np.random.Generator) -> FitResult:
        if not self.supports(study):
            raise CompatibilityError("Data/model incompatible or event logs missing.")

        adapter = resolve_stan_adapter(self.model)
        prog = adapter.program("indiv")

        stan_code = load_stan_code(prog.family, prog.key)
        if self._compiled is None:
            self._compiled = compile_cmdstan(stan_code, prog.program_name)

        required = adapter.required_priors("indiv")

        subj_hats: dict[str, dict[str, float]] = {}
        per_subject_diags: dict[str, Any] = {}

        for subj in study.subjects:
            data = subject_to_stan_data(subj)
            adapter.augment_subject_data(data)
            data |= self.priors

            seed = int(rng.integers(1, 2**31 - 1))
            fit = self._compiled.sample(
                data=data,
                chains=self.chains,
                iter_warmup=self.iter_warmup,
                iter_sampling=self.iter_sampling,
                seed=seed,
                adapt_delta=self.adapt_delta,
                max_treedepth=self.max_treedepth,
                show_progress=self.show_progress,
            )

            hats: dict[str, float] = {}
            for p in adapter.subject_param_names():
                draws = fit.stan_variable(p)
                hats[p] = float(np.mean(draws))
            subj_hats[subj.subject_id] = hats

            summ = None
            try:
                summ = fit.summary()
            except Exception:  # noqa: BLE001
                summ = None

            per_subject_diags[subj.subject_id] = {
                "seed": seed,
                "required_priors": list(required),
                "rhat_max": _safe_summary_metric(summ, ["R_hat", "r_hat"], agg="max"),
                "ess_bulk_min": _safe_summary_metric(summ, ["ESS_bulk", "ess_bulk"], agg="min"),
            }

        return FitResult(
            subject_hats=subj_hats,
            success=True,
            message="OK",
            diagnostics={"per_subject": per_subject_diags},
        )


@dataclass(slots=True)
class StanHierarchicalNUTSEstimator(StanEstimator):
    """Hierarchical Bayesian inference with NUTS over multiple subjects."""

    model: ComputationalModel
    hyper_priors: Mapping[str, Any]

    chains: int = 4
    iter_warmup: int = 800
    iter_sampling: int = 1200
    adapt_delta: float = 0.92
    max_treedepth: int = 12

    forbid_extra_priors: bool = True
    show_progress: bool = False

    _compiled: Any = field(default=None, init=False, repr=False)

    def supports(self, study: StudyData) -> bool:
        for subj in study.subjects:
            for blk in subj.blocks:
                if blk.env_spec is None or not self.model.supports(blk.env_spec):
                    return False
                if blk.event_log is None:
                    return False
        return True

    def fit(self, *, study: StudyData, rng: np.random.Generator) -> FitResult:
        if not self.supports(study):
            raise CompatibilityError("Data/model incompatible or event logs missing.")

        adapter = resolve_stan_adapter(self.model)
        prog = adapter.program("hier")

        stan_code = load_stan_code(prog.family, prog.key)
        if self._compiled is None:
            self._compiled = compile_cmdstan(stan_code, prog.program_name)

        data = study_to_stan_data(study)
        adapter.augment_study_data(data)

        data |= priors_to_stan_data(
            priors_cfg=self.hyper_priors,
            adapter=adapter,
            family="hier",
            forbid_extra=self.forbid_extra_priors,
        )

        seed = int(rng.integers(1, 2**31 - 1))
        fit = self._compiled.sample(
            data=data,
            chains=self.chains,
            iter_warmup=self.iter_warmup,
            iter_sampling=self.iter_sampling,
            seed=seed,
            adapt_delta=self.adapt_delta,
            max_treedepth=self.max_treedepth,
            show_progress=self.show_progress,
        )

        subj_hats: dict[str, dict[str, float]] = {}
        subj_ids = [s.subject_id for s in study.subjects]

        for p in adapter.subject_param_names():
            draws = fit.stan_variable(p)  # expects vector[N]
            means = np.mean(draws, axis=0)
            for i, sid in enumerate(subj_ids):
                subj_hats.setdefault(sid, {})[p] = float(means[i])

        pop_hat: dict[str, float] = {}
        for nm in adapter.population_var_names():
            try:
                pop_hat[nm] = float(np.mean(fit.stan_variable(nm)))
            except KeyError:
                continue

        summ = None
        try:
            summ = fit.summary()
        except Exception:  # noqa: BLE001
            summ = None

        diags = {
            "seed": seed,
            "rhat_max": _safe_summary_metric(summ, ["R_hat", "r_hat"], agg="max"),
            "ess_bulk_min": _safe_summary_metric(summ, ["ESS_bulk", "ess_bulk"], agg="min"),
        }

        return FitResult(
            population_hat=pop_hat,
            subject_hats=subj_hats,
            success=True,
            message="OK",
            diagnostics=diags,
        )
