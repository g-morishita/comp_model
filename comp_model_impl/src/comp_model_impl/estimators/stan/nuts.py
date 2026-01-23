from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from comp_model_core.data.types import StudyData
from comp_model_core.errors import CompatibilityError
from comp_model_core.interfaces.estimator import Estimator, FitResult
from comp_model_core.interfaces.model import ComputationalModel

from comp_model_impl.estimators.stan.exporter import subject_to_stan_data, study_to_stan_data
from comp_model_impl.estimators.stan.priors import Prior, priors_to_stan_data
from comp_model_impl.estimators.stan.cmdstan_utils import load_stan_code, compile_cmdstan

# defaults you can override via config
DEFAULTS_VS_INDIV = {
    "alpha_p": Prior("beta", 1.0, 1.0),
    "alpha_i": Prior("beta", 1.0, 1.0),
    "beta":    Prior("lognormal", 0.0, 1.0),
    "kappa":   Prior("normal", 0.0, 1.0),
}
DEFAULTS_VIC_INDIV = {
    "alpha_o": Prior("beta", 1.0, 1.0),
    "beta":    Prior("lognormal", 0.0, 1.0),
}

# hyper defaults (hierarchical)
DEFAULTS_VS_HYPER = {
    "mu_ap": Prior("normal", 0.0, 1.5),
    "sd_ap": Prior("exponential", 1.0),
    "mu_ai": Prior("normal", 0.0, 1.5),
    "sd_ai": Prior("exponential", 1.0),
    "mu_b":  Prior("normal", 0.0, 1.5),
    "sd_b":  Prior("exponential", 1.0),
    "mu_k":  Prior("normal", 0.0, 1.5),
    "sd_k":  Prior("exponential", 1.0),
}
DEFAULTS_VIC_HYPER = {
    "mu_ao": Prior("normal", 0.0, 1.5),
    "sd_ao": Prior("exponential", 1.0),
    "mu_b":  Prior("normal", 0.0, 1.5),
    "sd_b":  Prior("exponential", 1.0),
}

def _model_key(model: ComputationalModel) -> str:
    n = model.__class__.__name__.lower()
    if n == "vs":
        return "vs"
    if "vicarious" in n:
        return "vicarious_rl"
    raise ValueError(f"No Stan mapping for model {model.__class__.__name__}")

@dataclass(slots=True)
class StanNUTSSubjectwiseEstimator(Estimator):
    model: ComputationalModel
    priors: Mapping[str, Any] | None = None  # config priors for individual parameters

    chains: int = 4
    iter_warmup: int = 500
    iter_sampling: int = 1000
    adapt_delta: float = 0.9
    max_treedepth: int = 12

    _compiled: Any = field(default=None, init=False, repr=False)

    def supports(self, study: StudyData) -> bool:
        for subj in study.subjects:
            for blk in subj.blocks:
                if blk.task_spec is None or not self.model.supports(blk.task_spec):
                    return False
                if "event_log" not in (blk.metadata or {}):
                    return False
        return True

    def fit(self, *, study: StudyData, rng: np.random.Generator) -> FitResult:
        if not self.supports(study):
            raise CompatibilityError("Data/model incompatible or event logs missing.")

        key = _model_key(self.model)
        stan_code = load_stan_code("indiv", key)
        if self._compiled is None:
            self._compiled = compile_cmdstan(stan_code, f"{key}_indiv")

        subj_hats: dict[str, dict[str, float]] = {}

        for subj in study.subjects:
            data = subject_to_stan_data(subj)

            # bounds / config
            data["beta_lower"] = 1e-6
            data["beta_upper"] = float(getattr(self.model, "beta_max", 20.0))
            if key == "vs":
                data["kappa_abs_max"] = float(getattr(self.model, "kappa_abs_max", 5.0))
                data["pseudo_reward"] = float(getattr(self.model, "pseudo_reward", 1.0))

            defaults = DEFAULTS_VS_INDIV if key == "vs" else DEFAULTS_VIC_INDIV
            data |= priors_to_stan_data(self.priors, defaults)

            seed = int(rng.integers(1, 2**31 - 1))
            fit = self._compiled.sample(
                data=data,
                chains=self.chains,
                iter_warmup=self.iter_warmup,
                iter_sampling=self.iter_sampling,
                seed=seed,
                adapt_delta=self.adapt_delta,
                max_treedepth=self.max_treedepth,
                show_progress=False,
            )

            hats: dict[str, float] = {}
            for p in self.model.param_schema.names:
                try:
                    draws = fit.stan_variable(p)
                except KeyError:
                    continue
                hats[p] = float(np.mean(draws))
            subj_hats[subj.subject_id] = hats

        return FitResult(subject_hats=subj_hats, success=True, message="OK", diagnostics={})

@dataclass(slots=True)
class StanHierarchicalNUTSEstimator(Estimator):
    model: ComputationalModel
    hyper_priors: Mapping[str, Any] | None = None

    chains: int = 4
    iter_warmup: int = 800
    iter_sampling: int = 1200
    adapt_delta: float = 0.92
    max_treedepth: int = 12

    _compiled: Any = field(default=None, init=False, repr=False)

    def supports(self, study: StudyData) -> bool:
        for subj in study.subjects:
            for blk in subj.blocks:
                if blk.task_spec is None or not self.model.supports(blk.task_spec):
                    return False
                if "event_log" not in (blk.metadata or {}):
                    return False
        return True

    def fit(self, *, study: StudyData, rng: np.random.Generator) -> FitResult:
        if not self.supports(study):
            raise CompatibilityError("Data/model incompatible or event logs missing.")

        key = _model_key(self.model)
        stan_code = load_stan_code("hier", key)
        if self._compiled is None:
            self._compiled = compile_cmdstan(stan_code, f"{key}_hier")

        data = study_to_stan_data(study)
        data["beta_lower"] = 1e-6
        data["beta_upper"] = float(getattr(self.model, "beta_max", 20.0))
        if key == "vs":
            data["kappa_abs_max"] = float(getattr(self.model, "kappa_abs_max", 5.0))
            data["pseudo_reward"] = float(getattr(self.model, "pseudo_reward", 1.0))

        defaults = DEFAULTS_VS_HYPER if key == "vs" else DEFAULTS_VIC_HYPER
        data |= priors_to_stan_data(self.hyper_priors, defaults)

        seed = int(rng.integers(1, 2**31 - 1))
        fit = self._compiled.sample(
            data=data,
            chains=self.chains,
            iter_warmup=self.iter_warmup,
            iter_sampling=self.iter_sampling,
            seed=seed,
            adapt_delta=self.adapt_delta,
            max_treedepth=self.max_treedepth,
            show_progress=False,
        )

        # subject-level means
        subj_hats: dict[str, dict[str, float]] = {}
        subj_ids = [s.subject_id for s in study.subjects]
        for p in self.model.param_schema.names:
            try:
                draws = fit.stan_variable(p)  # expects vector[N]
            except KeyError:
                continue
            means = np.mean(draws, axis=0)
            for i, sid in enumerate(subj_ids):
                subj_hats.setdefault(sid, {})[p] = float(means[i])

        return FitResult(subject_hats=subj_hats, success=True, message="OK", diagnostics={})
