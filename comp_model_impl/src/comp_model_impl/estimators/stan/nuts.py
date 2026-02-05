"""Stan-based NUTS estimators.

This module provides two concrete estimators built on CmdStanPy:

* :class:`StanNUTSSubjectwiseEstimator` - fits each subject independently.
* :class:`StanHierarchicalNUTSEstimator` - fits a hierarchical model across subjects.

Both estimators rely on an event-log representation stored on blocks and on Stan
program templates distributed with this package.

Notes
-----
Stan programs are selected through :class:`StanAdapter` implementations. The
adapter maps a model to a Stan program family and provides prior requirements.

Examples
--------
Subject-wise NUTS:

>>> from comp_model_impl.models import QRL
>>> from comp_model_impl.estimators.stan.nuts import StanNUTSSubjectwiseEstimator
>>> est = StanNUTSSubjectwiseEstimator(model=QRL(), priors={"alpha": {"family": "beta", "a": 2, "b": 2}})
>>> # est.fit(study=study, rng=np.random.default_rng(0))  # doctest: +SKIP
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
from .exporter import (
    subject_to_stan_data,
    study_to_stan_data,
    subject_to_stan_data_within_subject,
    study_to_stan_data_within_subject,
)
from ...models.within_subject_shared_delta import (
    ConditionedSharedDeltaModel,
    ConditionedSharedDeltaSocialModel,
)

from ..within_subject_shared_delta import (
    _infer_conditions_from_study,
    _ensure_within_subject_structure,
)

from .priors import priors_to_stan_data
from .adapters import StanAdapter


def _safe_summary_metric(summary: Any, candidates: list[str], *, agg: str) -> float | None:
    """Extract a scalar diagnostic from a CmdStanPy summary dataframe.

    Parameters
    ----------
    summary : Any
        CmdStanPy summary dataframe-like object.
    candidates : list[str]
        Candidate column names to search for (case-insensitive).
    agg : {"max", "min"}
        Aggregation to apply across the column.

    Returns
    -------
    float or None
        Aggregated diagnostic value, or ``None`` if unavailable.
    """
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


def _strip_hat(name: str) -> str:
    """Strip a trailing ``_hat`` suffix if present.

    Parameters
    ----------
    name : str
        Variable name.

    Returns
    -------
    str
        Name without the trailing ``_hat`` suffix.
    """
    return name[:-4] if name.endswith("_hat") else name


def _delta_labels(condition_labels: list[str], baseline_idx_1based: int) -> list[str]:
    """Return non-baseline condition labels.

    Parameters
    ----------
    condition_labels : list[str]
        Full ordered list of conditions.
    baseline_idx_1based : int
        One-based index of the baseline condition.

    Returns
    -------
    list[str]
        Condition labels excluding the baseline.
    """
    return [lbl for i, lbl in enumerate(condition_labels, start=1) if i != baseline_idx_1based]


def _flatten_mean(
    *,
    name: str,
    mean: Any,
    condition_labels: list[str] | None = None,
    baseline_idx_1based: int | None = None,
) -> dict[str, float]:
    """Flatten a mean value (possibly vector/matrix) into a dict of scalars.

    Parameters
    ----------
    name : str
        Base variable name.
    mean : Any
        Mean value (scalar, vector, or matrix-like).
    condition_labels : list[str] or None, optional
        Condition labels used to name entries for condition-indexed vectors.
    baseline_idx_1based : int or None, optional
        One-based index of the baseline condition.

    Returns
    -------
    dict[str, float]
        Flat mapping from names to scalar values.

    Notes
    -----
    Heuristics:
    - Scalars keep their name (minus optional ``_hat`` suffix).
    - 1D arrays of length ``C`` are labelled with condition labels if provided.
    - 1D arrays of length ``C-1`` are labelled with non-baseline condition
      labels if provided.
    - Other arrays are indexed with 1-based ``[i]`` or ``[i,j]`` suffixes.

    Examples
    --------
    >>> _flatten_mean(name="alpha_p_hat", mean=[0.1, 0.2], condition_labels=["A", "B"])
    {'alpha_p__A': 0.1, 'alpha_p__B': 0.2}
    >>> _flatten_mean(
    ...     name="beta__delta_hat",
    ...     mean=[-0.5, 0.25],
    ...     condition_labels=["A", "B", "C"],
    ...     baseline_idx_1based=2,
    ... )
    {'beta__delta__A': -0.5, 'beta__delta__C': 0.25}
    >>> _flatten_mean(name="mu_alpha_hat", mean=[[1.0, 2.0], [3.0, 4.0]])
    {'mu_alpha[1,1]': 1.0, 'mu_alpha[1,2]': 2.0, 'mu_alpha[2,1]': 3.0, 'mu_alpha[2,2]': 4.0}
    >>> _flatten_mean(name="tau_hat", mean=[[[1.0, 2.0], [3.0, 4.0]]])
    {'tau[1,1,1]': 1.0, 'tau[1,1,2]': 2.0, 'tau[1,2,1]': 3.0, 'tau[1,2,2]': 4.0}
    """
    base = _strip_hat(str(name))

    arr = np.asarray(mean)
    if arr.shape == ():
        return {base: float(arr)}

    # 1D vector
    if arr.ndim == 1:
        L = int(arr.shape[0])
        if condition_labels is not None and L == len(condition_labels):
            return {f"{base}__{condition_labels[i]}": float(arr[i]) for i in range(L)}
        if (
            condition_labels is not None
            and baseline_idx_1based is not None
            and L == len(condition_labels) - 1
        ):
            dl = _delta_labels(condition_labels, baseline_idx_1based)
            if len(dl) == L:
                return {f"{base}__{dl[i]}": float(arr[i]) for i in range(L)}
        return {f"{base}[{i+1}]": float(arr[i]) for i in range(L)}

    # 2D matrix
    if arr.ndim == 2:
        R, C = int(arr.shape[0]), int(arr.shape[1])
        out: dict[str, float] = {}
        for i in range(R):
            for j in range(C):
                out[f"{base}[{i+1},{j+1}]"] = float(arr[i, j])
        return out

    # higher dims: flatten indices
    out: dict[str, float] = {}
    it = np.nditer(arr, flags=["multi_index"])
    for x in it:
        idx = ",".join(str(i + 1) for i in it.multi_index)
        out[f"{base}[{idx}]"] = float(x)
    return out


def _is_within_subject_model(model: ComputationalModel) -> bool:
    """Return ``True`` if the model uses the shared+delta wrapper."""
    return isinstance(model, (ConditionedSharedDeltaModel, ConditionedSharedDeltaSocialModel))


def _load_yaml(path_to_yml: str) -> dict[str, Any]:
    """Load a YAML file into a dictionary.

    Parameters
    ----------
    path_to_yml : str
        Path to a YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed YAML mapping.

    Raises
    ------
    ImportError
        If PyYAML is not installed.
    """
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError("PyYAML is required to load YAML plans. Install with: pip install pyyaml") from e
    with open(path_to_yml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    return cfg


@dataclass(slots=True)
class StanNUTSSubjectwiseEstimator(Estimator):
    """Independent (per-subject) Bayesian inference with NUTS.

    Parameters
    ----------
    model : comp_model_core.interfaces.model.ComputationalModel
        Model to fit. Must have a registered :class:`StanAdapter`.
    priors : Mapping[str, Any] or str
        Prior configuration mapping or path to a YAML file.
    chains : int, optional
        Number of MCMC chains.
    iter_warmup : int, optional
        Warmup iterations per chain.
    iter_sampling : int, optional
        Sampling iterations per chain.
    adapt_delta : float, optional
        Target acceptance probability for NUTS.
    max_treedepth : int, optional
        Maximum tree depth for NUTS.
    forbid_extra_priors : bool, optional
        If ``True``, reject priors not required by the adapter.
    show_progress : bool, optional
        If ``True``, display CmdStanPy progress output.

    Notes
    -----
    This estimator expects event logs on each block. For within-subject shared+delta
    models, condition labels are included in the Stan data.
    """

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
        """Load priors from YAML if a path was provided."""
        if isinstance(self.priors, str):
            self.priors = _load_yaml(self.priors)


    def supports(self, study: StudyData) -> bool:
        """Return ``True`` if the study is compatible with the model."""
        # We require block-level conditions.
        try:
            _ = _infer_conditions_from_study(study)
        except Exception:
            return False
        
        for subj in study.subjects:
            for blk in subj.blocks:
                if blk.env_spec is None or not self.model.supports(blk.env_spec):
                    return False
                if blk.event_log is None:
                    return False
        return True

    def fit(self, *, study: StudyData, rng: np.random.Generator) -> FitResult:
        """Fit each subject independently using NUTS.

        Parameters
        ----------
        study : comp_model_core.data.types.StudyData
            Study with event logs.
        rng : numpy.random.Generator
            Random number generator used for per-subject seeds.

        Returns
        -------
        comp_model_core.interfaces.estimator.FitResult
            Subject-level parameter estimates and diagnostics.
        """
        if not self.supports(study):
            raise CompatibilityError("Data/model incompatible or event logs missing.")

        adapter = resolve_stan_adapter(self.model)
        prog = adapter.program("indiv")

        stan_code = load_stan_code(prog.family, prog.key)
        if self._compiled is None:
            self._compiled = compile_cmdstan(stan_code, prog.program_name)

        is_ws = _is_within_subject_model(self.model)
        condition_labels: list[str] | None = None
        baseline_cond_label: str | None = None
        if is_ws:
            condition_labels = [str(c) for c in getattr(self.model, 'conditions')]
            baseline_cond_label = str(getattr(self.model, 'baseline_condition'))

        subj_hats: dict[str, dict[str, float]] = {}
        per_subject_diags: dict[str, Any] = {}

        for subj in study.subjects:
            if is_ws:
                data = subject_to_stan_data_within_subject(
                    subj,
                    conditions=condition_labels or [],
                    baseline_condition=baseline_cond_label or '',
                )
            else:
                data = subject_to_stan_data(subj)
            adapter.augment_subject_data(data)
            data |= priors_to_stan_data(
                priors_cfg=self.priors,
                adapter=adapter,
                family="indiv",
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

            hats: dict[str, float] = {}
            if is_ws:
                # For within-subject models, Stan variables include per-condition vectors.
                bl_idx = int(data.get("baseline_cond", 1))
                for p in adapter.subject_param_names():
                    draws = fit.stan_variable(p)
                    mean = np.mean(draws, axis=0)
                    hats.update(
                        _flatten_mean(
                            name=p,
                            mean=mean,
                            condition_labels=condition_labels,
                            baseline_idx_1based=bl_idx,
                        )
                    )
            else:
                for p in adapter.subject_param_names():
                    draws = fit.stan_variable(p)
                    hats[p] = float(np.mean(draws))
            subj_hats[subj.subject_id] = hats

            summ = None
            try:
                summ = fit.summary()
            except Exception:  # noqa: BLE001
                summ = None

            required = adapter.required_priors("indiv")
            per_subject_diags[subj.subject_id] = {
                "seed": seed,
                "required_priors": list(required),
                "rhat_max": _safe_summary_metric(summ, ["R_hat", "r_hat"], agg="max"),
                "ess_bulk_min": _safe_summary_metric(summ, ["ESS_bulk", "ess_bulk"], agg="min"),
            }
            if is_ws:
                per_subject_diags[subj.subject_id]["conditions"] = list(condition_labels or [])
                per_subject_diags[subj.subject_id]["baseline_condition"] = baseline_cond_label

        return FitResult(
            subject_hats=subj_hats,
            success=True,
            message="OK",
            diagnostics={"per_subject": per_subject_diags},
        )


@dataclass(slots=True)
class StanHierarchicalNUTSEstimator(Estimator):
    """Hierarchical Bayesian inference with NUTS over multiple subjects.

    Parameters
    ----------
    model : comp_model_core.interfaces.model.ComputationalModel
        Model to fit. Must have a registered :class:`StanAdapter`.
    hyper_priors : Mapping[str, Any] or str
        Hyper-prior configuration mapping or path to a YAML file.
    chains : int, optional
        Number of MCMC chains.
    iter_warmup : int, optional
        Warmup iterations per chain.
    iter_sampling : int, optional
        Sampling iterations per chain.
    adapt_delta : float, optional
        Target acceptance probability for NUTS.
    max_treedepth : int, optional
        Maximum tree depth for NUTS.
    forbid_extra_priors : bool, optional
        If ``True``, reject priors not required by the adapter.
    show_progress : bool, optional
        If ``True``, display CmdStanPy progress output.

    Notes
    -----
    This estimator fits a population model across subjects and optionally
    returns population-level summaries when supported by the Stan program.
    """

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

    def __post_init__(self):
        """Load hyper-priors from YAML if a path was provided."""
        if isinstance(self.hyper_priors, str):
            self.hyper_priors = _load_yaml(self.hyper_priors)

    def supports(self, study: StudyData) -> bool:
        """Return ``True`` if the study is compatible with the model."""
        for subj in study.subjects:
            for blk in subj.blocks:
                if blk.env_spec is None or not self.model.supports(blk.env_spec):
                    return False
                if blk.event_log is None:
                    return False
        return True

    def fit(self, *, study: StudyData, rng: np.random.Generator) -> FitResult:
        """Fit a hierarchical model using NUTS.

        Parameters
        ----------
        study : comp_model_core.data.types.StudyData
            Study with event logs.
        rng : numpy.random.Generator
            Random number generator used for sampling seed.

        Returns
        -------
        comp_model_core.interfaces.estimator.FitResult
            Subject- and population-level parameter estimates with diagnostics.
        """
        if not self.supports(study):
            raise CompatibilityError("Data/model incompatible or event logs missing.")

        adapter = resolve_stan_adapter(self.model)
        prog = adapter.program("hier")

        is_ws = _is_within_subject_model(self.model)
        condition_labels: list[str] | None = None
        baseline_cond_label: str | None = None
        if is_ws:
            condition_labels = [str(c) for c in getattr(self.model, 'conditions')]
            baseline_cond_label = str(getattr(self.model, 'baseline_condition'))

        stan_code = load_stan_code(prog.family, prog.key)
        if self._compiled is None:
            self._compiled = compile_cmdstan(stan_code, prog.program_name)

        if is_ws:
            data = study_to_stan_data_within_subject(
                study,
                conditions=condition_labels or [],
                baseline_condition=baseline_cond_label or '',
            )
        else:
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

        if is_ws:
            bl_idx = int(data.get("baseline_cond", 1))
            for p in adapter.subject_param_names():
                draws = fit.stan_variable(p)
                means = np.mean(draws, axis=0)
                arr = np.asarray(means)
                base = _strip_hat(str(p))
                # subject-level vectors: shape (N,)
                if arr.ndim == 1 and int(arr.shape[0]) == len(subj_ids):
                    for i, sid in enumerate(subj_ids):
                        subj_hats.setdefault(sid, {})[base] = float(arr[i])
                    continue

                # subject-level matrices: shape (N, K)
                if arr.ndim >= 2 and int(arr.shape[0]) == len(subj_ids):
                    for i, sid in enumerate(subj_ids):
                        row = arr[i]
                        subj_hats.setdefault(sid, {}).update(
                            _flatten_mean(
                                name=base,
                                mean=row,
                                condition_labels=condition_labels,
                                baseline_idx_1based=bl_idx,
                            )
                        )
                    continue

                # fallback: store a single scalar across subjects
                for sid in subj_ids:
                    subj_hats.setdefault(sid, {})[base] = float(np.mean(arr))
        else:
            for p in adapter.subject_param_names():
                draws = fit.stan_variable(p)  # expects vector[N]
                means = np.mean(draws, axis=0)
                for i, sid in enumerate(subj_ids):
                    subj_hats.setdefault(sid, {})[p] = float(means[i])

        pop_hat: dict[str, float] = {}
        if is_ws:
            bl_idx = int(data.get("baseline_cond", 1))
            for nm in adapter.population_var_names():
                try:
                    draws = fit.stan_variable(nm)
                except KeyError:
                    continue
                mean = np.mean(draws, axis=0)
                pop_hat.update(
                    _flatten_mean(
                        name=nm,
                        mean=mean,
                        condition_labels=condition_labels,
                        baseline_idx_1based=bl_idx,
                    )
                )
        else:
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
