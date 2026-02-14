"""Tests for model recovery runtime helpers and runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

from comp_model_core.data.types import StudyData, SubjectData
from comp_model_core.interfaces.estimator import Estimator, FitResult
from comp_model_core.interfaces.model import ComputationalModel
from comp_model_core.params import ParamDef, ParameterSchema
from comp_model_core.plans.block import BlockPlan, StudyPlan
from comp_model_core.spec import EnvironmentSpec

from comp_model_impl.models.qrl.qrl import QRL
from comp_model_impl.recovery.model.config import (
    CandidateModelSpec,
    GeneratingModelSpec,
    ModelRecoveryConfig,
    OutputSpec,
    SelectionSpec,
)
from comp_model_impl.recovery.model.criteria import get_criterion
from comp_model_impl.recovery.model.likelihood import LikelihoodSummary
from comp_model_impl.recovery.model import run as run_mod
from comp_model_impl.recovery.model import resolution as resolve_mod
from comp_model_impl.recovery.parameter.config import SamplingSpec
from comp_model_impl.register import make_registry


@dataclass(slots=True)
class _TinyModel(ComputationalModel):
    """Minimal model implementation used in run tests."""

    label: str = "model"
    theta: float = 0.5

    @property
    def param_schema(self) -> ParameterSchema:
        return ParameterSchema((ParamDef("theta", float(self.theta)),))

    def supports(self, spec: EnvironmentSpec) -> bool:
        return True

    def reset_block(self, *, spec: EnvironmentSpec) -> None:
        return

    def action_probs(self, *, state, spec: EnvironmentSpec) -> np.ndarray:
        return np.array([0.5, 0.5], dtype=float)

    def update(self, *, state, action: int, outcome, spec: EnvironmentSpec, info=None, rng=None) -> None:
        return


class _NoSchemaModel:
    """Model-like object with no usable param schema names."""

    param_schema = object()


@dataclass(slots=True)
class _DummyEstimator(Estimator):
    """Simple estimator for controlled run-model-recovery tests."""

    model: ComputationalModel
    should_fail: bool = False

    def supports(self, study: StudyData) -> bool:
        return True

    def fit(
        self,
        *,
        study: StudyData,
        rng: np.random.Generator,
        fixed_params: Mapping[str, float] | None = None,
    ) -> FitResult:
        if self.should_fail:
            raise RuntimeError("estimation failed")
        return FitResult(
            subject_hats={"s1": {"theta": 0.4}},
            value=1.23,
            success=True,
            message="ok",
            diagnostics={"fixed_params": None if fixed_params is None else dict(fixed_params)},
        )


class _DummyGenerator:
    """Generator-like object with deterministic study output."""

    def __init__(self, study: StudyData) -> None:
        self._study = study

    def simulate_study(
        self,
        *,
        block_runner_builder,
        model: ComputationalModel,
        subj_params,
        subject_block_plans,
        rng: np.random.Generator,
    ) -> StudyData:
        return self._study


def _minimal_plan() -> StudyPlan:
    """Build a minimal one-subject plan."""
    block = BlockPlan(
        block_id="b1",
        n_trials=1,
        condition="c1",
        bandit_type="BernoulliBanditEnv",
        bandit_config={"probs": [0.2, 0.8]},
        trial_specs=[
            {
                "self_outcome": {"kind": "VERIDICAL"},
                "available_actions": [0, 1],
            }
        ],
    )
    return StudyPlan(subjects={"s1": [block]})


def test_resolve_estimator_callable() -> None:
    """Estimator resolver should use estimator registry keys."""
    registries = make_registry()

    cls = resolve_mod.resolve_estimator_callable(
        "TransformedMLESubjectwiseEstimator",
        registries=registries,
    )
    assert cls.__name__ == "TransformedMLESubjectwiseEstimator"

    with pytest.raises(ValueError, match="registered estimator key"):
        _ = resolve_mod.resolve_estimator_callable(
            "comp_model_impl.estimators.mle_event_log.TransformedMLESubjectwiseEstimator",
            registries=registries,
        )

    with pytest.raises(ValueError, match="Could not resolve estimator"):
        _ = resolve_mod.resolve_estimator_callable(
            "DefinitelyUnknownEstimator",
            registries=registries,
        )


def test_build_nested_and_build_kwargs() -> None:
    """Nested inline factory mappings should be resolved recursively."""
    registries = make_registry()

    built = run_mod._build_nested(
        {"factory": "QRL", "kwargs": {"alpha": 0.2, "beta": 3.0}},
        registries=registries,
    )
    assert isinstance(built, QRL)
    assert built.alpha == pytest.approx(0.2)
    assert built.beta == pytest.approx(3.0)

    kwargs = run_mod._build_kwargs(
        {
            "wrapped_model": {"factory": "QRL", "kwargs": {"alpha": 0.1, "beta": 2.0}},
            "n_starts": 3,
        },
        registries=registries,
    )
    assert isinstance(kwargs["wrapped_model"], QRL)
    assert kwargs["n_starts"] == 3

    with pytest.raises(TypeError, match="Inline factory kwargs must be a mapping"):
        _ = run_mod._build_nested({"factory": "QRL", "kwargs": 1}, registries=registries)

    with pytest.raises(TypeError, match="kwargs must be a mapping"):
        _ = run_mod._build_kwargs(123, registries=registries)  # type: ignore[arg-type]


def test_build_from_reference_model_and_estimator() -> None:
    """Reference builder should construct both models and estimators."""
    registries = make_registry()

    model = run_mod._build_from_reference(
        reference="QRL",
        kwargs={"alpha": 0.3, "beta": 4.0},
        registries=registries,
        kind="model",
    )
    assert isinstance(model, QRL)
    assert model.alpha == pytest.approx(0.3)

    with pytest.raises(ValueError, match="registered model key"):
        _ = run_mod._build_from_reference(
            reference="comp_model_impl.models.qrl.qrl.QRL",
            kwargs={"alpha": 0.1, "beta": 2.0},
            registries=registries,
            kind="model",
        )

    est = run_mod._build_from_reference(
        reference="TransformedMLESubjectwiseEstimator",
        kwargs={"model": QRL(), "n_starts": 1, "maxiter": 5},
        registries=registries,
        kind="estimator",
    )
    assert isinstance(est, Estimator)

    with pytest.raises(ValueError, match="Unknown kind"):
        _ = run_mod._build_from_reference(reference="QRL", kwargs={}, registries=registries, kind="unknown")


def test_build_model_unknown_key() -> None:
    """Model builder should reject unknown model keys."""
    with pytest.raises(ValueError, match="Could not resolve model"):
        _ = run_mod._build_model(
            "builtins.dict",
            model_kwargs={},
            registries=make_registry(),
        )


def test_build_estimator_injects_model() -> None:
    """Estimator builder should inject model argument when supported."""
    model = QRL()
    est = run_mod._build_estimator(
        "TransformedMLESubjectwiseEstimator",
        estimator_kwargs={"n_starts": 1, "maxiter": 5},
        model=model,
        registries=make_registry(),
    )
    assert isinstance(est, Estimator)
    assert getattr(est, "model", None) is model


def test_subject_id_and_params_hat_helpers() -> None:
    """Plan subject IDs and FitResult parameter mappings should normalize correctly."""
    plan = _minimal_plan()
    assert run_mod._subject_ids_from_plan(plan) == ["s1"]

    fit_subject = FitResult(subject_hats={"s1": {"a": 1.0}})
    assert run_mod._params_hat_by_subject(fit_subject, ["s1"]) == {"s1": {"a": 1.0}}

    fit_shared = FitResult(params_hat={"a": 2.0})
    assert run_mod._params_hat_by_subject(fit_shared, ["s1", "s2"]) == {
        "s1": {"a": 2.0},
        "s2": {"a": 2.0},
    }

    fit_empty = FitResult()
    assert run_mod._params_hat_by_subject(fit_empty, ["s1"]) == {}


def test_count_free_params_uses_schema_and_population() -> None:
    """Free-parameter counter should use schema names and population estimates."""
    model = QRL()
    fit = FitResult(population_hat={"mu": 0.1})
    k_per, k_total = run_mod._count_free_params(
        model=model,
        fit=fit,
        n_subjects=3,
        fixed_params={"alpha": 0.2},
    )
    assert k_per == 1
    assert k_total == 4


def test_count_free_params_falls_back_to_subject_hats() -> None:
    """Free-parameter counter should fall back to subject-hat dimensions."""
    fit = FitResult(
        subject_hats={
            "s1": {"a": 1.0, "b": 2.0},
            "s2": {"a": 0.5, "b": 1.5},
        }
    )
    k_per, k_total = run_mod._count_free_params(
        model=_NoSchemaModel(),  # type: ignore[arg-type]
        fit=fit,
        n_subjects=2,
        fixed_params={"a": 0.0},
    )
    assert k_per == 1
    assert k_total == 2


def test_extract_waic_from_fit_diagnostics_top_level() -> None:
    """WAIC extraction should read top-level diagnostics directly."""
    fit = FitResult(
        diagnostics={
            "waic": 123.4,
            "elpd_waic": -61.7,
            "p_waic": 9.1,
            "waic_n_obs": 200,
        }
    )
    out = run_mod._extract_waic_from_fit_diagnostics(fit)
    assert out == {
        "waic": pytest.approx(123.4),
        "elpd_waic": pytest.approx(-61.7),
        "p_waic": pytest.approx(9.1),
        "waic_n_obs": pytest.approx(200.0),
    }


def test_extract_waic_from_fit_diagnostics_per_subject_sum() -> None:
    """WAIC extraction should aggregate per-subject diagnostics when present."""
    fit = FitResult(
        diagnostics={
            "per_subject": {
                "s1": {"waic": 10.0, "elpd_waic": -5.0, "p_waic": 1.0, "waic_n_obs": 20},
                "s2": {"waic": 12.0, "elpd_waic": -6.0, "p_waic": 1.5, "waic_n_obs": 25},
            }
        }
    )
    out = run_mod._extract_waic_from_fit_diagnostics(fit)
    assert out == {
        "waic": pytest.approx(22.0),
        "elpd_waic": pytest.approx(-11.0),
        "p_waic": pytest.approx(2.5),
        "waic_n_obs": pytest.approx(45.0),
    }


def test_extract_waic_from_fit_diagnostics_missing_returns_none() -> None:
    """WAIC extraction should return None when diagnostics do not carry WAIC."""
    fit = FitResult(diagnostics={"per_subject": {"s1": {"rhat_max": 1.01}}})
    assert run_mod._extract_waic_from_fit_diagnostics(fit) is None


def test_select_winner_logic() -> None:
    """Winner selection should respect criterion direction and tie strategy."""
    rows = [
        {"candidate_model": "A", "score": 10.0, "k_total": 3, "ll_total": 10.0},
        {"candidate_model": "B", "score": 8.0, "k_total": 1, "ll_total": 8.0},
    ]
    out = run_mod._select_winner(
        rows,
        criterion=get_criterion("loglike"),
        tie_break="first",
        atol=1e-9,
    )
    assert out["selected_model"] == "A"
    assert out["second_best_model"] == "B"
    assert out["delta_to_second"] == pytest.approx(2.0)

    tie_rows = [
        {"candidate_model": "A", "score": 1.0, "k_total": 4, "ll_total": -1.0},
        {"candidate_model": "B", "score": 1.0, "k_total": 2, "ll_total": -1.0},
    ]
    out_simpler = run_mod._select_winner(
        tie_rows,
        criterion=get_criterion("aic"),
        tie_break="simpler",
        atol=1e-9,
    )
    assert out_simpler["selected_model"] == "B"

    out_first = run_mod._select_winner(
        tie_rows,
        criterion=get_criterion("aic"),
        tie_break="first",
        atol=1e-9,
    )
    assert out_first["selected_model"] == "A"

    with pytest.raises(ValueError, match="No candidate rows"):
        _ = run_mod._select_winner([], criterion=get_criterion("bic"), tie_break="first", atol=1e-9)


def test_run_model_recovery_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Model recovery run should produce fit/winner tables and artifact files."""
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    plan = _minimal_plan()

    study = StudyData(subjects=[SubjectData(subject_id="s1", blocks=[])], metadata={})
    generator = _DummyGenerator(study=study)

    out_root = tmp_path / "out"

    cfg = ModelRecoveryConfig(
        plan_path=str(plan_path),
        n_reps=1,
        seed=123,
        generating=[
            GeneratingModelSpec(
                name="gen",
                model="gen_model",
                sampling=SamplingSpec(mode="fixed", fixed={"theta": 0.2}),
            )
        ],
        candidates=[
            CandidateModelSpec(name="good", model="good_model", estimator="ok_est"),
            CandidateModelSpec(name="bad", model="bad_model", estimator="raise_est"),
        ],
        selection=SelectionSpec(criterion="loglike", tie_break="first"),
        output=OutputSpec(
            out_dir=str(out_root),
            save_format="csv",
            save_config=True,
            save_fit_diagnostics=True,
            save_simulated_study=False,
        ),
    )

    def fake_make_unique_run_dir(base: str, git_commit: str | None = None) -> Path:
        out = Path(base) / "run_fixed"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def fake_build_model(model: str, *, model_kwargs, registries) -> ComputationalModel:
        return _TinyModel(label=str(model))

    def fake_build_estimator(estimator: str, *, estimator_kwargs, model, registries) -> Estimator:
        return _DummyEstimator(model=model, should_fail=(estimator == "raise_est"))

    def fake_sample_subject_params(*, cfg, model, subject_ids, rng):
        return ({sid: {"theta": 0.2} for sid in subject_ids}, None)

    def fake_compute_likelihood_summary(*, study, model, subject_params):
        ll = 12.0 if getattr(model, "label", "") == "good_model" else 3.0
        return LikelihoodSummary(
            ll_total=ll,
            ll_by_subject={"s1": ll},
            n_obs_total=5,
            n_obs_by_subject={"s1": 5},
        )

    progress_calls: list[tuple[int, int]] = []

    monkeypatch.setattr(run_mod, "load_study_plan_json", lambda _: plan)
    monkeypatch.setattr(run_mod, "git_info_for_module", lambda _: {"comp_model_impl_git_commit": "abc"})
    monkeypatch.setattr(run_mod, "make_unique_run_dir", fake_make_unique_run_dir)
    monkeypatch.setattr(run_mod, "_plan_summary", lambda _: {"n_subjects": 1})
    monkeypatch.setattr(run_mod, "_build_model", fake_build_model)
    monkeypatch.setattr(run_mod, "_build_estimator", fake_build_estimator)
    monkeypatch.setattr(run_mod, "compute_likelihood_summary", fake_compute_likelihood_summary)

    import comp_model_impl.recovery.parameter.sampling as sampling_mod

    monkeypatch.setattr(sampling_mod, "sample_subject_params", fake_sample_subject_params)

    out = run_mod.run_model_recovery(
        config=cfg,
        generator=generator,  # type: ignore[arg-type]
        progress_callback=lambda done, total: progress_calls.append((done, total)),
    )

    assert progress_calls == [(1, 1)]
    assert len(out.fit_table) == 2
    assert len(out.winners) == 1
    assert out.winners.iloc[0]["selected_model"] == "good"

    bad = out.fit_table.loc[out.fit_table["candidate_model"] == "bad"].iloc[0]
    assert bool(bad["success"]) is False
    assert "EXCEPTION" in str(bad["message"])

    run_dir = Path(out.out_dir)
    assert (run_dir / "model_recovery_fit_table.csv").exists()
    assert (run_dir / "model_recovery_winners.csv").exists()
    assert (run_dir / "model_recovery_fit_diagnostics.jsonl").exists()
    assert (run_dir / "model_recovery_manifest.json").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "model_recovery_confusion_matrix.csv").exists()
    assert (run_dir / "model_recovery_recovery_rates.csv").exists()

    fit_rows = (run_dir / "model_recovery_fit_table.csv").read_text(encoding="utf-8").splitlines()
    assert len(fit_rows) == 3

    diag_lines = (run_dir / "model_recovery_fit_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(diag_lines) == 2
    _ = json.loads(diag_lines[0])


def test_run_model_recovery_rejects_bad_plan_extension(tmp_path: Path) -> None:
    """Runner should reject plan paths without JSON/YAML suffix."""
    bad_plan = tmp_path / "plan.txt"
    bad_plan.write_text("{}", encoding="utf-8")

    cfg = ModelRecoveryConfig(
        plan_path=str(bad_plan),
        generating=[
            GeneratingModelSpec(
                name="gen",
                model="QRL",
                sampling=SamplingSpec(mode="fixed", fixed={"alpha": 0.2, "beta": 3.0}),
            )
        ],
        candidates=[
            CandidateModelSpec(
                name="cand",
                model="QRL",
                estimator="TransformedMLESubjectwiseEstimator",
            )
        ],
    )

    with pytest.raises(ValueError, match="plan_path must end with"):
        _ = run_mod.run_model_recovery(config=cfg, generator=_DummyGenerator(StudyData(subjects=[], metadata={})))  # type: ignore[arg-type]


def test_run_model_recovery_requires_nonempty_generating_and_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Runner should validate non-empty generating and candidate lists."""
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    plan = _minimal_plan()
    monkeypatch.setattr(run_mod, "load_study_plan_json", lambda _: plan)

    cfg_no_generating = ModelRecoveryConfig(
        plan_path=str(plan_path),
        generating=[],
        candidates=[
            CandidateModelSpec(
                name="cand",
                model="QRL",
                estimator="TransformedMLESubjectwiseEstimator",
            )
        ],
    )
    with pytest.raises(ValueError, match="config.generating is empty"):
        _ = run_mod.run_model_recovery(
            config=cfg_no_generating,
            generator=_DummyGenerator(StudyData(subjects=[], metadata={})),  # type: ignore[arg-type]
        )

    cfg_no_candidates = ModelRecoveryConfig(
        plan_path=str(plan_path),
        generating=[
            GeneratingModelSpec(
                name="gen",
                model="QRL",
                sampling=SamplingSpec(mode="fixed", fixed={"alpha": 0.2, "beta": 3.0}),
            )
        ],
        candidates=[],
    )
    with pytest.raises(ValueError, match="config.candidates is empty"):
        _ = run_mod.run_model_recovery(
            config=cfg_no_candidates,
            generator=_DummyGenerator(StudyData(subjects=[], metadata={})),  # type: ignore[arg-type]
        )
