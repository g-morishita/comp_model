import numpy as np
import pytest

from comp_model_core.data.types import StudyData
from comp_model_core.interfaces.bandit import BanditEnv, EnvStep
from comp_model_core.interfaces.block_runner import BlockRunner, StepResult
from comp_model_core.interfaces.demonstrator import Demonstrator
from comp_model_core.interfaces.estimator import Estimator, FitResult
from comp_model_core.interfaces.generator import Generator
from comp_model_core.interfaces.model import ComputationalModel
from comp_model_core.params.schema import ParamDef, ParameterSchema
from comp_model_core.spec import EnvironmentSpec, OutcomeType, TrialSpec, OutcomeObservationSpec, OutcomeObservationKind


def test_abstract_interfaces_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BanditEnv()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        BlockRunner()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        Generator()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        Estimator()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        Demonstrator()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        ComputationalModel()  # type: ignore[abstract]


class MinimalEnv(BanditEnv):
    def __init__(self) -> None:
        self._spec = EnvironmentSpec(n_actions=2, outcome_type=OutcomeType.BINARY)
        self._state = 0

    @property
    def spec(self) -> EnvironmentSpec:
        return self._spec

    def reset(self, *, rng: np.random.Generator):
        self._state = 0
        return self._state

    def step(self, *, action: int, rng: np.random.Generator) -> EnvStep:
        self._state += 1
        return EnvStep(outcome=float(action))

    def get_state(self):
        return self._state


class MinimalRunner(BlockRunner):
    def __init__(self) -> None:
        self._spec = EnvironmentSpec(n_actions=3, outcome_type=OutcomeType.BINARY)
        self._t = 0

    @property
    def spec(self) -> EnvironmentSpec:
        return self._spec

    def reset(self, *, rng: np.random.Generator):
        self._t = 0
        return 0

    def step(self, *, t: int, action: int, rng: np.random.Generator) -> StepResult:
        self._t = t + 1
        return StepResult(outcome=float(action), observed_outcome=float(action))

    def get_state(self):
        return 0

    def trial_spec(self, *, t: int) -> TrialSpec:
        # Trial 0 is forced choice; other trials have all actions.
        if t == 0:
            return TrialSpec(
                self_outcome=OutcomeObservationSpec(kind=OutcomeObservationKind.VERIDICAL),
                available_actions=(1, 2),
            )
        return TrialSpec(self_outcome=OutcomeObservationSpec(kind=OutcomeObservationKind.VERIDICAL))


def test_block_runner_available_actions_helper():
    r = MinimalRunner()
    assert r.available_actions(t=0) == [1, 2]
    assert r.available_actions(t=1) == [0, 1, 2]


class MinimalEstimator(Estimator):
    def __init__(self, model: ComputationalModel):
        self.model = model

    def supports(self, study: StudyData) -> bool:
        return True

    def fit(self, *, study: StudyData, rng: np.random.Generator) -> FitResult:
        return FitResult(params_hat={})


class MinimalModel(ComputationalModel):
    def __init__(self):
        self.alpha = 0.1

    @property
    def param_schema(self) -> ParameterSchema:
        return ParameterSchema(params=(ParamDef("alpha", default=0.1),))

    def reset_block(self, *, spec: EnvironmentSpec) -> None:
        return

    def action_probs(self, *, state, spec: EnvironmentSpec) -> np.ndarray:
        return np.array([0.5, 0.5], dtype=float)

    def update(self, *, state, action: int, outcome: float | None, spec: EnvironmentSpec, info=None, rng=None) -> None:
        return


def test_estimator_fit_result_constructible():
    est = MinimalEstimator(MinimalModel())
    res = est.fit(study=StudyData(subjects=[]), rng=np.random.default_rng(0))
    assert isinstance(res, FitResult)