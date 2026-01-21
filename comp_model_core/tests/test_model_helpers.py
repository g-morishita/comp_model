import numpy as np
import pytest

from comp_model_core.errors import ParameterValidationError
from comp_model_core.interfaces.model import ComputationalModel
from comp_model_core.params.bounds import Bound
from comp_model_core.params.schema import ParamDef, ParameterSchema
from comp_model_core.spec import TaskSpec, OutcomeType


class ToyModel(ComputationalModel):
    def __init__(self):
        self.alpha = 0.5

    @property
    def param_schema(self) -> ParameterSchema:
        return ParameterSchema(params=(ParamDef("alpha", default=0.5, bound=Bound(0.0, 1.0)),))

    def reset_block(self, *, spec: TaskSpec) -> None:
        return

    def action_probs(self, *, state, spec: TaskSpec) -> np.ndarray:
        return np.array([0.5, 0.5], dtype=float)

    def update(self, *, state, action: int, outcome: float, spec: TaskSpec, info=None) -> None:
        return


def test_model_set_get_params():
    m = ToyModel()
    assert m.get_params() == {"alpha": 0.5}

    m.set_params({"alpha": 0.2})
    assert m.alpha == 0.2


def test_model_set_params_strict_unknown_raises():
    m = ToyModel()
    with pytest.raises(ParameterValidationError):
        m.set_params({"alpha": 0.2, "unknown": 1.0}, strict=True)


def test_model_set_params_bounds_check():
    m = ToyModel()
    with pytest.raises(ParameterValidationError):
        m.set_params({"alpha": 2.0}, check_bounds=True)
