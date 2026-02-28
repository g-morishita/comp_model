"""Reference model implementations for the new architecture."""

from .q_learning import (
    AsocialQValueSoftmaxConfig,
    AsocialQValueSoftmaxModel,
    create_asocial_q_value_softmax_model,
)
from .qrl_family import (
    AsocialStateQValueSoftmaxModel,
    AsocialStateQValueSoftmaxPerseverationModel,
    AsocialStateQValueSoftmaxSplitAlphaModel,
    create_asocial_state_q_value_softmax_model,
    create_asocial_state_q_value_softmax_perseveration_model,
    create_asocial_state_q_value_softmax_split_alpha_model,
)
from .random_agent import (
    UniformRandomPolicyModel,
    create_uniform_random_policy_model,
)

__all__ = [
    "AsocialQValueSoftmaxConfig",
    "AsocialQValueSoftmaxModel",
    "AsocialStateQValueSoftmaxModel",
    "AsocialStateQValueSoftmaxPerseverationModel",
    "AsocialStateQValueSoftmaxSplitAlphaModel",
    "UniformRandomPolicyModel",
    "create_asocial_q_value_softmax_model",
    "create_asocial_state_q_value_softmax_model",
    "create_asocial_state_q_value_softmax_perseveration_model",
    "create_asocial_state_q_value_softmax_split_alpha_model",
    "create_uniform_random_policy_model",
]
