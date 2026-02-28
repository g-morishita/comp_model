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
from .social_self_outcome_value_shaping import (
    SocialSelfOutcomeValueShapingModel,
    create_social_self_outcome_value_shaping_model,
)

__all__ = [
    "AsocialQValueSoftmaxConfig",
    "AsocialQValueSoftmaxModel",
    "AsocialStateQValueSoftmaxModel",
    "AsocialStateQValueSoftmaxPerseverationModel",
    "AsocialStateQValueSoftmaxSplitAlphaModel",
    "SocialSelfOutcomeValueShapingModel",
    "UniformRandomPolicyModel",
    "create_asocial_q_value_softmax_model",
    "create_asocial_state_q_value_softmax_model",
    "create_asocial_state_q_value_softmax_perseveration_model",
    "create_asocial_state_q_value_softmax_split_alpha_model",
    "create_social_self_outcome_value_shaping_model",
    "create_uniform_random_policy_model",
]
