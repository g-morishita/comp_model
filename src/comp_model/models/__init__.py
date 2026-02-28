"""Reference model implementations for the new architecture.

Canonical names are exported first. Deprecated aliases are kept for transition
and scheduled for removal in v0.3.0.
"""

from .q_learning import (
    AsocialQValueSoftmaxConfig,
    AsocialQValueSoftmaxModel,
    QLearningAgent,
    QLearningConfig,
    create_asocial_q_value_softmax_model,
    create_q_learning_agent,
)
from .qrl_family import (
    AsocialStateQValueSoftmaxModel,
    AsocialStateQValueSoftmaxPerseverationModel,
    AsocialStateQValueSoftmaxSplitAlphaModel,
    QRL,
    QRL_Stay,
    UnidentifiableQRL,
    create_asocial_state_q_value_softmax_model,
    create_asocial_state_q_value_softmax_perseveration_model,
    create_asocial_state_q_value_softmax_split_alpha_model,
    create_qrl,
    create_qrl_stay,
    create_unidentifiable_qrl,
)
from .random_agent import (
    RandomAgent,
    UniformRandomPolicyModel,
    create_random_agent,
    create_uniform_random_policy_model,
)

__all__ = [
    # Canonical names
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
    # Deprecated aliases
    "QLearningAgent",
    "QLearningConfig",
    "QRL",
    "QRL_Stay",
    "RandomAgent",
    "UnidentifiableQRL",
    "create_q_learning_agent",
    "create_qrl",
    "create_qrl_stay",
    "create_random_agent",
    "create_unidentifiable_qrl",
]
