"""Reference model implementations for the new architecture."""

from .parity import LegacyModelParityEntry, V1_MODEL_PARITY
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
from .social_observed_outcome_models import (
    SocialObservedOutcomeQModel,
    SocialObservedOutcomeQPerseverationModel,
    SocialObservedOutcomeValueShapingModel,
    SocialObservedOutcomeValueShapingPerseverationModel,
    create_social_observed_outcome_q_model,
    create_social_observed_outcome_q_perseveration_model,
    create_social_observed_outcome_value_shaping_model,
    create_social_observed_outcome_value_shaping_perseveration_model,
)
from .social_policy_mix_models import (
    SocialObservedOutcomePolicyIndependentMixPerseverationModel,
    SocialObservedOutcomePolicySharedMixModel,
    SocialObservedOutcomePolicySharedMixPerseverationModel,
    SocialPolicyLearningOnlyModel,
    SocialPolicyLearningOnlyPerseverationModel,
    create_social_observed_outcome_policy_independent_mix_perseveration_model,
    create_social_observed_outcome_policy_shared_mix_model,
    create_social_observed_outcome_policy_shared_mix_perseveration_model,
    create_social_policy_learning_only_model,
    create_social_policy_learning_only_perseveration_model,
)
from .social_reliability_bias_models import (
    SocialConstantDemoBiasObservedOutcomeQPerseverationModel,
    SocialDirichletReliabilityGatedDemoBiasObservedOutcomeQPerseverationModel,
    SocialPolicyReliabilityGatedDemoBiasObservedOutcomeQPerseverationModel,
    SocialPolicyReliabilityGatedValueShapingModel,
    create_social_constant_demo_bias_observed_outcome_q_perseveration_model,
    create_social_dirichlet_reliability_gated_demo_bias_observed_outcome_q_perseveration_model,
    create_social_policy_reliability_gated_demo_bias_observed_outcome_q_perseveration_model,
    create_social_policy_reliability_gated_value_shaping_model,
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
    "LegacyModelParityEntry",
    "SocialConstantDemoBiasObservedOutcomeQPerseverationModel",
    "SocialDirichletReliabilityGatedDemoBiasObservedOutcomeQPerseverationModel",
    "SocialObservedOutcomePolicyIndependentMixPerseverationModel",
    "SocialObservedOutcomePolicySharedMixModel",
    "SocialObservedOutcomePolicySharedMixPerseverationModel",
    "SocialObservedOutcomeQModel",
    "SocialObservedOutcomeQPerseverationModel",
    "SocialObservedOutcomeValueShapingModel",
    "SocialObservedOutcomeValueShapingPerseverationModel",
    "SocialPolicyLearningOnlyModel",
    "SocialPolicyLearningOnlyPerseverationModel",
    "SocialPolicyReliabilityGatedDemoBiasObservedOutcomeQPerseverationModel",
    "SocialPolicyReliabilityGatedValueShapingModel",
    "SocialSelfOutcomeValueShapingModel",
    "UniformRandomPolicyModel",
    "V1_MODEL_PARITY",
    "create_asocial_q_value_softmax_model",
    "create_asocial_state_q_value_softmax_model",
    "create_asocial_state_q_value_softmax_perseveration_model",
    "create_asocial_state_q_value_softmax_split_alpha_model",
    "create_social_constant_demo_bias_observed_outcome_q_perseveration_model",
    "create_social_dirichlet_reliability_gated_demo_bias_observed_outcome_q_perseveration_model",
    "create_social_observed_outcome_policy_independent_mix_perseveration_model",
    "create_social_observed_outcome_policy_shared_mix_model",
    "create_social_observed_outcome_policy_shared_mix_perseveration_model",
    "create_social_observed_outcome_q_model",
    "create_social_observed_outcome_q_perseveration_model",
    "create_social_observed_outcome_value_shaping_model",
    "create_social_observed_outcome_value_shaping_perseveration_model",
    "create_social_policy_learning_only_model",
    "create_social_policy_learning_only_perseveration_model",
    "create_social_policy_reliability_gated_demo_bias_observed_outcome_q_perseveration_model",
    "create_social_policy_reliability_gated_value_shaping_model",
    "create_social_self_outcome_value_shaping_model",
    "create_uniform_random_policy_model",
]
