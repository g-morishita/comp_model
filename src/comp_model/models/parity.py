"""Model-name to canonical implementation mapping matrix.

This module tracks capability mapping coverage from source model names into
clean-slate canonical class names and plugin IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ParityStatus = Literal["implemented", "planned"]


@dataclass(frozen=True, slots=True)
class ModelParityEntry:
    """One parity mapping entry from source naming to canonical naming.

    Parameters
    ----------
    source_name : str
        Source model name retained for mapping/reference.
    canonical_component_id : str | None
        Canonical plugin component ID in this repository.
        ``None`` indicates a planned-but-not-yet-implemented mapping.
    canonical_class_name : str | None
        Canonical class name in ``comp_model.models``.
        ``None`` indicates a planned-but-not-yet-implemented mapping.
    status : {"implemented", "planned"}
        Parity implementation status.
    notes : str
        Short explanation of mapping or blocker.
    """

    source_name: str
    canonical_component_id: str | None
    canonical_class_name: str | None
    status: ParityStatus
    notes: str


MODEL_PARITY: tuple[ModelParityEntry, ...] = (
    ModelParityEntry(
        source_name="QRL",
        canonical_component_id="asocial_state_q_value_softmax",
        canonical_class_name="AsocialStateQValueSoftmaxModel",
        status="implemented",
        notes="Asocial state-aware Q-learning with softmax action selection.",
    ),
    ModelParityEntry(
        source_name="QRL_Stay",
        canonical_component_id="asocial_state_q_value_softmax_perseveration",
        canonical_class_name="AsocialStateQValueSoftmaxPerseverationModel",
        status="implemented",
        notes="State-aware Q-learning plus stay/perseveration bias.",
    ),
    ModelParityEntry(
        source_name="UnidentifiableQRL",
        canonical_component_id="asocial_state_q_value_softmax_split_alpha",
        canonical_class_name="AsocialStateQValueSoftmaxSplitAlphaModel",
        status="implemented",
        notes="Split-alpha asocial Q-learning family.",
    ),
    ModelParityEntry(
        source_name="VS",
        canonical_component_id="social_self_outcome_value_shaping",
        canonical_class_name="SocialSelfOutcomeValueShapingModel",
        status="implemented",
        notes="Self-outcome value shaping in social contexts.",
    ),
    ModelParityEntry(
        source_name="Vicarious_RL",
        canonical_component_id="social_observed_outcome_q",
        canonical_class_name="SocialObservedOutcomeQModel",
        status="implemented",
        notes="Observed-outcome social Q-learning.",
    ),
    ModelParityEntry(
        source_name="Vicarious_RL_Stay",
        canonical_component_id="social_observed_outcome_q_perseveration",
        canonical_class_name="SocialObservedOutcomeQPerseverationModel",
        status="implemented",
        notes="Observed-outcome social Q-learning with perseveration.",
    ),
    ModelParityEntry(
        source_name="Vicarious_VS",
        canonical_component_id="social_observed_outcome_value_shaping",
        canonical_class_name="SocialObservedOutcomeValueShapingModel",
        status="implemented",
        notes="Observed-outcome learning plus value shaping.",
    ),
    ModelParityEntry(
        source_name="Vicarious_VS_Stay",
        canonical_component_id="social_observed_outcome_value_shaping_perseveration",
        canonical_class_name="SocialObservedOutcomeValueShapingPerseverationModel",
        status="implemented",
        notes="Observed-outcome value shaping with perseveration.",
    ),
    ModelParityEntry(
        source_name="AP_RL_NoStay",
        canonical_component_id="social_policy_learning_only",
        canonical_class_name="SocialPolicyLearningOnlyModel",
        status="implemented",
        notes="Policy-learning-only social model (no perseveration).",
    ),
    ModelParityEntry(
        source_name="AP_RL_Stay",
        canonical_component_id="social_policy_learning_only_perseveration",
        canonical_class_name="SocialPolicyLearningOnlyPerseverationModel",
        status="implemented",
        notes="Policy-learning-only social model with perseveration.",
    ),
    ModelParityEntry(
        source_name="Vicarious_AP_VS",
        canonical_component_id="social_policy_reliability_gated_value_shaping",
        canonical_class_name="SocialPolicyReliabilityGatedValueShapingModel",
        status="implemented",
        notes="Reliability-gated value shaping from demonstrator policy learning.",
    ),
    ModelParityEntry(
        source_name="Vicarious_AP_DB_STAY",
        canonical_component_id="social_policy_reliability_gated_demo_bias_observed_outcome_q_perseveration",
        canonical_class_name="SocialPolicyReliabilityGatedDemoBiasObservedOutcomeQPerseverationModel",
        status="implemented",
        notes="Reliability-gated demo-bias with observed-outcome Q and perseveration.",
    ),
    ModelParityEntry(
        source_name="Vicarious_Dir_DB_Stay",
        canonical_component_id="social_dirichlet_reliability_gated_demo_bias_observed_outcome_q_perseveration",
        canonical_class_name="SocialDirichletReliabilityGatedDemoBiasObservedOutcomeQPerseverationModel",
        status="implemented",
        notes="Dirichlet reliability-gated demo-bias variant with perseveration.",
    ),
    ModelParityEntry(
        source_name="Vicarious_DB_Stay",
        canonical_component_id="social_constant_demo_bias_observed_outcome_q_perseveration",
        canonical_class_name="SocialConstantDemoBiasObservedOutcomeQPerseverationModel",
        status="implemented",
        notes="Constant demo-bias with observed-outcome Q and perseveration.",
    ),
    ModelParityEntry(
        source_name="VicQ_AP_DualW_NoStay",
        canonical_component_id="social_observed_outcome_policy_shared_mix",
        canonical_class_name="SocialObservedOutcomePolicySharedMixModel",
        status="implemented",
        notes="Observed-outcome and policy-drive shared-weight mix without perseveration.",
    ),
    ModelParityEntry(
        source_name="VicQ_AP_DualW_Stay",
        canonical_component_id="social_observed_outcome_policy_shared_mix_perseveration",
        canonical_class_name="SocialObservedOutcomePolicySharedMixPerseverationModel",
        status="implemented",
        notes="Shared-mix observed-outcome and policy-drive model with perseveration.",
    ),
    ModelParityEntry(
        source_name="VicQ_AP_IndepDualW",
        canonical_component_id="social_observed_outcome_policy_independent_mix_perseveration",
        canonical_class_name="SocialObservedOutcomePolicyIndependentMixPerseverationModel",
        status="implemented",
        notes="Independent decision weights for value and policy drives with perseveration.",
    ),
    ModelParityEntry(
        source_name="ConditionedSharedDeltaModel",
        canonical_component_id=None,
        canonical_class_name="ConditionedSharedDeltaModel",
        status="implemented",
        notes="Generic within-subject shared+delta parameter-tying wrapper.",
    ),
    ModelParityEntry(
        source_name="ConditionedSharedDeltaSocialModel",
        canonical_component_id=None,
        canonical_class_name="ConditionedSharedDeltaSocialModel",
        status="implemented",
        notes="Social within-subject shared+delta parameter-tying wrapper.",
    ),
)


__all__ = ["MODEL_PARITY", "ModelParityEntry"]
