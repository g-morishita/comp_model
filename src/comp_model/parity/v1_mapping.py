"""v1-to-v2 model capability mapping.

This file records how every public v1 model name maps to the clean-slate v2
architecture. The mapping is intentionally explicit so parity audits do not
depend on historical memory.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class V1ModelParityEntry:
    """One v1 model-name to v2 capability mapping row.

    Parameters
    ----------
    legacy_name : str
        Public v1 model symbol exported from `comp_model_impl.models`.
    replacement_component_id : str | None
        v2 plugin model component ID providing equivalent behavior. `None` is
        used when the v1 symbol was a wrapper/helper rather than a registry
        model component.
    replacement_api : str
        v2 API path that users should use for this capability.
    notes : str, optional
        Short migration note.
    """

    legacy_name: str
    replacement_component_id: str | None
    replacement_api: str
    notes: str = ""


V1_MODEL_PARITY_MAP: tuple[V1ModelParityEntry, ...] = (
    V1ModelParityEntry(
        legacy_name="QRL",
        replacement_component_id="asocial_state_q_value_softmax",
        replacement_api="comp_model.models.AsocialStateQValueSoftmaxModel",
    ),
    V1ModelParityEntry(
        legacy_name="QRL_Stay",
        replacement_component_id="asocial_state_q_value_softmax_perseveration",
        replacement_api="comp_model.models.AsocialStateQValueSoftmaxPerseverationModel",
    ),
    V1ModelParityEntry(
        legacy_name="UnidentifiableQRL",
        replacement_component_id="asocial_state_q_value_softmax_split_alpha",
        replacement_api="comp_model.models.AsocialStateQValueSoftmaxSplitAlphaModel",
    ),
    V1ModelParityEntry(
        legacy_name="VS",
        replacement_component_id="social_self_outcome_value_shaping",
        replacement_api="comp_model.models.SocialSelfOutcomeValueShapingModel",
    ),
    V1ModelParityEntry(
        legacy_name="Vicarious_VS",
        replacement_component_id="social_observed_outcome_value_shaping",
        replacement_api="comp_model.models.SocialObservedOutcomeValueShapingModel",
    ),
    V1ModelParityEntry(
        legacy_name="Vicarious_VS_Stay",
        replacement_component_id="social_observed_outcome_value_shaping_perseveration",
        replacement_api="comp_model.models.SocialObservedOutcomeValueShapingPerseverationModel",
    ),
    V1ModelParityEntry(
        legacy_name="Vicarious_RL",
        replacement_component_id="social_observed_outcome_q",
        replacement_api="comp_model.models.SocialObservedOutcomeQModel",
    ),
    V1ModelParityEntry(
        legacy_name="Vicarious_RL_Stay",
        replacement_component_id="social_observed_outcome_q_perseveration",
        replacement_api="comp_model.models.SocialObservedOutcomeQPerseverationModel",
    ),
    V1ModelParityEntry(
        legacy_name="AP_RL_NoStay",
        replacement_component_id="social_policy_learning_only",
        replacement_api="comp_model.models.SocialPolicyLearningOnlyModel",
    ),
    V1ModelParityEntry(
        legacy_name="AP_RL_Stay",
        replacement_component_id="social_policy_learning_only_perseveration",
        replacement_api="comp_model.models.SocialPolicyLearningOnlyPerseverationModel",
    ),
    V1ModelParityEntry(
        legacy_name="Vicarious_AP_VS",
        replacement_component_id="social_policy_reliability_gated_value_shaping",
        replacement_api="comp_model.models.SocialPolicyReliabilityGatedValueShapingModel",
    ),
    V1ModelParityEntry(
        legacy_name="VicQ_AP_DualW_Stay",
        replacement_component_id="social_observed_outcome_policy_shared_mix_perseveration",
        replacement_api="comp_model.models.SocialObservedOutcomePolicySharedMixPerseverationModel",
    ),
    V1ModelParityEntry(
        legacy_name="VicQ_AP_DualW_NoStay",
        replacement_component_id="social_observed_outcome_policy_shared_mix",
        replacement_api="comp_model.models.SocialObservedOutcomePolicySharedMixModel",
    ),
    V1ModelParityEntry(
        legacy_name="VicQ_AP_IndepDualW",
        replacement_component_id="social_observed_outcome_policy_independent_mix_perseveration",
        replacement_api="comp_model.models.SocialObservedOutcomePolicyIndependentMixPerseverationModel",
    ),
    V1ModelParityEntry(
        legacy_name="Vicarious_DB_Stay",
        replacement_component_id="social_constant_demo_bias_observed_outcome_q_perseveration",
        replacement_api="comp_model.models.SocialConstantDemoBiasObservedOutcomeQPerseverationModel",
    ),
    V1ModelParityEntry(
        legacy_name="Vicarious_AP_DB_STAY",
        replacement_component_id="social_policy_reliability_gated_demo_bias_observed_outcome_q_perseveration",
        replacement_api="comp_model.models.SocialPolicyReliabilityGatedDemoBiasObservedOutcomeQPerseverationModel",
    ),
    V1ModelParityEntry(
        legacy_name="Vicarious_Dir_DB_Stay",
        replacement_component_id="social_dirichlet_reliability_gated_demo_bias_observed_outcome_q_perseveration",
        replacement_api="comp_model.models.SocialDirichletReliabilityGatedDemoBiasObservedOutcomeQPerseverationModel",
    ),
    V1ModelParityEntry(
        legacy_name="ConditionedSharedDeltaModel",
        replacement_component_id=None,
        replacement_api="comp_model.models.ConditionedSharedDeltaModel",
        notes="Wrapper capability preserved as direct class API (not registry component).",
    ),
    V1ModelParityEntry(
        legacy_name="ConditionedSharedDeltaSocialModel",
        replacement_component_id=None,
        replacement_api="comp_model.models.ConditionedSharedDeltaSocialModel",
        notes="Wrapper capability preserved as direct class API (not registry component).",
    ),
    V1ModelParityEntry(
        legacy_name="wrap_model_with_shared_delta_conditions",
        replacement_component_id=None,
        replacement_api=(
            "comp_model.models.ConditionedSharedDeltaModel + "
            "comp_model.models.SharedDeltaParameterSpec"
        ),
        notes="Replaced by explicit shared+delta wrapper construction.",
    ),
)


__all__ = ["V1ModelParityEntry", "V1_MODEL_PARITY_MAP"]
