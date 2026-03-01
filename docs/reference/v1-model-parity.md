# Reference: v1 Model Parity

This page documents explicit model-capability parity between the old internal
v1 package and the current clean-slate architecture.

## Scope

- v1 source of truth: `/Users/morishitag/comp_model_v1`
- v2 source of truth: plugin model component IDs in this repository

## Mapping

| v1 symbol | v2 component ID / API |
| --- | --- |
| `QRL` | `asocial_state_q_value_softmax` |
| `QRL_Stay` | `asocial_state_q_value_softmax_perseveration` |
| `UnidentifiableQRL` | `asocial_state_q_value_softmax_split_alpha` |
| `VS` | `social_self_outcome_value_shaping` |
| `Vicarious_VS` | `social_observed_outcome_value_shaping` |
| `Vicarious_VS_Stay` | `social_observed_outcome_value_shaping_perseveration` |
| `Vicarious_RL` | `social_observed_outcome_q` |
| `Vicarious_RL_Stay` | `social_observed_outcome_q_perseveration` |
| `AP_RL_NoStay` | `social_policy_learning_only` |
| `AP_RL_Stay` | `social_policy_learning_only_perseveration` |
| `Vicarious_AP_VS` | `social_policy_reliability_gated_value_shaping` |
| `VicQ_AP_DualW_Stay` | `social_observed_outcome_policy_shared_mix_perseveration` |
| `VicQ_AP_DualW_NoStay` | `social_observed_outcome_policy_shared_mix` |
| `VicQ_AP_IndepDualW` | `social_observed_outcome_policy_independent_mix_perseveration` |
| `Vicarious_DB_Stay` | `social_constant_demo_bias_observed_outcome_q_perseveration` |
| `Vicarious_AP_DB_STAY` | `social_policy_reliability_gated_demo_bias_observed_outcome_q_perseveration` |
| `Vicarious_Dir_DB_Stay` | `social_dirichlet_reliability_gated_demo_bias_observed_outcome_q_perseveration` |
| `ConditionedSharedDeltaModel` | `comp_model.models.ConditionedSharedDeltaModel` |
| `ConditionedSharedDeltaSocialModel` | `comp_model.models.ConditionedSharedDeltaSocialModel` |
| `wrap_model_with_shared_delta_conditions` | `ConditionedSharedDeltaModel` + `SharedDeltaParameterSpec` |

## Verification

Automated checks:

- `tests/test_v1_parity_mapping.py`: validates mapping completeness and v2 registry coverage.

Local cross-repo check (requires local v1 checkout):

```bash
.venv/bin/python scripts/parity/check_v1_model_parity.py \
  --v1-root /Users/morishitag/comp_model_v1
```
