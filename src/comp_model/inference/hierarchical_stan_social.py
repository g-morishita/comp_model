"""Social-model Stan adapter utilities for hierarchical within-subject NUTS.

This module provides:

- Supported social component IDs for the Stan backend.
- A generic social Stan program covering all social model families.
- Input-building helpers that convert subject block data into Stan arrays while
  preserving trial-event timing (subject and demonstrator decision rows).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from comp_model.core.data import BlockData, SubjectData, TrialDecision, trial_decisions_from_trace

_TRANSFORM_CODE_BY_KIND: dict[str, int] = {
    "identity": 0,
    "unit_interval_logit": 1,
    "positive_log": 2,
}

_SOCIAL_PARAM_CODE_BY_NAME: dict[str, int] = {
    "alpha_self": 1,
    "alpha_observed": 2,
    "alpha_social": 3,
    "alpha_policy": 4,
    "alpha_social_base": 5,
    "beta": 6,
    "beta_q": 7,
    "beta_policy": 8,
    "kappa": 9,
    "mix_weight": 10,
    "demo_bias": 11,
    "demo_bias_rel": 12,
    "demo_dirichlet_prior": 13,
    "initial_value": 14,
    "pseudo_reward": 15,
}

_SOCIAL_FIXED_DEFAULTS: dict[str, float] = {
    "alpha_self": 0.2,
    "alpha_observed": 0.2,
    "alpha_social": 0.2,
    "alpha_policy": 0.2,
    "alpha_social_base": 0.2,
    "beta": 3.0,
    "beta_q": 3.0,
    "beta_policy": 3.0,
    "kappa": 0.0,
    "mix_weight": 0.5,
    "demo_bias": 1.0,
    "demo_bias_rel": 1.0,
    "demo_dirichlet_prior": 1.0,
    "initial_value": 0.0,
    "pseudo_reward": 1.0,
}

_UNIT_INTERVAL_PARAMS = {
    "alpha_self",
    "alpha_observed",
    "alpha_social",
    "alpha_policy",
    "alpha_social_base",
    "mix_weight",
}

_NONNEGATIVE_PARAMS = {
    "beta",
    "beta_q",
    "beta_policy",
}

_POSITIVE_PARAMS = {
    "demo_dirichlet_prior",
}


@dataclass(frozen=True, slots=True)
class _SocialStanSpec:
    """Configuration for one social model family in the Stan backend."""

    component_id: str
    supported_params: tuple[str, ...]
    defaults: Mapping[str, float]
    flag_subject_outcome_learning: int = 0
    flag_demo_outcome_learning: int = 0
    flag_social_shaping_on_demo: int = 0
    flag_social_shaping_from_subject_observation: int = 0
    reliability_for_social_shaping: int = 0
    reliability_for_demo_bias: int = 0
    flag_policy_learning: int = 0
    flag_last_choice: int = 0
    flag_recent_demo_choice: int = 0
    flag_use_shared_mix: int = 0
    flag_use_independent_mix: int = 0
    flag_include_q_in_decision: int = 1
    flag_include_policy_in_decision: int = 0

    def flags(self) -> dict[str, int]:
        """Return integer Stan flags for this model spec."""

        return {
            "flag_subject_outcome_learning": int(self.flag_subject_outcome_learning),
            "flag_demo_outcome_learning": int(self.flag_demo_outcome_learning),
            "flag_social_shaping_on_demo": int(self.flag_social_shaping_on_demo),
            "flag_social_shaping_from_subject_observation": int(
                self.flag_social_shaping_from_subject_observation
            ),
            "reliability_for_social_shaping": int(self.reliability_for_social_shaping),
            "reliability_for_demo_bias": int(self.reliability_for_demo_bias),
            "flag_policy_learning": int(self.flag_policy_learning),
            "flag_last_choice": int(self.flag_last_choice),
            "flag_recent_demo_choice": int(self.flag_recent_demo_choice),
            "flag_use_shared_mix": int(self.flag_use_shared_mix),
            "flag_use_independent_mix": int(self.flag_use_independent_mix),
            "flag_include_q_in_decision": int(self.flag_include_q_in_decision),
            "flag_include_policy_in_decision": int(self.flag_include_policy_in_decision),
        }


def _build_social_specs() -> dict[str, _SocialStanSpec]:
    """Create social component mapping used by the Stan adapter."""

    specs = [
        _SocialStanSpec(
            component_id="social_observed_outcome_q",
            supported_params=("alpha_observed", "beta", "initial_value"),
            defaults={"alpha_observed": 0.2, "beta": 3.0, "initial_value": 0.0},
            flag_demo_outcome_learning=1,
            flag_include_q_in_decision=1,
        ),
        _SocialStanSpec(
            component_id="social_observed_outcome_q_perseveration",
            supported_params=("alpha_observed", "beta", "kappa", "initial_value"),
            defaults={"alpha_observed": 0.2, "beta": 3.0, "kappa": 0.0, "initial_value": 0.0},
            flag_demo_outcome_learning=1,
            flag_last_choice=1,
            flag_include_q_in_decision=1,
        ),
        _SocialStanSpec(
            component_id="social_observed_outcome_value_shaping",
            supported_params=("alpha_observed", "alpha_social", "beta", "pseudo_reward", "initial_value"),
            defaults={
                "alpha_observed": 0.2,
                "alpha_social": 0.2,
                "beta": 3.0,
                "pseudo_reward": 1.0,
                "initial_value": 0.0,
            },
            flag_demo_outcome_learning=1,
            flag_social_shaping_on_demo=1,
            flag_include_q_in_decision=1,
        ),
        _SocialStanSpec(
            component_id="social_observed_outcome_value_shaping_perseveration",
            supported_params=("alpha_observed", "alpha_social", "beta", "kappa", "pseudo_reward", "initial_value"),
            defaults={
                "alpha_observed": 0.2,
                "alpha_social": 0.2,
                "beta": 3.0,
                "kappa": 0.0,
                "pseudo_reward": 1.0,
                "initial_value": 0.0,
            },
            flag_demo_outcome_learning=1,
            flag_social_shaping_on_demo=1,
            flag_last_choice=1,
            flag_include_q_in_decision=1,
        ),
        _SocialStanSpec(
            component_id="social_policy_reliability_gated_value_shaping",
            supported_params=(
                "alpha_observed",
                "alpha_social_base",
                "alpha_policy",
                "beta",
                "kappa",
                "pseudo_reward",
                "initial_value",
            ),
            defaults={
                "alpha_observed": 0.2,
                "alpha_social_base": 0.2,
                "alpha_policy": 0.2,
                "beta": 3.0,
                "kappa": 0.0,
                "pseudo_reward": 1.0,
                "initial_value": 0.0,
            },
            flag_demo_outcome_learning=1,
            flag_social_shaping_on_demo=1,
            reliability_for_social_shaping=1,
            flag_policy_learning=1,
            flag_last_choice=1,
            flag_include_q_in_decision=1,
        ),
        _SocialStanSpec(
            component_id="social_constant_demo_bias_observed_outcome_q_perseveration",
            supported_params=("alpha_observed", "demo_bias", "beta", "kappa", "initial_value"),
            defaults={
                "alpha_observed": 0.2,
                "demo_bias": 1.0,
                "beta": 3.0,
                "kappa": 0.0,
                "initial_value": 0.0,
            },
            flag_demo_outcome_learning=1,
            flag_last_choice=1,
            flag_recent_demo_choice=1,
            flag_include_q_in_decision=1,
        ),
        _SocialStanSpec(
            component_id="social_policy_reliability_gated_demo_bias_observed_outcome_q_perseveration",
            supported_params=("alpha_observed", "alpha_policy", "demo_bias_rel", "beta", "kappa", "initial_value"),
            defaults={
                "alpha_observed": 0.2,
                "alpha_policy": 0.2,
                "demo_bias_rel": 1.0,
                "beta": 3.0,
                "kappa": 0.0,
                "initial_value": 0.0,
            },
            flag_demo_outcome_learning=1,
            flag_policy_learning=1,
            flag_last_choice=1,
            flag_recent_demo_choice=1,
            reliability_for_demo_bias=1,
            flag_include_q_in_decision=1,
        ),
        _SocialStanSpec(
            component_id="social_dirichlet_reliability_gated_demo_bias_observed_outcome_q_perseveration",
            supported_params=("alpha_observed", "demo_bias_rel", "beta", "kappa", "demo_dirichlet_prior", "initial_value"),
            defaults={
                "alpha_observed": 0.2,
                "demo_bias_rel": 1.0,
                "beta": 3.0,
                "kappa": 0.0,
                "demo_dirichlet_prior": 1.0,
                "initial_value": 0.0,
            },
            flag_demo_outcome_learning=1,
            flag_last_choice=1,
            flag_recent_demo_choice=1,
            reliability_for_demo_bias=2,
            flag_include_q_in_decision=1,
        ),
        _SocialStanSpec(
            component_id="social_observed_outcome_policy_shared_mix",
            supported_params=("alpha_observed", "alpha_policy", "beta", "mix_weight", "initial_value"),
            defaults={
                "alpha_observed": 0.2,
                "alpha_policy": 0.2,
                "beta": 6.0,
                "mix_weight": 0.5,
                "initial_value": 0.0,
            },
            flag_demo_outcome_learning=1,
            flag_policy_learning=1,
            flag_use_shared_mix=1,
            flag_include_q_in_decision=1,
            flag_include_policy_in_decision=1,
        ),
        _SocialStanSpec(
            component_id="social_observed_outcome_policy_shared_mix_perseveration",
            supported_params=("alpha_observed", "alpha_policy", "beta", "mix_weight", "kappa", "initial_value"),
            defaults={
                "alpha_observed": 0.2,
                "alpha_policy": 0.2,
                "beta": 6.0,
                "mix_weight": 0.5,
                "kappa": 0.0,
                "initial_value": 0.0,
            },
            flag_demo_outcome_learning=1,
            flag_policy_learning=1,
            flag_last_choice=1,
            flag_use_shared_mix=1,
            flag_include_q_in_decision=1,
            flag_include_policy_in_decision=1,
        ),
        _SocialStanSpec(
            component_id="social_observed_outcome_policy_independent_mix_perseveration",
            supported_params=("alpha_observed", "alpha_policy", "beta_q", "beta_policy", "kappa", "initial_value"),
            defaults={
                "alpha_observed": 0.2,
                "alpha_policy": 0.2,
                "beta_q": 3.0,
                "beta_policy": 3.0,
                "kappa": 0.0,
                "initial_value": 0.0,
            },
            flag_demo_outcome_learning=1,
            flag_policy_learning=1,
            flag_last_choice=1,
            flag_use_independent_mix=1,
            flag_include_q_in_decision=1,
            flag_include_policy_in_decision=1,
        ),
        _SocialStanSpec(
            component_id="social_policy_learning_only",
            supported_params=("alpha_policy", "beta"),
            defaults={"alpha_policy": 0.2, "beta": 6.0},
            flag_policy_learning=1,
            flag_include_q_in_decision=0,
            flag_include_policy_in_decision=1,
        ),
        _SocialStanSpec(
            component_id="social_policy_learning_only_perseveration",
            supported_params=("alpha_policy", "beta", "kappa"),
            defaults={"alpha_policy": 0.2, "beta": 6.0, "kappa": 0.0},
            flag_policy_learning=1,
            flag_last_choice=1,
            flag_include_q_in_decision=0,
            flag_include_policy_in_decision=1,
        ),
        _SocialStanSpec(
            component_id="social_self_outcome_value_shaping",
            supported_params=("alpha_self", "alpha_social", "beta", "kappa", "pseudo_reward", "initial_value"),
            defaults={
                "alpha_self": 0.2,
                "alpha_social": 0.2,
                "beta": 3.0,
                "kappa": 0.0,
                "pseudo_reward": 1.0,
                "initial_value": 0.0,
            },
            flag_subject_outcome_learning=1,
            flag_social_shaping_from_subject_observation=1,
            flag_last_choice=1,
            flag_include_q_in_decision=1,
        ),
    ]
    return {spec.component_id: spec for spec in specs}


_SOCIAL_STAN_SPECS = _build_social_specs()

_STAN_WITHIN_SUBJECT_DIR = Path(__file__).with_name("stan") / "within_subject"
_SOCIAL_STAN_FILENAME = "social_generic.stan"


def load_social_stan_code() -> str:
    """Load the shared social Stan program source from file."""

    path = _STAN_WITHIN_SUBJECT_DIR / _SOCIAL_STAN_FILENAME
    if not path.exists():
        raise RuntimeError(f"Stan program file is missing: {path}")
    return path.read_text(encoding="utf-8")



def social_supported_component_ids() -> tuple[str, ...]:
    """Return social model component IDs supported by Stan hierarchical NUTS."""

    return tuple(sorted(_SOCIAL_STAN_SPECS))


def social_cache_tag(component_id: str) -> str:
    """Return compile-cache tag for one social model component."""

    if component_id not in _SOCIAL_STAN_SPECS:
        raise ValueError(
            f"unsupported social component_id {component_id!r}; "
            f"supported {sorted(_SOCIAL_STAN_SPECS)}"
        )
    return f"hierarchical_{component_id}"


def build_social_subject_inputs(
    *,
    subject: SubjectData,
    component_id: str,
    parameter_names: Sequence[str],
    transform_kinds: Mapping[str, str] | None,
    model_kwargs: Mapping[str, Any] | None,
    initial_group_location: Mapping[str, float] | None,
    initial_group_scale: Mapping[str, float] | None,
    initial_block_params: Sequence[Mapping[str, float]] | None,
    mu_prior_mean: float | Mapping[str, float],
    mu_prior_std: float | Mapping[str, float],
    log_sigma_prior_mean: float | Mapping[str, float],
    log_sigma_prior_std: float | Mapping[str, float],
) -> tuple[dict[str, Any], tuple[str, ...], int]:
    """Build Stan data arrays for one social model and subject."""

    spec = _SOCIAL_STAN_SPECS.get(component_id)
    if spec is None:
        raise ValueError(
            f"unsupported social component_id {component_id!r}; "
            f"supported {sorted(_SOCIAL_STAN_SPECS)}"
        )

    names = tuple(str(name) for name in parameter_names)
    if len(names) == 0:
        raise ValueError("parameter_names must include at least one parameter")
    if len(set(names)) != len(names):
        raise ValueError("parameter_names must be unique")

    unknown = [name for name in names if name not in spec.supported_params]
    if unknown:
        raise ValueError(
            f"unsupported parameter_names for {component_id!r}: {unknown!r}; "
            f"supported {sorted(spec.supported_params)}"
        )

    resolved_transform_kinds = _resolve_transform_kinds(
        parameter_names=names,
        transform_kinds=transform_kinds,
    )
    mu_prior_mean_vec = _resolve_prior_vector(
        mu_prior_mean,
        parameter_names=names,
        default_value=0.0,
        field_name="mu_prior_mean",
    )
    mu_prior_std_vec = _resolve_prior_vector(
        mu_prior_std,
        parameter_names=names,
        default_value=2.0,
        field_name="mu_prior_std",
        must_be_positive=True,
    )
    log_sigma_prior_mean_vec = _resolve_prior_vector(
        log_sigma_prior_mean,
        parameter_names=names,
        default_value=-1.0,
        field_name="log_sigma_prior_mean",
    )
    log_sigma_prior_std_vec = _resolve_prior_vector(
        log_sigma_prior_std,
        parameter_names=names,
        default_value=1.0,
        field_name="log_sigma_prior_std",
        must_be_positive=True,
    )

    fixed = dict(_SOCIAL_FIXED_DEFAULTS)
    for key, value in spec.defaults.items():
        fixed[key] = float(value)

    provided_kwargs = dict(model_kwargs or {})
    for key, value in provided_kwargs.items():
        if key not in spec.supported_params:
            raise ValueError(
                f"unsupported model kwarg {key!r} for component {component_id!r}; "
                f"supported keys are {sorted(spec.supported_params)}"
            )
        if key in names:
            raise ValueError(
                f"model kwarg {key!r} conflicts with sampled parameter_names; remove one source"
            )
        fixed[key] = float(value)

    for parameter_name in spec.supported_params:
        _validate_fixed_social_parameter(parameter_name, float(fixed[parameter_name]))

    block_rows = tuple(_rows_for_block(block) for block in subject.blocks)
    if initial_block_params is not None and len(initial_block_params) != len(block_rows):
        raise ValueError("initial_block_params must match number of subject blocks")

    has_subject = False
    action_to_index: dict[Any, int] = {}
    state_to_index: dict[int, int] = {}

    for rows in block_rows:
        for row in rows:
            if row.actor_id == "subject":
                has_subject = True
            if row.actor_id not in {"subject", "demonstrator"}:
                raise ValueError(
                    f"unsupported actor_id {row.actor_id!r} in social Stan backend; "
                    "supported actor IDs are 'subject' and 'demonstrator'"
                )

            if row.available_actions is None:
                raise ValueError("trial decision requires available_actions for Stan backend")
            if row.action is None:
                raise ValueError("trial decision requires action for Stan backend")

            state = _state_from_observation(row.observation)
            if state not in state_to_index:
                state_to_index[state] = len(state_to_index) + 1

            for action in row.available_actions:
                if action not in action_to_index:
                    action_to_index[action] = len(action_to_index) + 1
            if row.action not in action_to_index:
                action_to_index[row.action] = len(action_to_index) + 1

    if not has_subject:
        raise ValueError("social Stan backend requires at least one subject decision per subject")
    if not action_to_index:
        raise ValueError("no available actions found for Stan backend")
    if not state_to_index:
        raise ValueError("no states found for Stan backend")

    n_blocks = len(block_rows)
    block_lengths = [len(rows) for rows in block_rows]
    if any(length <= 0 for length in block_lengths):
        raise ValueError("each block must include at least one decision")

    t_max = max(block_lengths)
    n_actions = len(action_to_index)
    n_states = len(state_to_index)

    state_idx = np.ones((n_blocks, t_max), dtype=int)
    action_idx = np.ones((n_blocks, t_max), dtype=int)
    actor_code = np.ones((n_blocks, t_max), dtype=int)
    reward = np.zeros((n_blocks, t_max), dtype=float)
    has_reward = np.zeros((n_blocks, t_max), dtype=int)
    is_available = np.zeros((n_blocks, t_max, n_actions), dtype=int)
    obs_demo_action_idx = np.zeros((n_blocks, t_max), dtype=int)

    for block_index, rows in enumerate(block_rows):
        for decision_index, row in enumerate(rows):
            state_value = _state_from_observation(row.observation)
            if row.available_actions is None or row.action is None:
                raise ValueError("trial decision is missing actions for Stan backend")

            state_idx[block_index, decision_index] = state_to_index[state_value]
            action_idx[block_index, decision_index] = action_to_index[row.action]
            actor_code[block_index, decision_index] = _actor_code(row.actor_id)

            row_reward, row_has_reward = _reward_from_row_optional(row)
            reward[block_index, decision_index] = row_reward
            has_reward[block_index, decision_index] = row_has_reward

            if len(row.available_actions) == 0:
                raise ValueError("available_actions must not be empty")

            for available_action in row.available_actions:
                is_available[block_index, decision_index, action_to_index[available_action] - 1] = 1

            chosen_position = action_to_index[row.action] - 1
            if is_available[block_index, decision_index, chosen_position] != 1:
                raise ValueError("observed action must be present in available_actions")

            demo_action = _observed_demo_action(row.observation)
            if demo_action is not None:
                if demo_action not in action_to_index:
                    raise ValueError(
                        f"observation.demonstrator_action={demo_action!r} is unknown "
                        "to the current action space"
                    )
                demo_position = action_to_index[demo_action] - 1
                if is_available[block_index, decision_index, demo_position] != 1:
                    raise ValueError(
                        "observation.demonstrator_action must be one of available_actions"
                    )
                obs_demo_action_idx[block_index, decision_index] = action_to_index[demo_action]

    group_loc_init, group_log_scale_init, block_z_init = _build_latent_initialization(
        parameter_names=names,
        transform_kinds=resolved_transform_kinds,
        n_blocks=n_blocks,
        initial_group_location=initial_group_location,
        initial_group_scale=initial_group_scale,
        initial_block_params=initial_block_params,
    )

    stan_data: dict[str, Any] = {
        "B": n_blocks,
        "K": len(names),
        "S": n_states,
        "A": n_actions,
        "T_max": int(t_max),
        "T": [int(value) for value in block_lengths],
        "state_idx": state_idx.tolist(),
        "action_idx": action_idx.tolist(),
        "actor_code": actor_code.tolist(),
        "reward": reward.tolist(),
        "has_reward": has_reward.tolist(),
        "is_available": is_available.tolist(),
        "obs_demo_action_idx": obs_demo_action_idx.tolist(),
        "param_codes": [_SOCIAL_PARAM_CODE_BY_NAME[name] for name in names],
        "transform_codes": [_TRANSFORM_CODE_BY_KIND[resolved_transform_kinds[name]] for name in names],
        "fixed_alpha_self": float(fixed["alpha_self"]),
        "fixed_alpha_observed": float(fixed["alpha_observed"]),
        "fixed_alpha_social": float(fixed["alpha_social"]),
        "fixed_alpha_policy": float(fixed["alpha_policy"]),
        "fixed_alpha_social_base": float(fixed["alpha_social_base"]),
        "fixed_beta": float(fixed["beta"]),
        "fixed_beta_q": float(fixed["beta_q"]),
        "fixed_beta_policy": float(fixed["beta_policy"]),
        "fixed_kappa": float(fixed["kappa"]),
        "fixed_mix_weight": float(fixed["mix_weight"]),
        "fixed_demo_bias": float(fixed["demo_bias"]),
        "fixed_demo_bias_rel": float(fixed["demo_bias_rel"]),
        "fixed_demo_dirichlet_prior": float(fixed["demo_dirichlet_prior"]),
        "fixed_initial_value": float(fixed["initial_value"]),
        "fixed_pseudo_reward": float(fixed["pseudo_reward"]),
        "mu_prior_mean": mu_prior_mean_vec.tolist(),
        "mu_prior_std": mu_prior_std_vec.tolist(),
        "log_sigma_prior_mean": log_sigma_prior_mean_vec.tolist(),
        "log_sigma_prior_std": log_sigma_prior_std_vec.tolist(),
        "group_loc_init": group_loc_init.tolist(),
        "group_log_scale_init": group_log_scale_init.tolist(),
        "block_z_init": block_z_init.tolist(),
    }
    stan_data.update(spec.flags())
    return stan_data, names, n_blocks


def _rows_for_block(block: BlockData) -> tuple[TrialDecision, ...]:
    """Return all block decision rows in chronological order."""

    if block.trials:
        return tuple(block.trials)
    if block.event_trace is not None:
        return trial_decisions_from_trace(block.event_trace)
    raise ValueError("block has neither trials nor event_trace")


def _state_from_observation(observation: Any) -> int:
    """Extract integer state index from observation payload."""

    if isinstance(observation, Mapping) and "state" in observation:
        return int(observation["state"])
    return 0


def _observed_demo_action(observation: Any) -> Any | None:
    """Extract optional demonstrator action embedded in subject observation."""

    if isinstance(observation, Mapping) and "demonstrator_action" in observation:
        return observation["demonstrator_action"]
    return None


def _reward_from_row_optional(row: TrialDecision) -> tuple[float, int]:
    """Extract reward when available, returning ``(reward, has_reward)``."""

    if row.reward is not None:
        return float(row.reward), 1

    if isinstance(row.outcome, Mapping) and "reward" in row.outcome:
        return float(row.outcome["reward"]), 1

    if row.outcome is not None and hasattr(row.outcome, "reward"):
        return float(getattr(row.outcome, "reward")), 1

    return 0.0, 0


def _actor_code(actor_id: str) -> int:
    """Map actor ID into Stan actor code."""

    if actor_id == "subject":
        return 1
    if actor_id == "demonstrator":
        return 2
    raise ValueError(f"unsupported actor_id {actor_id!r}")


def _validate_fixed_social_parameter(parameter_name: str, value: float) -> None:
    """Validate fixed social parameter against known numeric constraints."""

    if parameter_name in _UNIT_INTERVAL_PARAMS and (value < 0.0 or value > 1.0):
        raise ValueError(f"fixed {parameter_name} must be in [0, 1]")

    if parameter_name in _NONNEGATIVE_PARAMS and value < 0.0:
        raise ValueError(f"fixed {parameter_name} must be >= 0")

    if parameter_name in _POSITIVE_PARAMS and value <= 0.0:
        raise ValueError(f"fixed {parameter_name} must be > 0")


def _resolve_transform_kinds(
    *,
    parameter_names: tuple[str, ...],
    transform_kinds: Mapping[str, str] | None,
) -> dict[str, str]:
    """Resolve transform kinds for sampled parameters."""

    raw_mapping = dict(transform_kinds or {})
    out: dict[str, str] = {}
    for name in parameter_names:
        raw_kind = raw_mapping.get(name, _default_transform_kind(name))
        kind = str(raw_kind).strip()
        if kind not in _TRANSFORM_CODE_BY_KIND:
            raise ValueError(
                f"unsupported transform kind {kind!r} for parameter {name!r}; "
                f"expected one of {sorted(_TRANSFORM_CODE_BY_KIND)}"
            )
        out[name] = kind
    return out


def _resolve_prior_vector(
    spec: float | Mapping[str, float],
    *,
    parameter_names: tuple[str, ...],
    default_value: float,
    field_name: str,
    must_be_positive: bool = False,
) -> np.ndarray:
    """Resolve scalar/mapping prior input into parameter-aligned vector."""

    values = np.full(len(parameter_names), float(default_value), dtype=float)
    if isinstance(spec, Mapping):
        mapping = {str(key): float(value) for key, value in spec.items()}
        unknown = sorted(set(mapping).difference(parameter_names))
        if unknown:
            raise ValueError(
                f"{field_name} has unknown parameter names {unknown!r}; "
                f"expected subset of {list(parameter_names)!r}"
            )
        for index, name in enumerate(parameter_names):
            if name in mapping:
                values[index] = float(mapping[name])
    else:
        values[:] = float(spec)

    if must_be_positive and np.any(values <= 0.0):
        raise ValueError(f"{field_name} values must be > 0")
    return values


def _default_transform_kind(parameter_name: str) -> str:
    """Return default transform kind for known social parameters."""

    if parameter_name in _UNIT_INTERVAL_PARAMS:
        return "unit_interval_logit"
    if parameter_name in _NONNEGATIVE_PARAMS or parameter_name in _POSITIVE_PARAMS:
        return "positive_log"
    return "identity"


def _build_latent_initialization(
    *,
    parameter_names: tuple[str, ...],
    transform_kinds: Mapping[str, str],
    n_blocks: int,
    initial_group_location: Mapping[str, float] | None,
    initial_group_scale: Mapping[str, float] | None,
    initial_block_params: Sequence[Mapping[str, float]] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build initial values for Stan latent parameters."""

    group_loc_init = np.zeros(len(parameter_names), dtype=float)
    group_log_scale_init = np.zeros(len(parameter_names), dtype=float)

    if initial_group_location is not None:
        for param_index, name in enumerate(parameter_names):
            if name in initial_group_location:
                theta = float(initial_group_location[name])
                group_loc_init[param_index] = _inverse_transform(theta, transform_kinds[name])

    if initial_group_scale is not None:
        for param_index, name in enumerate(parameter_names):
            if name in initial_group_scale:
                sigma = float(initial_group_scale[name])
                if sigma <= 0.0:
                    raise ValueError(f"initial_group_scale[{name!r}] must be > 0")
                group_log_scale_init[param_index] = float(np.log(sigma))

    block_z_init = np.zeros((n_blocks, len(parameter_names)), dtype=float)
    if initial_block_params is not None:
        for block_index, block_params in enumerate(initial_block_params):
            for param_index, name in enumerate(parameter_names):
                if name in block_params:
                    theta = float(block_params[name])
                    block_z_init[block_index, param_index] = _inverse_transform(theta, transform_kinds[name])
                elif initial_group_location is not None and name in initial_group_location:
                    block_z_init[block_index, param_index] = group_loc_init[param_index]
                else:
                    block_z_init[block_index, param_index] = 0.0
    else:
        for param_index, name in enumerate(parameter_names):
            default_value = 0.0
            if initial_group_location is not None and name in initial_group_location:
                default_value = group_loc_init[param_index]
            block_z_init[:, param_index] = default_value

    return group_loc_init, group_log_scale_init, block_z_init


def _inverse_transform(value: float, kind: str) -> float:
    """Map constrained value to latent ``z`` according to transform kind."""

    if kind == "identity":
        return float(value)
    if kind == "positive_log":
        clipped = max(float(value), 1e-12)
        return float(np.log(clipped))
    if kind == "unit_interval_logit":
        theta = float(np.clip(float(value), 1e-9, 1.0 - 1e-9))
        return float(np.log(theta / (1.0 - theta)))
    raise ValueError(f"unsupported transform kind {kind!r}")


__all__ = [
    "build_social_subject_inputs",
    "load_social_stan_code",
    "social_cache_tag",
    "social_supported_component_ids",
]
