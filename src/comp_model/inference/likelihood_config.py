"""Config-driven likelihood-program parsing helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .likelihood import ActionReplayLikelihood, ActorSubsetReplayLikelihood, LikelihoodProgram


def likelihood_program_from_config(
    likelihood_cfg: Mapping[str, Any] | None,
) -> LikelihoodProgram:
    """Build a likelihood program from declarative config.

    Parameters
    ----------
    likelihood_cfg : Mapping[str, Any] | None
        Likelihood configuration mapping. When ``None``, returns
        :class:`ActionReplayLikelihood`.

    Returns
    -------
    LikelihoodProgram
        Parsed likelihood evaluator.

    Raises
    ------
    ValueError
        If configuration fields are invalid.
    """

    if likelihood_cfg is None:
        return ActionReplayLikelihood()

    cfg = _require_mapping(likelihood_cfg, field_name="likelihood")
    likelihood_type = _coerce_non_empty_str(
        cfg.get("type", "action_replay"),
        field_name="likelihood.type",
    )

    if likelihood_type == "action_replay":
        return ActionReplayLikelihood()

    if likelihood_type == "actor_subset_replay":
        scored_actor_ids_raw = cfg.get("scored_actor_ids", ["subject"])
        scored_actor_ids: tuple[str, ...] | None
        if scored_actor_ids_raw is None:
            scored_actor_ids = None
        else:
            scored_actor_ids = tuple(
                _coerce_non_empty_str(
                    value,
                    field_name="likelihood.scored_actor_ids[]",
                )
                for value in _require_sequence(
                    scored_actor_ids_raw,
                    field_name="likelihood.scored_actor_ids",
                )
            )

        return ActorSubsetReplayLikelihood(
            fitted_actor_id=_coerce_non_empty_str(
                cfg.get("fitted_actor_id", "subject"),
                field_name="likelihood.fitted_actor_id",
            ),
            scored_actor_ids=scored_actor_ids,
            auto_fill_unmodeled_actors=bool(
                cfg.get("auto_fill_unmodeled_actors", True)
            ),
        )

    raise ValueError(
        "likelihood.type must be one of "
        "{'action_replay', 'actor_subset_replay'}"
    )


def _coerce_non_empty_str(raw: Any, *, field_name: str) -> str:
    """Coerce non-empty string with explicit field context."""

    if raw is None:
        raise ValueError(f"{field_name} must be a non-empty string")

    value = str(raw).strip()
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_mapping(raw: Any, *, field_name: str) -> dict[str, Any]:
    """Require dictionary-like config value."""

    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return dict(raw)


def _require_sequence(raw: Any, *, field_name: str) -> list[Any]:
    """Require list-like config value."""

    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{field_name} must be an array")
    return list(raw)


__all__ = ["likelihood_program_from_config"]
