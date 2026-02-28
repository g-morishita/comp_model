"""Fixed-sequence demonstrator implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from comp_model.core.contracts import DecisionContext
from comp_model.plugins import ComponentManifest


class FixedSequenceDemonstrator:
    """Demonstrator that follows a fixed action sequence by trial index.

    Parameters
    ----------
    sequence : Sequence[Any]
        Action sequence indexed by ``trial_index``.
    fallback : {"error", "repeat_last"}, optional
        Behavior when ``trial_index`` exceeds ``sequence`` length.

    Raises
    ------
    ValueError
        If sequence is empty or fallback is unsupported.
    """

    def __init__(self, sequence: Sequence[Any], *, fallback: str = "error") -> None:
        if len(sequence) == 0:
            raise ValueError("sequence must include at least one action")
        if fallback not in {"error", "repeat_last"}:
            raise ValueError("fallback must be 'error' or 'repeat_last'")

        self._sequence = tuple(sequence)
        self._fallback = fallback

    def start_episode(self) -> None:
        """Reset episode state.

        Notes
        -----
        This demonstrator is purely sequence-driven and has no mutable state.
        """

    def action_distribution(
        self,
        observation: Any,
        *,
        context: DecisionContext[Any],
    ) -> dict[Any, float]:
        """Return one-hot distribution for current sequence action.

        Parameters
        ----------
        observation : Any
            Unused observation payload.
        context : DecisionContext[Any]
            Runtime decision context.

        Returns
        -------
        dict[Any, float]
            One-hot action distribution.

        Raises
        ------
        ValueError
            If selected sequence action is unavailable.
        IndexError
            If trial index exceeds sequence and fallback is ``"error"``.
        """

        del observation
        target_action = self._resolve_action(context.trial_index)
        if target_action not in context.available_actions:
            raise ValueError(
                f"sequence action {target_action!r} unavailable for trial {context.trial_index}; "
                f"available={context.available_actions!r}"
            )

        return {action: 1.0 if action == target_action else 0.0 for action in context.available_actions}

    def update(
        self,
        observation: Any,
        action: Any,
        outcome: Any,
        *,
        context: DecisionContext[Any],
    ) -> None:
        """No-op update hook for runtime compatibility."""

        del observation, action, outcome, context

    def _resolve_action(self, trial_index: int) -> Any:
        """Return sequence action for trial index with fallback behavior."""

        if trial_index < len(self._sequence):
            return self._sequence[trial_index]

        if self._fallback == "repeat_last":
            return self._sequence[-1]

        raise IndexError(
            f"trial_index {trial_index} out of range for sequence length {len(self._sequence)}"
        )


def create_fixed_sequence_demonstrator(
    *,
    sequence: Sequence[Any],
    fallback: str = "error",
) -> FixedSequenceDemonstrator:
    """Factory used by plugin discovery."""

    return FixedSequenceDemonstrator(sequence=sequence, fallback=fallback)


PLUGIN_MANIFESTS = [
    ComponentManifest(
        kind="demonstrator",
        component_id="fixed_sequence_demonstrator",
        factory=create_fixed_sequence_demonstrator,
        description="Deterministic demonstrator that follows a fixed action sequence",
    )
]
