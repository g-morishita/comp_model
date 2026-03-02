"""Block-fit strategy helpers for subject/study inference APIs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

from comp_model.core.contracts import AgentModel
from comp_model.core.events import EpisodeTrace
from comp_model.runtime.replay import ReplayResult, ReplayStep

from .likelihood import ActionReplayLikelihood, LikelihoodProgram

BlockFitStrategy = Literal["independent", "joint"]
"""Strategy for handling multiple blocks within a subject fit."""

JOINT_BLOCK_ID = "__joint__"
"""Synthetic block ID used when all blocks are fit jointly."""


class JointBlockLikelihoodProgram:
    """Likelihood program that sums replay likelihood across multiple blocks.

    Parameters
    ----------
    block_traces : Sequence[EpisodeTrace]
        Canonical traces for all blocks belonging to one subject.
    likelihood_program : LikelihoodProgram | None, optional
        Base likelihood evaluator applied to each block trace. Defaults to
        :class:`ActionReplayLikelihood`.

    Notes
    -----
    The ``trace`` argument passed to :meth:`evaluate` is ignored. This allows
    existing estimator interfaces (which expect one trace argument) to optimize
    a single parameter set against all block traces jointly.
    """

    def __init__(
        self,
        *,
        block_traces: Sequence[EpisodeTrace],
        likelihood_program: LikelihoodProgram | None = None,
    ) -> None:
        if not block_traces:
            raise ValueError("block_traces must include at least one trace")
        self._block_traces = tuple(block_traces)
        self._likelihood_program = (
            likelihood_program if likelihood_program is not None else ActionReplayLikelihood()
        )

    def evaluate(self, trace: EpisodeTrace, model: AgentModel) -> ReplayResult:
        """Evaluate and sum likelihood across all stored block traces."""

        del trace
        total_log_likelihood = 0.0
        all_steps: list[ReplayStep] = []
        for block_trace in self._block_traces:
            replay = self._likelihood_program.evaluate(block_trace, model)
            total_log_likelihood += float(replay.total_log_likelihood)
            all_steps.extend(replay.steps)
        return ReplayResult(
            total_log_likelihood=float(total_log_likelihood),
            steps=tuple(all_steps),
        )


def coerce_block_fit_strategy(
    raw: Any,
    *,
    field_name: str = "block_fit_strategy",
) -> BlockFitStrategy:
    """Parse and validate block-fit strategy values.

    Parameters
    ----------
    raw : Any
        Raw strategy value.
    field_name : str, optional
        Field label used in validation errors.

    Returns
    -------
    {"independent", "joint"}
        Normalized strategy value.

    Raises
    ------
    ValueError
        If ``raw`` is not one of ``"independent"`` or ``"joint"``.
    """

    if raw is None:
        return "independent"

    value = str(raw).strip()
    if value in {"independent", "joint"}:
        return cast(BlockFitStrategy, value)

    raise ValueError(
        f"{field_name} must be one of {{'independent', 'joint'}}, got {raw!r}"
    )


__all__ = [
    "BlockFitStrategy",
    "JOINT_BLOCK_ID",
    "JointBlockLikelihoodProgram",
    "coerce_block_fit_strategy",
]
