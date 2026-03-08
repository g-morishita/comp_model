"""Block-fit strategy helpers for subject/study inference APIs."""

from __future__ import annotations

from typing import Any, Literal, cast

BlockFitStrategy = Literal["independent", "joint"]
"""Strategy for handling multiple blocks within a subject fit."""

JOINT_BLOCK_ID = "__joint__"
"""Synthetic block ID used when all blocks are fit jointly."""


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
    "coerce_block_fit_strategy",
]
