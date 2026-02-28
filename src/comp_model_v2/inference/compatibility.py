"""Compatibility checks between trace data and component requirements."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from comp_model_v2.core.events import EpisodeTrace, group_events_by_trial, validate_trace
from comp_model_v2.core.requirements import ComponentRequirements


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Compatibility check result.

    Parameters
    ----------
    is_compatible : bool
        ``True`` when no issues were detected.
    issues : tuple[str, ...]
        Human-readable issue descriptions.
    """

    is_compatible: bool
    issues: tuple[str, ...]


def check_trace_compatibility(
    trace: EpisodeTrace,
    requirements: ComponentRequirements,
) -> CompatibilityReport:
    """Check whether trace payloads satisfy component requirements.

    Parameters
    ----------
    trace : EpisodeTrace
        Event trace to validate.
    requirements : ComponentRequirements
        Declarative field requirements.

    Returns
    -------
    CompatibilityReport
        Compatibility outcome and issue list.
    """

    issues: list[str] = []

    try:
        validate_trace(trace)
    except ValueError as exc:
        issues.append(str(exc))
        return CompatibilityReport(is_compatible=False, issues=tuple(issues))

    grouped = group_events_by_trial(trace)
    for trial_index in sorted(grouped):
        trial_events = grouped[trial_index]

        observation_payload = trial_events[0].payload
        outcome_payload = trial_events[2].payload

        observation = observation_payload.get("observation") if isinstance(observation_payload, Mapping) else None
        outcome = outcome_payload.get("outcome") if isinstance(outcome_payload, Mapping) else None

        for field in requirements.required_observation_fields:
            if not isinstance(observation, Mapping):
                issues.append(
                    f"trial {trial_index}: observation is not a mapping but requires field {field!r}"
                )
                continue
            if field not in observation:
                issues.append(f"trial {trial_index}: missing observation field {field!r}")

        for field in requirements.required_outcome_fields:
            if not _outcome_has_field(outcome, field):
                issues.append(f"trial {trial_index}: missing outcome field {field!r}")

    return CompatibilityReport(is_compatible=len(issues) == 0, issues=tuple(issues))


def assert_trace_compatible(trace: EpisodeTrace, requirements: ComponentRequirements) -> None:
    """Raise ``ValueError`` if trace does not satisfy requirements."""

    report = check_trace_compatibility(trace, requirements)
    if report.is_compatible:
        return

    formatted = "\n".join(f"- {issue}" for issue in report.issues)
    raise ValueError(f"trace is not compatible with component requirements:\n{formatted}")


def _outcome_has_field(outcome: Any, field: str) -> bool:
    """Return whether an outcome exposes a required field."""

    if outcome is None:
        return False

    if isinstance(outcome, Mapping):
        return field in outcome

    return hasattr(outcome, field)
