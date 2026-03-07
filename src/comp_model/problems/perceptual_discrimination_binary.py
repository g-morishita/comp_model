"""Binary perceptual discrimination problem implementation.

The class in this module is intentionally a concrete problem implementation,
not the library's core abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from comp_model.core.contracts import DecisionContext
from comp_model.plugins import ComponentManifest


@dataclass(frozen=True, slots=True)
class PerceptualDiscriminationBinary:
	"""Binary-choice perceptual task.

	Parameters
	----------
	direction_schedule : Sequence[int]
		Per-trial ground-truth direction labels encoded as ``0`` or ``1``.
	intensity_schedule : Sequence[float]
		Per-trial stimulus intensity values constrained to ``[0, 1]``. 1 indicates
		a maximally strong stimulus (100% coherence), while 0 indicates no stimulus 
		(0% coherence).

	Raises
	------
	ValueError
		If schedules are empty, lengths mismatch, intensities are out of bounds,
		or directions are not binary labels.
	"""

	def __post_init__(
		self,
		direction_schedule: Sequence[int],
		intensity_schedule: Sequence[float],
	) -> None:
		"""Validate and normalize configured trial schedules.

		Parameters
		----------
		direction_schedule : Sequence[int]
			Per-trial direction labels encoded as ``0`` or ``1``.
		intensity_schedule : Sequence[float]
			Per-trial stimulus intensities in ``[0, 1]``.

		Raises
		------
		ValueError
			If either schedule is empty, values are invalid, or lengths differ.
		"""

		self.intensity_schedule = np.array(intensity_schedule, dtype=np.float32)
		self.direction_schedule = np.array(direction_schedule, dtype=np.int8)

		if np.any(self.intensity_schedule < 0) or np.any(self.intensity_schedule > 1):
			raise ValueError("Intensity values should be between 0 and 1.")
		
		if len(self.intensity_schedule) == 0:
			raise ValueError("Intensity schedule cannot be empty.")

		if np.any((self.direction_schedule != 0) & (self.direction_schedule != 1)):
			raise ValueError("Direction values should be either 0 or 1.")
		
		if len(self.direction_schedule) == 0:
			raise ValueError("Direction schedule cannot be empty.")
		
		if len(self.intensity_schedule) != len(self.direction_schedule):
			raise ValueError("Intensity and direction schedules must have the same length.")
		
	def reset(self, *, rng: np.random.Generator) -> None:
		"""Reset state before an episode.

		Parameters
		----------
		rng : numpy.random.Generator
			Runtime RNG. Unused by this schedule-driven implementation.
		"""

		pass

	def available_actions(self, *, trial_index: int) -> tuple[int]:
		"""Return legal choice IDs for a trial.

		Parameters
		----------
		trial_index : int
			Zero-based trial index. Unused because actions are constant.

		Returns
		-------
		tuple[int]
			Available actions, always ``(0, 1)``.
		"""

		return (0, 1)
	
	def observe(self, *, context: DecisionContext[int]) -> dict[str, int]:
		"""Return trial observation containing task condition values.

		Parameters
		----------
		context : DecisionContext[int]
			Per-trial context.

		Returns
		-------
		dict[str, int]
			Observation payload with ``trial_index``, ``intensity``, and
			``direction``.
		"""

		t = context.trial_index
		intensity = self.intensity_schedule[t]
		direction = self.direction_schedule[t]
		return {"trial_index": t, "intensity": intensity, "direction": direction}

	def transition(self, *, context: DecisionContext[int], rng: np.random.Generator) -> None:
		"""Advance one trial without change to environment.

		Parameters
		----------
		context : DecisionContext[int]
			Per-trial context. Unused because this problem has effect on
			envrionment.
		rng : numpy.random.Generator
			Runtime RNG. Unused because this problem has effect on envrionment.
		"""


def create_perceptual_discrimination_binary(
	*,
	direction_schedule: Sequence[int],
	intensity_schedule: Sequence[float],
) -> PerceptualDiscriminationBinary:
	"""Factory used by plugin discovery.

	Parameters
	----------
	direction_schedule : Sequence[int]
		Per-trial ground-truth direction labels encoded as ``0`` or ``1``.
	intensity_schedule : Sequence[float]
		Per-trial stimulus intensity values constrained to ``[0, 1]``.

	Returns
	-------
	PerceptualDiscriminationBinary
		Configured problem instance.
	"""

	return PerceptualDiscriminationBinary(
		direction_schedule=direction_schedule,
		intensity_schedule=intensity_schedule,
	)

PLUGIN_MANIFEST = [
	ComponentManifest(
		kind="problem",
		component_id="perceptual_discrimination_binary",
		factory=create_perceptual_discrimination_binary,
		description="A perceptual discrimination task with binary choices.",
	)
]
