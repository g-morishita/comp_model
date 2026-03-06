from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from comp_model.core.contracts import DecisionContext
from comp_model.plugins import ComponentManifest


@dataclass(frozen=True, slots=True)
class PerceptualDiscriminationBinary:
	def __post_init__(
		self,
		direction_schedule: Sequence[int],
		intensity_schedule: Sequence[float],
	) -> None:
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
		pass

	def available_actions(self, *, trial_index: int) -> tuple[int]:
		return (0, 1)
	
	def observe(self, *, context: DecisionContext[int]) -> dict[str, int]:
		t = context.trial_index
		intensity = self.intensity_schedule[t]
		direction = self.direction_schedule[t]
		return {"trial_index": t, "intensity": intensity, "direction": direction}

	def transition(self, *, context: DecisionContext[int], rng: np.random.Generator) -> None:
		pass


def create_perceptual_discrimination_binary(
	*,
	direction_schedule: Sequence[int],
	intensity_schedule: Sequence[float],
) -> PerceptualDiscriminationBinary:
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



	

	
		

		
		

