"""Reference model implementations for the new architecture."""

from .q_learning import QLearningAgent
from .random_agent import RandomAgent

__all__ = ["QLearningAgent", "RandomAgent"]
