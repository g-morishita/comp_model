"""Demonstrator implementations and factory helpers."""

from .factories import make_fixed_sequence, make_noisy_best, make_rl_demonstrator
from .fixed_sequence import FixedSequenceDemonstrator
from .noisy_best import NoisyBestArmDemonstrator
from .registry import DemonstratorFactory, DemonstratorRegistry
from .rl_agent import RLDemonstrator

__all__ = [
    "DemonstratorFactory",
    "DemonstratorRegistry",
    "FixedSequenceDemonstrator",
    "NoisyBestArmDemonstrator",
    "RLDemonstrator",
    "make_fixed_sequence",
    "make_noisy_best",
    "make_rl_demonstrator",
]
