"""Shared training settings, measured in environment transitions."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 256
    learning_rate: float = 1e-4
    checkpoint_every: int = 64

    def __post_init__(self):
        for name in ("steps", "checkpoint_every"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"training.{name} must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("training.learning_rate must be finite and positive")

    def checkpoint_due(self, previous: int, current: int) -> bool:
        """Save once when a collection batch crosses one or more thresholds."""
        return current // self.checkpoint_every > previous // self.checkpoint_every
