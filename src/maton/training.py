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


def environment_step(env, state, action):
    """Apply the episode's level reward, including during parallel level transitions."""
    import jax.numpy as jnp
    shaped = getattr(env, "step_with_efficiency", None)
    if shaped is None:
        return env.step(state, action)
    weights = jnp.asarray([level.efficiency_penalty for level in env.curriculum.levels], dtype=jnp.float32)
    return shaped(state, action, weights[state.level])
