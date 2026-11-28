"""Fixed-topology, randomly sampled block and trench courses."""
from dataclasses import dataclass
import math
from typing import cast

import einx
import jax
import jax.numpy as jnp
from jaxtyping import Array


@dataclass(frozen=True)
class TerrainConfig:
    slots: int = 6
    spacing: float = 1.5
    start_length: float = 2.0
    block_frequency: float = 0.2
    block_height: float = 0.12
    block_width: float = 0.35
    gap_frequency: float = 0.1
    gap_height: float = 0.15
    gap_width: float = 0.3
    central_gap_width: float = 0.0  # exact, guaranteed pit in the middle slot; zero disables
    central_gap_depth: float = 0.0
    run_out: float = 1.0  # clear floor between the last obstacle slot and the finish line

    def __post_init__(self):
        if not (math.isfinite(self.central_gap_width) and math.isfinite(self.central_gap_depth)
                and 0 <= self.central_gap_width < self.spacing and self.central_gap_depth >= 0):
            raise ValueError("Central gap needs a finite width inside slot spacing and nonnegative depth")
        if bool(self.central_gap_width) != bool(self.central_gap_depth):
            raise ValueError("Central gap width and depth must both be positive or both zero")
        if self.central_gap_width and (self.slots % 2 != 1 or self.start_length != self.run_out):
            raise ValueError("Central gap needs odd slots and equal start_length/run_out")
        if self.slots < 1 or self.spacing <= 0 or self.start_length < 1:
            raise ValueError("Course needs slots, positive spacing, and at least 1m clear start")
        if not (math.isfinite(self.run_out) and self.run_out >= 0):
            raise ValueError("run_out must be finite and nonnegative")
        if not (0 <= self.block_frequency <= 1 and 0 <= self.gap_frequency <= 1
                and self.block_frequency + self.gap_frequency <= 1):
            raise ValueError("Block and gap frequencies must be probabilities with sum <= 1")
        if not (0 < self.block_width < self.spacing and 0 < self.gap_width < self.spacing):
            raise ValueError("Obstacle widths must lie strictly inside slot spacing")
        if not all(math.isfinite(x) and x >= 0 for x in (self.block_height, self.gap_height)):
            raise ValueError("Heights and gap depths must be finite and nonnegative")

    @property
    def length(self) -> float:
        """Distance to the finish line."""
        return self.start_length + self.slots * self.spacing + self.run_out


def sample_terrain(key: Array, config: TerrainConfig) -> Array:
    """Return [slot, (signed height, width)]; a negative height is a trench."""
    kind_key, size_key = jax.random.split(key)
    kind = jax.random.uniform(kind_key, (config.slots,))
    scale = jax.random.uniform(size_key, (config.slots, 2), minval=0.5, maxval=1.0)
    block = kind < config.block_frequency
    gap = (kind >= config.block_frequency) & (kind < config.block_frequency + config.gap_frequency)
    height = jnp.where(block, config.block_height, jnp.where(gap, -config.gap_height, 0.0))
    width = jnp.where(block, config.block_width, config.gap_width)
    terrain = jnp.stack((height * scale[:, 0], width * scale[:, 1]), axis=-1)
    if config.central_gap_width:
        terrain = terrain.at[config.slots // 2].set(jnp.array([-config.central_gap_depth, config.central_gap_width]))
    return terrain


BEYOND_FINISH = 2.0  # floor past the finish line, so no course ends in a drop


def rectangle_count(config: TerrainConfig) -> int:
    return 2 + 3 * config.slots


def rectangles(terrain: Array, config: TerrainConfig) -> Array:
    """Return [rectangle, (left, right, top)] for collision and lidar alike: the start strip,
    three tiles per slot (floor, obstacle, floor), then run-out floor past the finish."""
    center = config.start_length + (jnp.arange(config.slots) + 0.5) * config.spacing
    left = center - terrain[:, 1] / 2
    right = center + terrain[:, 1] / 2
    start = center - config.spacing / 2
    end = center + config.spacing / 2
    zero = jnp.zeros(config.slots)
    tiles = jnp.stack((jnp.stack((start, left, zero), -1),
                       jnp.stack((left, right, terrain[:, 0]), -1),
                       jnp.stack((right, end, zero), -1)), axis=1)
    flat = cast(Array, einx.id("slot part coord -> (slot part) coord", tiles))  # pyright: ignore[reportPrivateImportUsage]
    slots_end = config.start_length + config.slots * config.spacing
    return jnp.concatenate((jnp.array([[-2.0, config.start_length, 0.0]]), flat,
                            jnp.array([[slots_end, config.length + BEYOND_FINISH, 0.0]])))
