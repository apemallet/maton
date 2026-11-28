"""Record physics trajectories and summarize saved evaluations."""

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
import mujoco


def record(env: Any, policy: Callable, output: Path, seed: int, level: int = 0) -> Path:
    if not 0 <= level < len(env.curriculum.levels):
        raise ValueError("Invalid curriculum level")
    output.mkdir(parents=True, exist_ok=False)
    state = jax.jit(env.reset)(jax.random.key(seed), level)
    step = jax.jit(env.step)
    observe = jax.jit(env.observation)
    positions = [np.asarray(state.data.qpos)]
    velocities = [np.asarray(state.data.qvel)]
    actions, rewards, observations = [], [], []
    for _ in range(env.config.episode_length):
        obs = np.asarray(observe(state))
        control = np.asarray(policy(obs), dtype=np.float32)
        if control.shape != (env.action_size,) or not np.isfinite(control).all():
            raise ValueError(f"Policy must return {env.action_size} finite motor actions")
        state = step(state, control)
        observations.append(obs)
        actions.append(control)
        rewards.append(float(state.reward))
        positions.append(np.asarray(state.data.qpos))
        velocities.append(np.asarray(state.data.qvel))
        if bool(state.terminated) or bool(state.truncated):
            break
    if not np.isfinite(positions).all():
        raise FloatingPointError("Non-finite simulation state during evaluation")
    model = env.cpu_model(state)
    mujoco.mj_saveLastXML(str(output / "model.xml"), model)  # pyright: ignore[reportAttributeAccessIssue]
    np.savez_compressed(output / "trajectory.npz", qpos=np.asarray(positions),
                        qvel=np.asarray(velocities), actions=np.asarray(actions),
                        rewards=np.asarray(rewards), observations=np.asarray(observations),
                        terrain=np.asarray(state.terrain))
    summary = dict(seed=seed, steps=len(rewards), reward=sum(rewards),
                   level=level, level_name=env.curriculum.levels[level].name,
                   success=bool(state.success),
                   distance=float(positions[-1][0] - positions[0][0]),
                   terminated=bool(state.terminated), truncated=bool(state.truncated),
                   dt=env.config.timestep * env.config.frame_skip)
    poses = np.asarray(positions)
    summary["upright_fraction"] = float(np.mean((poses[:, 1] + env.walker.spawn_height >= 0.5 * env.walker.spawn_height) &
                                                (np.cos(poses[:, 2]) >= np.cos(1.4))))
    (output / "episode.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output


def summarize_episodes(episodes: list[Path], output: Path) -> dict:
    """Summarize held-out checkpoint rollouts, keeping per-seed evidence."""
    if not episodes:
        raise ValueError("At least one episode is required")
    records = [json.loads((episode / "episode.json").read_text()) for episode in episodes]
    summary = dict(episodes=len(records), successes=sum(row["success"] for row in records),
                   success_rate=float(np.mean([row["success"] for row in records])),
                   mean_distance=float(np.mean([row["distance"] for row in records])),
                   mean_return=float(np.mean([row["reward"] for row in records])),
                   mean_upright_fraction=float(np.mean([row["upright_fraction"] for row in records])),
                   results=records)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    return summary
