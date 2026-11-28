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
    weight = env.curriculum.levels[level].efficiency_penalty
    transition = jax.jit(lambda state, action: env.transition(state, action, weight))
    observe = jax.jit(env.observation)
    positions = [np.asarray(state.data.qpos)]
    velocities = [np.asarray(state.data.qvel)]
    actions, rewards, observations, work = [], [], [], []
    for _ in range(env.config.episode_length):
        obs = np.asarray(observe(state))
        control = np.asarray(policy(obs), dtype=np.float32)
        if control.shape != (env.action_size,) or not np.isfinite(control).all():
            raise ValueError(f"Policy must return {env.action_size} finite motor actions")
        state, energy = transition(state, control)
        work.append(float(energy))
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
                        terrain=np.asarray(state.terrain), work_joules=np.asarray(work))
    summary: dict[str, Any] = dict(seed=seed, steps=len(rewards), reward=sum(rewards),
                   level=level, level_name=env.curriculum.levels[level].name,
                   success=bool(state.success),
                   distance=float(positions[-1][0] - positions[0][0]),
                   terminated=bool(state.terminated), truncated=bool(state.truncated),
                   dt=env.config.timestep * env.config.frame_skip,
                   lidar_count=env.config.lidar_count, lidar_range=env.config.lidar_range)
    summary['work_joules'] = sum(work)
    summary['seconds'] = len(rewards) * summary['dt']
    # Report efficiency only for completed crossings; failure cannot look efficient by stopping.
    summary['cost_of_transport'] = (sum(work) / (env.weight * summary['distance'])
                                    if summary['success'] and summary['distance'] > 0 else None)
    poses = np.asarray(positions)
    heights = poses[:, 1] + env.walker.spawn_height
    if env.config.terrain_relative_height:
        from maton.terrain import rectangles
        rect = np.asarray(rectangles(state.terrain, env.terrain_config))
        inside = (poses[:, 0, None] >= rect[:, 0]) & (poses[:, 0, None] <= rect[:, 1])
        floor = np.max(np.where(inside, rect[:, 2], -np.inf), axis=1)
        heights -= np.where(np.isfinite(floor), floor, 0.0)
    summary["upright_fraction"] = float(np.mean((heights >= 0.5 * env.walker.spawn_height) &
                                                (np.cos(poses[:, 2]) >= np.cos(1.4))))
    (output / "episode.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output


def summarize_episodes(episodes: list[Path], output: Path) -> dict:
    """Summarize held-out checkpoint rollouts, keeping per-seed evidence."""
    if not episodes:
        raise ValueError("At least one episode is required")
    records = [json.loads((episode / "episode.json").read_text()) for episode in episodes]
    summary: dict[str, Any] = dict(episodes=len(records), successes=sum(row["success"] for row in records),
                   success_rate=float(np.mean([row["success"] for row in records])),
                   mean_distance=float(np.mean([row["distance"] for row in records])),
                   mean_return=float(np.mean([row["reward"] for row in records])),
                   mean_upright_fraction=float(np.mean([row["upright_fraction"] for row in records])),
                   results=records)
    successful_costs = [row['cost_of_transport'] for row in records if row.get('cost_of_transport') is not None]
    summary['mean_work_joules'] = float(np.mean([row['work_joules'] for row in records]))
    summary['mean_successful_cost_of_transport'] = float(np.mean(successful_costs)) if successful_costs else None
    summary['mean_successful_seconds'] = (float(np.mean([row['seconds'] for row in records if row['success']]))
                                           if summary['successes'] else None)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    return summary
