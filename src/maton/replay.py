"""Record physics trajectories; render them without loading a trained network."""

from collections.abc import Callable
import json
import os
from pathlib import Path
from typing import Any, cast

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
        if control.shape != (4,) or not np.isfinite(control).all():
            raise ValueError("Policy must return four finite motor actions")
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
    summary["upright_fraction"] = float(np.mean((poses[:, 1] + 1.0 >= 0.5) &
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


def render_episode(episode: Path, output: Path | None = None) -> Path:
    """Side-on geometry projection, portable to machines without an OpenGL display."""
    # imageio's SIGINT preexec hook forces fork(), which is unsafe after JAX
    # starts worker threads. Without that hook Python can use posix_spawn().
    os.environ.setdefault("IMAGEIO_FFMPEG_NO_PREVENT_SIGINT", "1")
    import imageio.v2 as imageio
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Polygon

    output = output or episode / "episode.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    # MuJoCo's compiled bindings are missing exports in its Python type metadata.
    model = mujoco.MjModel.from_xml_path(str(episode / "model.xml"))  # pyright: ignore[reportAttributeAccessIssue]
    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    geom_types = mujoco.mjtGeom  # pyright: ignore[reportAttributeAccessIssue]
    with np.load(episode / "trajectory.npz") as saved:
        positions, velocities = saved["qpos"], saved["qvel"]
        rewards = saved["rewards"]
    metadata = json.loads((episode / "episode.json").read_text())
    dt = metadata["dt"]
    figure = Figure(figsize=(9.6, 5.4), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes((0.04, 0.08, 0.94, 0.84))
    with imageio.get_writer(str(output), fps=1 / dt, codec="libx264", macro_block_size=2) as writer:
        for index, (qpos, qvel) in enumerate(zip(positions, velocities, strict=True)):
            data.qpos[:], data.qvel[:] = qpos, qvel
            mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]
            axes.clear()
            center = max(1.5, float(qpos[0]))
            axes.set(xlim=(center - 3, center + 4), ylim=(-1.4, 2.4),
                     xlabel="distance (m)", ylabel="height (m)", aspect="equal")
            axes.set_facecolor("#eef3f7")
            for geom in range(model.ngeom):
                # Python ints avoid NumPy/pybind enum membership comparisons.
                kind = int(model.geom_type[geom])
                pos = data.geom_xpos[geom][[0, 2]]
                matrix = data.geom_xmat[geom].reshape(3, 3)
                size = model.geom_size[geom]
                color = model.geom_rgba[geom]
                if kind == geom_types.mjGEOM_BOX:
                    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * size[[0, 2]]
                    corners = corners @ matrix[np.ix_([0, 2], [0, 2])].T + pos
                    axes.add_patch(Polygon(corners, color=color))
                elif kind in (geom_types.mjGEOM_CAPSULE, geom_types.mjGEOM_CYLINDER):
                    direction = matrix[[0, 2], 2] * size[1]
                    perpendicular = np.array([-direction[1], direction[0]])
                    perpendicular = perpendicular / max(np.linalg.norm(perpendicular), 1e-8) * size[0]
                    corners = np.array([pos - direction - perpendicular, pos + direction - perpendicular,
                                        pos + direction + perpendicular, pos - direction + perpendicular])
                    axes.add_patch(Polygon(corners, color=color))
                    for endpoint in (pos - direction, pos + direction):
                        axes.add_patch(Circle(endpoint, size[0], color=color))
                elif kind == geom_types.mjGEOM_SPHERE:
                    axes.add_patch(Circle(pos, size[0], color=color))
            score = float(np.sum(rewards[:index]))
            axes.set_title(f"t={index * dt:.2f}s   x={qpos[0]:.2f}m   return={score:.3f}")
            canvas.draw()
            cast(Any, writer).append_data(np.asarray(canvas.buffer_rgba())[..., :3])
    return output
