"""Display-independent geometry projections and saved-episode videos."""

from itertools import product
import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.spatial import ConvexHull


def limb_colors(model: Any) -> np.ndarray:
    """Stable per-body brightness offsets; preserve torso, terrain, and alpha."""
    colors = np.array(model.geom_rgba, copy=True)
    # Walker limbs descend from the root torso; terrain belongs to the world.
    bodies = [body for body in range(1, model.nbody) if model.body_parentid[body] != 0]
    for body, brightness in zip(bodies, np.linspace(0.85, 1.15, len(bodies)), strict=True):
        mask = model.geom_bodyid == body
        colors[mask, :3] = np.clip(colors[mask, :3] * brightness, 0, 1)
    return colors


def eye_rays(model: Any, data: Any, count: int, reach: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Planar lidar segments against saved terrain, excluding the walker's body."""
    origin = np.asarray(data.xpos[1])[[0, 2]]
    angle = np.linspace(-np.pi / 2, np.pi / 12, count) - data.qpos[2]
    direction = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    # Match the sensor's slab intersection, including nearly parallel rays.
    direction = np.where(np.abs(direction) < 1e-6, 1e-6, direction)
    terrain = model.geom_bodyid == 0
    centers = data.geom_xpos[terrain][:, [0, 2]]
    sizes = model.geom_size[terrain][:, [0, 2]]
    t1 = (centers - sizes - origin) / direction[:, None, :]
    t2 = (centers + sizes - origin) / direction[:, None, :]
    near = np.maximum(np.minimum(t1, t2).max(axis=-1), 0)
    far = np.maximum(t1, t2).min(axis=-1)
    distance = np.minimum(np.where(far >= near, near, np.inf).min(axis=-1), reach)
    return origin, origin + direction * distance[:, None], distance < reach


def draw_eye_rays(model: Any, data: Any, axes: Any, count: int, reach: float) -> None:
    """Faint gold rays; terrain hits brighten and end in a small luminous dot."""
    origin, endpoints, hits = eye_rays(model, data, count, reach)
    for endpoint, hit in zip(endpoints, hits, strict=True):
        proximity = 1 - np.clip(np.linalg.norm(endpoint - origin) / reach, 0, 1)
        axes.plot([origin[0], endpoint[0]], [origin[1], endpoint[1]],
                  color="#ffd78a", linewidth=0.85, alpha=0.24 + 0.22 * proximity if hit else 0.16,
                  solid_capstyle="round", zorder=0.8)
        if hit:
            axes.plot(*endpoint, "o", color="#ffd78a", markersize=7, alpha=0.09,
                      markeredgewidth=0, zorder=3)
            axes.plot(*endpoint, "o", color="#ffd78a", markersize=3.2, alpha=0.95,
                      markeredgewidth=0, zorder=3)


def draw_model(model: Any, data: Any, axes: Any, plane: tuple[int, int] = (0, 2),
               *, colors: np.ndarray | None = None) -> None:
    """Draw sphere, capsule, cylinder, and box silhouettes in a coordinate plane."""
    from matplotlib.patches import Circle, Polygon
    import mujoco

    kinds = mujoco.mjtGeom  # pyright: ignore[reportAttributeAccessIssue]
    coordinates = list(plane)
    for geom in range(model.ngeom):
        kind = int(model.geom_type[geom])
        pos = data.geom_xpos[geom][coordinates]
        matrix = data.geom_xmat[geom].reshape(3, 3)
        size = model.geom_size[geom]
        color = (model.geom_rgba if colors is None else colors)[geom]
        if kind == kinds.mjGEOM_BOX:
            vertices = np.array(list(product((-1, 1), repeat=3))) * size
            points = (vertices @ matrix.T)[:, coordinates] + pos
            silhouette = points[ConvexHull(points).vertices]
            if model.geom_bodyid[geom] == 0:
                # Static terrain: a luminous rim over a subdued blue interior.
                axes.add_patch(Polygon(silhouette, facecolor="#071b24", edgecolor="#39e7ff",
                                       linewidth=1.25, joinstyle="round"))
            else:
                axes.add_patch(Polygon(silhouette, color=color))
        elif kind in (kinds.mjGEOM_CAPSULE, kinds.mjGEOM_CYLINDER):
            direction = matrix[coordinates, 2] * size[1]
            perpendicular = np.array([-direction[1], direction[0]])
            perpendicular = perpendicular / max(np.linalg.norm(perpendicular), 1e-8) * size[0]
            corners = np.array([pos - direction - perpendicular, pos + direction - perpendicular,
                                pos + direction + perpendicular, pos - direction + perpendicular])
            axes.add_patch(Polygon(corners, color=color))
            for endpoint in (pos - direction, pos + direction):
                axes.add_patch(Circle(endpoint, size[0], color=color))
        elif kind == kinds.mjGEOM_SPHERE:
            axes.add_patch(Circle(pos, size[0], color=color))


def render_episode(episode: Path, output: Path | None = None) -> Path:
    """Side-on geometry projection, portable to machines without an OpenGL display."""
    # imageio's SIGINT preexec hook forces fork(), which is unsafe after JAX
    # starts worker threads. Without that hook Python can use posix_spawn().
    os.environ.setdefault("IMAGEIO_FFMPEG_NO_PREVENT_SIGINT", "1")
    import imageio.v2 as imageio
    import mujoco
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.style import context

    output = output or episode / "episode.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    # MuJoCo's compiled bindings are missing exports in its Python type metadata.
    model = mujoco.MjModel.from_xml_path(str(episode / "model.xml"))  # pyright: ignore[reportAttributeAccessIssue]
    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    colors = limb_colors(model)
    with np.load(episode / "trajectory.npz") as saved:
        positions, velocities = saved["qpos"], saved["qvel"]
        rewards = saved["rewards"]
    metadata = json.loads((episode / "episode.json").read_text())
    dt = metadata["dt"]
    lidar_count, lidar_range = metadata.get("lidar_count", 9), metadata.get("lidar_range", 3.0)
    with context("dark_background"), imageio.get_writer(
        str(output), fps=1 / dt, codec="libx264", macro_block_size=2,
    ) as writer:
        figure = Figure(figsize=(9.6, 5.4), dpi=100)
        canvas = FigureCanvasAgg(figure)
        axes = figure.add_axes((0.04, 0.08, 0.94, 0.84))
        for index, (qpos, qvel) in enumerate(zip(positions, velocities, strict=True)):
            data.qpos[:], data.qvel[:] = qpos, qvel
            mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]
            axes.clear()
            center = max(1.5, float(qpos[0]))
            axes.set(xlim=(center - 3, center + 4), ylim=(-1.4, 2.4),
                     xlabel="distance (m)", ylabel="height (m)", aspect="equal")
            draw_model(model, data, axes, colors=colors)
            draw_eye_rays(model, data, axes, lidar_count, lidar_range)
            score = float(np.sum(rewards[:index]))
            axes.set_title(f"t={index * dt:.2f}s   x={qpos[0]:.2f}m   return={score:.3f}")
            canvas.draw()
            cast(Any, writer).append_data(np.asarray(canvas.buffer_rgba())[..., :3])
    return output
