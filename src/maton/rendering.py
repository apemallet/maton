"""Display-independent projections of compiled MuJoCo geometry."""

from itertools import product
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull


def draw_model(model: Any, data: Any, axes: Any, plane: tuple[int, int] = (0, 2)) -> None:
    """Draw sphere, capsule, cylinder, and box silhouettes in a coordinate plane."""
    from matplotlib.patches import Circle, Polygon
    import mujoco

    kinds = mujoco.mjtGeom  # pyright: ignore[reportAttributeAccessIssue]
    coordinates = list(plane)
    for geom in range(model.ngeom):
        kind = int(model.geom_type[geom])
        pos = data.geom_xpos[geom][coordinates]
        matrix = data.geom_xmat[geom].reshape(3, 3)
        size, color = model.geom_size[geom], model.geom_rgba[geom]
        if kind == kinds.mjGEOM_BOX:
            vertices = np.array(list(product((-1, 1), repeat=3))) * size
            points = (vertices @ matrix.T)[:, coordinates] + pos
            axes.add_patch(Polygon(points[ConvexHull(points).vertices], color=color))
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
