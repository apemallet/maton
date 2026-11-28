import json
from types import SimpleNamespace

import imageio.v2 as imageio
import matplotlib as mpl
import mujoco
import numpy as np

from maton.env import EnvConfig, WalkerEnv
from maton.replay import record, summarize_episodes
from maton.rendering import limb_colors, render_episode
from maton.terrain import TerrainConfig


def test_record_exact_course_and_render_without_policy(tmp_path):
    env = WalkerEnv(EnvConfig(episode_length=2, frame_skip=1),
                    TerrainConfig(slots=1, block_frequency=0, gap_frequency=1))
    episode = record(env, lambda _: np.zeros(4), tmp_path / "episode", 9)
    saved = np.load(episode / "trajectory.npz")
    assert saved["qpos"].shape == (3, 7)
    assert saved["actions"].shape == (2, 4)
    assert np.isfinite(saved["qpos"]).all()
    assert saved["terrain"][0, 0] < 0
    model = mujoco.MjModel.from_xml_path(str(episode / "model.xml"))
    assert np.isclose(model.geom_pos[2, 2] + model.geom_size[2, 2], saved["terrain"][0, 0])
    assert json.loads((episode / "episode.json").read_text())["truncated"]
    summary = summarize_episodes([episode], tmp_path / "summary.json")
    assert summary["episodes"] == 1
    assert summary["successes"] == 0
    assert summary["mean_upright_fraction"] == 1.0
    assert json.loads((tmp_path / "summary.json").read_text())["results"][0]["seed"] == 9
    original_style = dict(mpl.rcParams)
    video = render_episode(episode)
    assert dict(mpl.rcParams) == original_style
    with imageio.get_reader(video) as reader:
        frame = reader.get_data(0)
        assert frame.shape == (540, 960, 3)
        # Both the figure border and plot background use the dark theme.
        assert frame[5, 5].max() < 10
        assert np.mean(np.all(frame < 20, axis=-1)) > 0.5
        # Terrain retains a bright cyan rim after video compression.
        cyan = (frame[..., 0] < 100) & (frame[..., 1] > 150) & (frame[..., 2] > 180)
        assert cyan.sum() > 100
        # Blue torso must appear, as well as terrain and feet.
        blue = (frame[..., 2].astype(float) > frame[..., 0] + 50) & (frame[..., 0] < 100)
        assert blue.sum() > 100
    assert video.stat().st_size > 0


def test_limb_brightness_is_stable_and_shared_by_each_bodys_geometries():
    model = SimpleNamespace(
        nbody=4, body_parentid=np.array([0, 0, 1, 2]),
        geom_bodyid=np.array([0, 1, 2, 2, 3]),
        geom_rgba=np.tile([0.2, 0.35, 0.8, 0.7], (5, 1)),
    )
    original = model.geom_rgba.copy()
    colors = limb_colors(model)
    np.testing.assert_array_equal(colors[:2], original[:2])
    np.testing.assert_allclose(colors[2:4, :3], original[2:4, :3] * 0.85)
    np.testing.assert_allclose(colors[4, :3], original[4, :3] * 1.15)
    np.testing.assert_array_equal(colors[:, 3], original[:, 3])
    np.testing.assert_array_equal(model.geom_rgba, original)
    np.testing.assert_array_equal(limb_colors(model), colors)
