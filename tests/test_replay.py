import json
from types import SimpleNamespace

import imageio.v2 as imageio
import equinox as eqx
import jax
import matplotlib as mpl
from matplotlib.figure import Figure
import mujoco
import numpy as np
import pytest

from maton.env import EnvConfig, WalkerEnv
from maton.replay import record, summarize_episodes
from maton.rendering import draw_eye_rays, eye_rays, limb_colors, render_episode
from maton.terrain import TerrainConfig


def test_record_exact_course_and_render_without_policy(tmp_path):
    env = WalkerEnv(EnvConfig(episode_length=2, frame_skip=1, lidar_count=5, lidar_range=4),
                    TerrainConfig(slots=1, block_frequency=0, gap_frequency=1))
    episode = record(env, lambda _: np.zeros(4), tmp_path / "episode", 9)
    saved = np.load(episode / "trajectory.npz")
    assert saved["qpos"].shape == (3, 7)
    assert saved["actions"].shape == (2, 4)
    assert np.isfinite(saved["qpos"]).all()
    assert saved["terrain"][0, 0] < 0
    model = mujoco.MjModel.from_xml_path(str(episode / "model.xml"))
    assert np.isclose(model.geom_pos[2, 2] + model.geom_size[2, 2], saved["terrain"][0, 0])
    metadata = json.loads((episode / "episode.json").read_text())
    assert metadata['work_joules'] >= 0
    assert 'work_joules' in saved
    assert saved['work_joules'].sum() == pytest.approx(metadata['work_joules'])
    assert metadata['cost_of_transport'] is None  # failures never count as efficient crossings
    assert metadata["truncated"]
    assert metadata["lidar_count"] == 5 and metadata["lidar_range"] == 4
    summary = summarize_episodes([episode], tmp_path / "summary.json")
    assert summary["episodes"] == 1
    assert summary['mean_successful_cost_of_transport'] is None
    assert summary['mean_successful_seconds'] is None
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
        gold = (frame[..., 0] > 130) & (frame[..., 1] > 100) & (frame[..., 2] < frame[..., 0] * 0.8)
        assert gold.sum() > 5
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


@pytest.mark.parametrize("x,pitch,reach", [(0, 0, 3), (2.7, 0.6, 4), (2.7, -1.2, 30)])
def test_eye_rays_match_sensor_on_rotated_torso_and_gap(x, pitch, reach):
    env = WalkerEnv(EnvConfig(lidar_count=7, lidar_range=reach, reset_noise=0),
                    TerrainConfig(slots=1, block_frequency=0, gap_frequency=1, gap_height=5))
    state = jax.jit(env.reset)(jax.random.key(4))
    state = eqx.tree_at(lambda s: s.data.qpos, state, state.data.qpos.at[0].set(x).at[2].set(pitch))
    model = env.cpu_model(state)
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(state.data.qpos)
    mujoco.mj_forward(model, data)
    origin, endpoints, hits = eye_rays(model, data, 7, reach)
    distances = np.linalg.norm(endpoints - origin, axis=-1)
    expected = np.asarray(env.lidar(state))
    np.testing.assert_allclose(distances / reach, expected, atol=1e-6)
    np.testing.assert_array_equal(hits, expected < 1)
    np.testing.assert_allclose(origin, [x, env.walker.spawn_height])


def test_eye_ray_hits_are_brighter_and_marked_but_misses_are_not():
    model = SimpleNamespace(geom_bodyid=np.array([0, 1]),
                            geom_size=np.array([[10, 1, 0.5], [1, 1, 1]]))
    data = SimpleNamespace(xpos=np.array([[0, 0, 0], [0, 0, 1]]), qpos=np.zeros(3),
                           geom_xpos=np.array([[0, 0, -0.5], [0, 0, 1]]))
    origin, endpoints, hits = eye_rays(model, data, 5, 3)
    assert hits.any() and not hits.all()
    np.testing.assert_allclose(endpoints[hits, 1], 0, atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(endpoints[~hits] - origin, axis=-1), 3)
    axes = Figure().subplots()
    draw_eye_rays(model, data, axes, 5, 3)
    rays = [line for line in axes.lines if line.get_marker() != "o"]
    markers = [line for line in axes.lines if line.get_marker() == "o"]
    assert len(rays) == 5 and len(markers) == 2 * hits.sum()
    assert min(line.get_alpha() for line, hit in zip(rays, hits) if hit) > max(
        line.get_alpha() for line, hit in zip(rays, hits) if not hit)


def test_efficiency_summary_compares_only_successful_crossings(tmp_path):
    episodes = []
    for index, (success, cost, energy, seconds) in enumerate([(True, 2, 200, 10), (True, 4, 400, 20), (False, None, 0, 1)]):
        path = tmp_path / str(index)
        path.mkdir()
        (path / 'episode.json').write_text(json.dumps(dict(success=success, cost_of_transport=cost,
                                                           work_joules=energy, seconds=seconds,
                                                           distance=8 if success else 0,
                                                           reward=1, upright_fraction=1)))
        episodes.append(path)
    summary = summarize_episodes(episodes, tmp_path / 'summary.json')
    assert summary['successes'] == 2
    assert summary['mean_successful_cost_of_transport'] == 3
    assert summary['mean_successful_seconds'] == 15
    assert summary['mean_work_joules'] == 200


def test_evaluation_uses_selected_level_efficiency_reward(tmp_path, monkeypatch):
    from dataclasses import replace
    import jax.numpy as jnp
    from maton.curriculum import CurriculumConfig, Level
    terrain = TerrainConfig(slots=1, block_frequency=0, gap_frequency=0)
    curriculum = CurriculumConfig((Level('crossing', 4, terrain),
                                   Level('efficient', 4, terrain, efficiency_penalty=2)))
    env = WalkerEnv(EnvConfig(reset_noise=0, frame_skip=1), curriculum=curriculum)

    def transition(state, action, weight):
        data = state.data.replace(qpos=state.data.qpos.at[0].set(terrain.length))
        return replace(state, data=data, success=jnp.array(True), terminated=jnp.array(True),
                       reward=jnp.float32(100) - weight * 10 / env.weight), jnp.float32(10)

    monkeypatch.setattr(env, 'transition', transition)
    base = record(env, lambda _: np.zeros(4), tmp_path / 'crossing', 0, level=0)
    efficient = record(env, lambda _: np.zeros(4), tmp_path / 'efficient', 0, level=1)
    first = json.loads((base / 'episode.json').read_text())
    second = json.loads((efficient / 'episode.json').read_text())
    assert first['reward'] - second['reward'] == pytest.approx(20 / env.weight, abs=1e-5)
    assert second['level'] == 1 and second['success']
    assert first['cost_of_transport'] == second['cost_of_transport']
