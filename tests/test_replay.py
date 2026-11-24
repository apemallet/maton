import json

import imageio.v2 as imageio
import mujoco
import numpy as np

from maton.env import EnvConfig, WalkerEnv
from maton.replay import record, render_episode, summarize_episodes
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
    video = render_episode(episode)
    with imageio.get_reader(video) as reader:
        frame = reader.get_data(0)
        assert frame.shape == (540, 960, 3)
        # Blue torso must appear, as well as terrain and feet.
        blue = (frame[..., 2].astype(float) > frame[..., 0] + 50) & (frame[..., 0] < 100)
        assert blue.sum() > 100
    assert video.stat().st_size > 0
