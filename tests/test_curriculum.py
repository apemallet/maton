from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from mujoco import mjx
import numpy as np
import pytest

from maton.config import Config, load_run, save_run
from maton.curriculum import CurriculumConfig, CurriculumProgress, Level
from maton.env import EnvConfig, WalkerEnv
from maton.terrain import TerrainConfig, sample_terrain


def test_default_curriculum_starts_flat_and_ends_with_deep_gaps():
    curriculum = CurriculumConfig.load(Path("configs/curricula/walker.toml"))
    assert len(curriculum.levels) >= 5
    terrain = sample_terrain(jax.random.key(0), curriculum.levels[0].terrain)
    np.testing.assert_array_equal(terrain[:, 0], 0)
    assert curriculum.levels[-1].terrain.gap_height >= 20


def test_progress_needs_a_recent_success_rate_and_ignores_old_parallel_episodes():
    flat = TerrainConfig(block_frequency=0, gap_frequency=0)
    progress = CurriculumProgress(CurriculumConfig((Level("one", 2, flat, window=3), Level("two", 2, flat, window=3))))
    assert not progress.record(True, 0)
    assert not progress.record(False, 0)
    assert not progress.record(False, 0)  # Window [T, F, F]: one recent success.
    assert not progress.record(True, 0)  # Oldest success slid out: [F, F, T].
    assert progress.level == 0 and progress.successes == 1
    assert progress.record(True, 0)  # [F, T, T]
    assert progress.level == 1 and progress.successes == 0
    assert not progress.record(True, 0)  # Stale episode from a parallel environment.
    assert progress.successes == 0
    progress.record(True, 1)
    progress.record(True, 1)
    assert progress.complete and progress.level == 1
    progress.record(True, 1)
    assert progress.successes == 2


def test_progress_state_round_trips():
    flat = TerrainConfig(block_frequency=0, gap_frequency=0)
    config = CurriculumConfig((Level("one", 3, flat), Level("two", 3, flat)))
    progress = CurriculumProgress(config)
    for success in (True, False, True):
        progress.record(success, 0)
    restored = CurriculumProgress(config)
    restored.restore(progress.state())
    assert restored == progress and restored.snapshot()["successes"] == 2


def test_curriculum_reference_is_relative_and_snapshot_is_self_contained(tmp_path, monkeypatch):
    folder = tmp_path / "jobs"
    folder.mkdir()
    reference = folder / "course.toml"
    reference.write_text('[[levels]]\nname="flat"\n[levels.terrain]\nblock_frequency=0\ngap_frequency=0\n')
    job = folder / "job.toml"
    walker = Path('configs/walkers/basic.toml').resolve().as_posix()
    job.write_text(f'algorithm = "ppo"\nwalker = "{walker}"\ncurriculum = "course.toml"\n')
    monkeypatch.chdir(tmp_path)
    config = Config.load(job)
    save_run(config, tmp_path / "run")
    reference.unlink()
    restored = load_run(tmp_path / "run")
    assert restored.resolved() == config.resolved()


def test_invalid_curricula():
    flat = TerrainConfig(block_frequency=0, gap_frequency=0)
    with pytest.raises(ValueError):
        CurriculumConfig(())
    with pytest.raises(ValueError):
        Level("flat", 0, flat)
    with pytest.raises(ValueError, match="window"):
        Level("flat", 5, flat, window=4)
    with pytest.raises(ValueError, match="share"):
        CurriculumConfig((Level("one", 1, flat), Level("two", 1, replace(flat, slots=2))))
    with pytest.raises(ValueError, match="share"):
        CurriculumConfig((Level("one", 1, flat), Level("two", 1, replace(flat, run_out=2.0))))
    # Fine-tuning curricula may start directly on hard terrain.
    CurriculumConfig((Level("one", 1, TerrainConfig()), Level("two", 1, TerrainConfig())))


def test_sinking_into_a_deep_gap_ends_the_episode_with_valid_collision_and_lidar_geometry():
    flat = TerrainConfig(slots=1, block_frequency=0, gap_frequency=0)
    pit = replace(flat, gap_frequency=1, gap_height=20, gap_width=0.8)
    curriculum = CurriculumConfig((Level("flat", 1, flat), Level("deep", 1, pit)))
    env = WalkerEnv(EnvConfig(reset_noise=0, frame_skip=1, episode_length=8), curriculum=curriculum)
    state = jax.jit(env.reset)(jax.random.key(0), 1)
    # Place the whole walker below the original surface, inside the physical pit.
    qpos = state.data.qpos.at[0].set(2.75).at[1].set(-2.0)
    data = mjx.forward(env._model(state.terrain), state.data.replace(qpos=qpos))
    state = eqx.tree_at(lambda s: s.data, state, data)
    model = env.cpu_model(state)
    assert np.all(model.geom_size[:5] > 0)
    assert model.geom_pos[2, 2] + model.geom_size[2, 2] == pytest.approx(float(state.terrain[0, 0]))
    assert env._floor_bottom < float(state.terrain[0, 0])
    assert np.isfinite(env.lidar(state)).all()
    # Straight-down ray still sees the pit bottom when range is long enough.
    env_long = WalkerEnv(EnvConfig(lidar_range=30), curriculum=curriculum)
    distance = float(env_long.lidar(state)[0]) * 30
    assert distance == pytest.approx(-1 - float(state.terrain[0, 0]), abs=1e-4)
    # A torso this far below the floor cannot climb out: the episode fails at once.
    state = jax.jit(env.step)(state, jnp.zeros(4))
    assert bool(state.terminated) and not bool(state.success) and not bool(state.truncated)
    assert np.isfinite(env.observation(state)).all()


def test_finish_line_ends_every_episode_but_only_upright_finishes_succeed():
    env = WalkerEnv(EnvConfig(reset_noise=0, frame_skip=1), TerrainConfig(slots=1))
    state = jax.jit(env.reset)(jax.random.key(0))
    qpos = state.data.qpos.at[0].set(env.terrain_config.length + 0.01)
    step = jax.jit(env.step)
    upright = step(eqx.tree_at(lambda s: s.data.qpos, state, qpos), jnp.zeros(4))
    assert bool(upright.success) and bool(upright.terminated)
    fallen = step(eqx.tree_at(lambda s: s.data.qpos, state, qpos.at[2].set(1.6)), jnp.zeros(4))
    assert bool(fallen.terminated) and not bool(fallen.success)


def test_efficiency_is_enabled_by_promotion_and_survives_completion_and_resume():
    terrain = TerrainConfig()
    config = CurriculumConfig((Level('hard', 4, terrain, window=10),
                               Level('efficient', 4, terrain, window=10, efficiency_penalty=2.0)))
    progress = CurriculumProgress(config)
    for result in [True, True, True, False, False]:
        progress.record(result, 0)
        assert progress.efficiency_weight == 0
    progress.record(True, 1)  # stale episode from another level
    assert progress.efficiency_weight == 0
    progress.record(True, 0)
    assert not progress.complete and progress.level == 1 and progress.efficiency_weight == 2
    for _ in range(20):
        progress.record(False, 1)
    assert progress.efficiency_weight == 2  # remains active before final-level success
    for _ in range(4):
        progress.record(True, 1)
    assert progress.complete and progress.efficiency_weight == 2
    restored = CurriculumProgress(config)
    restored.restore(progress.state())
    assert restored.snapshot()['efficiency_weight'] == 2
    assert CurriculumConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize('penalty', [-1, float('nan'), float('inf')])
def test_invalid_efficiency_penalty(penalty):
    with pytest.raises(ValueError, match='efficiency_penalty'):
        Level('hard', 4, TerrainConfig(), efficiency_penalty=penalty)


def test_standing_in_no_mans_land_hole_is_recoverable():
    config = Config.load('configs/jobs/advanced/biped-tdmpc2-NML-s0.toml')
    env = config.environment()
    state = jax.jit(env.reset)(jax.random.key(0))
    qpos = state.data.qpos.at[0].set(env.terrain_config.length / 2).at[1].set(-0.8)
    state = eqx.tree_at(lambda s: s.data.qpos, state, qpos)
    assert float(env.height(state.data, state.terrain)) == pytest.approx(1.0)
    state = jax.jit(env.step)(state, jnp.zeros(env.action_size))
    assert not bool(state.terminated) and int(state.fallen_steps) == 0
    assert np.isfinite(env.observation(state)).all()
    assert env._floor_bottom < -0.8


def test_global_efficiency_penalty_is_rejected():
    with pytest.raises(ValueError, match='Unknown curriculum settings'):
        CurriculumConfig.from_dict(dict(efficiency_penalty=2, levels=[dict(name='hard')]))
