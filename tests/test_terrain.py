import jax
import numpy as np
import pytest

from maton.terrain import BEYOND_FINISH, TerrainConfig, rectangle_count, rectangles, sample_terrain


def test_seed_reproducibility_and_gap_geometry():
    cfg = TerrainConfig(slots=4, block_frequency=0, gap_frequency=1)
    terrain = sample_terrain(jax.random.key(3), cfg)
    np.testing.assert_array_equal(terrain, sample_terrain(jax.random.key(3), cfg))
    assert np.all(np.asarray(terrain[:, 0]) < 0)
    rect = np.asarray(rectangles(terrain, cfg))
    np.testing.assert_allclose(rect[1::3, 2], 0)
    np.testing.assert_allclose(rect[2::3, 2], terrain[:, 0])
    np.testing.assert_allclose(rect[:-1, 1], rect[1:, 0], atol=1e-6)
    assert len(rect) == rectangle_count(cfg)
    # Clear run-out floor before the finish line, and floor beyond it.
    assert rect[-1, 0] == pytest.approx(cfg.length - cfg.run_out)
    assert rect[-1, 1] == pytest.approx(cfg.length + BEYOND_FINISH)
    assert rect[-1, 2] == 0


def test_blocks_are_bounded_and_flat_course_is_flat():
    cfg = TerrainConfig(block_frequency=1, gap_frequency=0)
    terrain = sample_terrain(jax.random.key(2), cfg)
    assert np.all(np.asarray(terrain[:, 0]) >= cfg.block_height / 2)
    assert np.all(np.asarray(terrain[:, 0]) <= cfg.block_height)
    cfg = TerrainConfig(block_frequency=0, gap_frequency=0)
    assert np.all(np.asarray(sample_terrain(jax.random.key(2), cfg)[:, 0]) == 0)


@pytest.mark.parametrize('kwargs', [dict(gap_frequency=0.9), dict(block_width=2), dict(gap_height=-1),
                                    dict(gap_height=float('inf')), dict(slots=0), dict(run_out=-1)])
def test_invalid_terrain(kwargs):
    with pytest.raises(ValueError):
        TerrainConfig(**kwargs)


def test_no_mans_land_has_exactly_one_recoverable_central_hole():
    from pathlib import Path
    from maton.curriculum import CurriculumConfig
    course = CurriculumConfig.load(Path('configs/curricula/no-mans-land.toml'))
    assert len(course.levels) == 2
    assert course.levels[0].terrain == course.levels[1].terrain
    assert [level.efficiency_penalty for level in course.levels] == [0.0, 2.0]
    config = course.levels[0].terrain
    for seed in range(16):
        terrain = np.asarray(sample_terrain(jax.random.key(seed), config))
        assert np.count_nonzero(terrain[:, 0] < 0) == 1
        middle = config.slots // 2
        np.testing.assert_allclose(terrain[middle], [-0.8, 0.85])
        rect = np.asarray(rectangles(terrain, config))
        hole = rect[2 + 3 * middle]
        assert (hole[0] + hole[1]) / 2 == pytest.approx(config.length / 2)
        assert hole[1] - hole[0] == pytest.approx(0.85)
        assert hole[2] == pytest.approx(-0.8)
        np.testing.assert_allclose(rect[:-1, 1], rect[1:, 0], atol=1e-6)


@pytest.mark.parametrize('kwargs', [dict(central_gap_width=0.8), dict(central_gap_depth=0.8),
                                    dict(central_gap_width=2, central_gap_depth=0.8),
                                    dict(central_gap_width=0.8, central_gap_depth=0.8, slots=4),
                                    dict(central_gap_width=float('nan'))])
def test_invalid_central_holes(kwargs):
    with pytest.raises(ValueError):
        TerrainConfig(**kwargs)
