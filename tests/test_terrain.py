import jax
import numpy as np
import pytest

from maton.terrain import TerrainConfig, rectangles, sample_terrain


def test_seed_reproducibility_and_gap_geometry():
    cfg = TerrainConfig(slots=4, block_frequency=0, gap_frequency=1)
    terrain = sample_terrain(jax.random.key(3), cfg)
    np.testing.assert_array_equal(terrain, sample_terrain(jax.random.key(3), cfg))
    assert np.all(np.asarray(terrain[:, 0]) < 0)
    rect = np.asarray(rectangles(terrain, cfg))
    np.testing.assert_allclose(rect[1::3, 2], 0)
    np.testing.assert_allclose(rect[2::3, 2], terrain[:, 0])
    np.testing.assert_allclose(rect[:-1, 1], rect[1:, 0], atol=1e-6)
    assert rect[-1, 1] == pytest.approx(cfg.length)


def test_blocks_are_bounded_and_flat_course_is_flat():
    cfg = TerrainConfig(block_frequency=1, gap_frequency=0)
    terrain = sample_terrain(jax.random.key(2), cfg)
    assert np.all(np.asarray(terrain[:, 0]) >= cfg.block_height / 2)
    assert np.all(np.asarray(terrain[:, 0]) <= cfg.block_height)
    cfg = TerrainConfig(block_frequency=0, gap_frequency=0)
    assert np.all(np.asarray(sample_terrain(jax.random.key(2), cfg)[:, 0]) == 0)


@pytest.mark.parametrize('kwargs', [dict(gap_frequency=0.9), dict(block_width=2), dict(gap_height=-1),
                                    dict(gap_height=float('inf')), dict(slots=0)])
def test_invalid_terrain(kwargs):
    with pytest.raises(ValueError):
        TerrainConfig(**kwargs)
