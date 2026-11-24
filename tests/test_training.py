import pytest

from maton.training import TrainingConfig


@pytest.mark.parametrize('kwargs', [dict(steps=0), dict(steps=1.5),
                                    dict(checkpoint_every=-1), dict(checkpoint_every=True),
                                    dict(learning_rate=float('nan')), dict(learning_rate=float('inf')),
                                    dict(learning_rate=0)])
def test_invalid_training_config(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_checkpoint_threshold_crossing():
    config = TrainingConfig(checkpoint_every=5)
    assert config.checkpoint_due(0, 8)
    assert config.checkpoint_due(8, 16)
    assert not config.checkpoint_due(16, 18)
