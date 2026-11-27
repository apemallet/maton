from dataclasses import is_dataclass
from pathlib import Path

import pytest

from maton.algorithms import ALGORITHMS, algorithm


@pytest.mark.parametrize('name', sorted(ALGORITHMS))
def test_registered_trainers_share_one_interface(name):
    spec = ALGORITHMS[name]
    module = spec.load()
    assert is_dataclass(module.Config) and callable(module.train) and callable(module.load_agent)
    assert all(Path(source).is_file() for source in spec.sources())
    assert spec.checkpoint(Path('t'), '00000008') != spec.checkpoint(Path('t'))


def test_unknown_algorithm_is_rejected():
    with pytest.raises(ValueError, match='must be one of'):
        algorithm('unknown')
