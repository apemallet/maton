from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from maton.cli import main
from maton.preview import preview_walker


@pytest.mark.parametrize('name,neutral', [('basic', True), ('biped', False), ('quadruped', False)])
def test_preview_walker(name, neutral, tmp_path):
    output = tmp_path / 'nested' / f'{name}.png'
    result = preview_walker(Path('configs/walkers') / f'{name}.toml', output, neutral=neutral)
    assert result == output
    frame = imageio.imread(result)
    assert frame.shape == (720, 1800, 4)
    # Each of the three projections must contain visible walker geometry.
    for view in np.array_split(frame, 3, axis=1):
        blue = (view[..., 2].astype(float) > view[..., 0].astype(float) + 50) & (view[..., 0] < 100)
        assert blue.sum() > 100


def test_cli_preview_default_destination(tmp_path, monkeypatch, capsys):
    walker = Path('configs/walkers/basic.toml').resolve()
    monkeypatch.chdir(tmp_path)
    main(['view-walker', str(walker), '--neutral'])
    assert Path('data/walkers/basic.png').is_file()
    assert capsys.readouterr().out.strip() == 'data/walkers/basic.png'


def test_preview_rejects_invalid_structure(tmp_path):
    walker = tmp_path / 'bad.toml'
    walker.write_text('name="broken"\nbodies=[]\n')
    with pytest.raises(ValueError):
        preview_walker(walker, tmp_path / 'bad.png')
    assert not (tmp_path / 'bad.png').exists()
