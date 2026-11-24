import json

import pytest

from maton.config import Config, load_run, save_run


def test_root_configs_and_resolved_snapshot(tmp_path):
    config = Config.load("configs/smoke.toml")
    assert config.resolved()["terrain"]["slots"] == 1
    assert config.resolved()["training"]["steps"] == 16
    output = tmp_path / "run"
    save_run(config, "ppo", output)
    restored, algorithm = load_run(output)
    assert algorithm == "ppo"
    assert restored.resolved() == config.resolved()
    assert json.loads((output / "run.json").read_text())["env"]["lidar_count"] == 9
    with pytest.raises(FileExistsError):
        save_run(config, "ppo", output)
    Config.load("configs/walker.toml")


@pytest.mark.parametrize("contents", ["[unknown]\nx=1", "[env]\nlidar_counts=9",
                                     "[terrain]\nblock_frequency=0.9\ngap_frequency=0.9",
                                     "[ppo]\nupdates=4", "[dreamer]\ncheckpoint_every=1",
                                     "[training]\nsteps=0"])
def test_reject_bad_config(tmp_path, contents):
    path = tmp_path / "bad.toml"
    path.write_text(contents)
    with pytest.raises((ValueError, TypeError)):
        Config.load(path)
