import json
from pathlib import Path

import pytest

from maton.config import Config, load_run, save_run


def test_root_configs_and_resolved_snapshot(tmp_path):
    config = Config.load("configs/smoke.toml")
    assert len(config.resolved()["curriculum"]["levels"]) == 6
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
    walker = Path("configs/walkers/basic.toml").resolve()
    reference = Path("configs/curricula/walker.toml").resolve()
    path.write_text(f'walker = "{walker}"\ncurriculum = "{reference}"\n' + contents)
    with pytest.raises((ValueError, TypeError)):
        Config.load(path)


def test_walker_reference_is_relative_and_snapshot_is_self_contained(tmp_path):
    source = Path("configs/walkers/quadruped.toml")
    structure = tmp_path / "structure.toml"
    structure.write_text(source.read_text())
    curriculum = Path("configs/curricula/walker.toml").resolve()
    job = tmp_path / "job.toml"
    job.write_text(f'walker = "structure.toml"\ncurriculum = "{curriculum}"\n')
    config = Config.load(job)
    assert config.environment().action_size == 8
    output = tmp_path / "run"
    save_run(config, "ppo", output)
    structure.unlink()
    restored, _ = load_run(output)
    assert restored.resolved() == config.resolved()
    assert restored.environment().action_size == 8


def test_job_requires_walker(tmp_path):
    job = tmp_path / "job.toml"
    job.write_text('curriculum = "unused.toml"\n')
    with pytest.raises(ValueError, match="walker"):
        Config.load(job)
