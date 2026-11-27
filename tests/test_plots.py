import csv
import json

import pytest

from maton.plots import plot_training, training_curve

LEVEL = {"level": 1, "name": "x", "successes": 0, "successes_required": 5, "complete": False}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_offpolicy_windows_skip_rows_without_episodes(tmp_path):
    write_jsonl(tmp_path / "metrics.jsonl", [
        {"steps": 2048, "episode_return": None, "curriculum": LEVEL},
        {"steps": 4096, "episode_return": 3.5, "curriculum": LEVEL, "pi_loss": 0.1}])
    curve = training_curve(tmp_path)
    assert curve["steps"].tolist() == [4096] and curve["returns"].tolist() == [3.5]
    assert curve["levels"].tolist() == [1]


def test_dreamer_keeps_episode_rows_and_skips_loss_rows(tmp_path):
    write_jsonl(tmp_path / "metrics.jsonl", [
        {"step": 4, "updates": 1, "loss/dyn": 2.0},
        {"step": 512, "episode_return": 1.0, "episodes": 1, "success": False, "curriculum": LEVEL}])
    assert training_curve(tmp_path)["steps"].tolist() == [512]


def test_ppo_rows_need_completed_episodes(tmp_path):
    with (tmp_path / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, ["steps", "episodes", "episode_return_mean", "curriculum_level"])
        writer.writeheader()
        writer.writerow(dict(steps=2048, episodes=0, episode_return_mean=0.0, curriculum_level=0))
        writer.writerow(dict(steps=4096, episodes=3, episode_return_mean=2.0, curriculum_level=2))
    curve = training_curve(tmp_path)
    assert curve["steps"].tolist() == [4096] and curve["levels"].tolist() == [2]


def test_plot_writes_png_and_rejects_empty_metrics(tmp_path):
    write_jsonl(tmp_path / "metrics.jsonl", [
        {"steps": 2048 * i, "episode_return": float(i), "curriculum": LEVEL} for i in range(1, 60)])
    output = plot_training(tmp_path, tmp_path / "plots" / "training.png")
    assert output.read_bytes().startswith(b"\x89PNG")
    write_jsonl(tmp_path / "metrics.jsonl", [{"steps": 2048, "episode_return": None, "curriculum": LEVEL}])
    with pytest.raises(ValueError, match="no completed episodes"):
        plot_training(tmp_path, tmp_path / "empty.png")
