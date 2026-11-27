"""Training curves: episode return and curriculum level against environment transitions."""

import csv
import json
from pathlib import Path

import numpy as np

# Reference data-viz palette (light): one series per panel, recessive grid and axes.
SURFACE, INK, MUTED, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def training_curve(folder: Path) -> dict[str, np.ndarray]:
    """Points with a completed-episode return, normalized across the trainers' metric files.
    PPO: one CSV row per update. SAC/TD-MPC2: one JSONL row per log window.
    Dreamer: one JSONL row per episode, interleaved with loss rows (skipped)."""
    steps, returns, levels = [], [], []
    if (folder / "metrics.csv").exists():
        with (folder / "metrics.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if int(row["episodes"]) > 0:
                    steps.append(int(row["steps"]))
                    returns.append(float(row["episode_return_mean"]))
                    levels.append(int(row["curriculum_level"]))
    else:
        for line in (folder / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("episode_return") is not None:
                steps.append(int(row.get("steps", row.get("step"))))
                returns.append(float(row["episode_return"]))
                levels.append(int(row["curriculum"]["level"]))
    return dict(steps=np.array(steps, int), returns=np.array(returns, float), levels=np.array(levels, int))


def _subtitle(folder: Path, window: int) -> str:
    parts = []
    path = folder / "progress.json"
    if path.exists():
        log = json.loads(path.read_text(encoding="utf-8"))
        devices = sorted({session["device"] for session in log["sessions"]})
        sessions = len(log["sessions"])
        parts += [log["algorithm"], f"{log['hours']:.2f} h over {sessions} session{'s' if sessions != 1 else ''} "
                                    f"on {' + '.join(devices)}"]
    parts.append("dots: raw" + (f"; line: mean of {window}" if window > 1 else ""))
    return "  ·  ".join(parts)


def plot_training(folder: Path, output: Path, title: str | None = None) -> Path:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    curve = training_curve(folder)
    x, y = curve["steps"], curve["returns"]
    if not len(x):
        raise ValueError(f"{folder}: no completed episodes to plot")
    window = max(1, len(y) // 25)

    figure = Figure(figsize=(8, 5.2), dpi=150, facecolor=SURFACE)
    FigureCanvasAgg(figure)
    top, bottom = figure.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=(2, 1), hspace=0.12))
    top.scatter(x, y, s=8, color=SERIES, alpha=0.22, linewidths=0)
    if window > 1:
        top.plot(x[window - 1:], np.convolve(y, np.ones(window) / window, mode="valid"), color=SERIES, lw=2)
    else:
        top.plot(x, y, color=SERIES, lw=2)
    bottom.step(x, curve["levels"] + 1, where="post", color=SERIES, lw=2)  # 1-based, as in `just status`

    for axes in (top, bottom):
        axes.set_facecolor(SURFACE)
        axes.grid(axis="y", color=GRID, lw=0.8)
        axes.set_axisbelow(True)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color(GRID)
        axes.tick_params(colors=MUTED, labelsize=9, length=0)
    top.set_ylabel("Episode return", color=MUTED, fontsize=10)
    bottom.set_ylabel("Curriculum level", color=MUTED, fontsize=10)
    bottom.set_xlabel("Environment transitions", color=MUTED, fontsize=10)
    bottom.yaxis.set_major_locator(MaxNLocator(integer=True))
    bottom.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k" if value else "0"))
    bottom.set_xlim(left=0)
    top.set_title(title or folder.parent.name, loc="left", color=INK, fontsize=13, fontweight="bold", pad=22)
    top.text(0, 1.02, _subtitle(folder, window), transform=top.transAxes, color=MUTED, fontsize=9)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor=SURFACE, bbox_inches="tight")
    return output


def main() -> None:
    import argparse
    import tempfile
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Published train/ (or progress/) directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Stage beside the destination; publish only a complete image.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".plot-", dir=args.output.parent) as temporary:
        stage = plot_training(args.train, Path(temporary) / args.output.name)
        stage.replace(args.output)


if __name__ == "__main__":
    main()
