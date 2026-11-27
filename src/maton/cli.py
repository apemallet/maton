"""Training and retrospective episode inspection."""

import argparse
import json
from pathlib import Path

from maton.algorithms import algorithm as lookup
from maton.config import load_run, save_run
from maton.jobs import Job
from maton.sessions import Pause
from maton.training import TrainingConfig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Planar MJX obstacle walker")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train the job's algorithm")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True,
                       help="New run directory, or an existing one with matching settings to resume")
    train.add_argument("--hours", type=float, help="Pause after this much wall-clock time")
    train.add_argument("--steps", type=int, help="Pause once this many total transitions are reached")
    rollout = commands.add_parser("rollout", help="Evaluate a saved checkpoint on fresh courses")
    rollout.add_argument("run", type=Path)
    rollout.add_argument("--checkpoint", type=Path)
    rollout.add_argument("--output", type=Path, required=True)
    rollout.add_argument("--seed", type=int, default=1000)
    rollout.add_argument("--episodes", type=int, default=1)
    rollout.add_argument("--level", type=int, help="Curriculum level (1-based); default: checkpoint's level")
    render = commands.add_parser("render", help="Render a saved episode to MP4")
    render.add_argument("episode", type=Path)
    render.add_argument("--output", type=Path)
    preview = commands.add_parser("view-walker", help="Save a labeled PNG preview of a walker TOML")
    preview.add_argument("walker", type=Path)
    preview.add_argument("--output", type=Path, help="PNG path; default: data/walkers/<walker-name>.png")
    preview.add_argument("--neutral", action="store_true", help="Pose motor joints at their neutral targets")
    args = parser.parse_args(argv)

    if args.command == "train":
        job = Job.load(args.config)
        config = job.config()
        env = config.environment()
        if (args.output / "run.json").exists():
            if load_run(args.output).resolved() != config.resolved():
                parser.error(f"{args.output} was trained with different settings; use a new output")
        else:
            save_run(config, args.output)
        trainer = lookup(config.algorithm).load()
        pause = Pause(args.hours, args.steps)
        with pause.catching_interrupt():
            result = trainer.train(env, config.trainer_config(), TrainingConfig(**config.training),
                                   args.output, config.seed, pause=pause)
        print(json.dumps(result, indent=2))
    elif args.command == "rollout":
        from maton.replay import record, summarize_episodes
        config = load_run(args.run)
        if args.episodes < 1:
            parser.error("--episodes must be positive")
        spec = lookup(config.algorithm)
        agent = spec.load().load_agent(args.checkpoint or spec.checkpoint(args.run))
        env = config.environment()
        level = args.level - 1 if args.level is not None else agent.curriculum_level
        if not 0 <= level < len(env.curriculum.levels):
            parser.error("--level is outside this curriculum")
        episodes = []
        for index in range(args.episodes):
            agent.reset()
            episodes.append(record(env, agent, args.output / f"episode-{index:04d}", args.seed + index, level))
        print(json.dumps(summarize_episodes(episodes, args.output / "summary.json"), indent=2))
    elif args.command == "view-walker":
        from maton.preview import preview_walker
        print(preview_walker(args.walker, args.output, neutral=args.neutral))
    else:
        from maton.replay import render_episode
        print(render_episode(args.episode, args.output))
