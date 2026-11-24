"""Training and retrospective episode inspection."""

import argparse
import json
from pathlib import Path

from maton.config import Config, load_run, save_run
from maton.training import TrainingConfig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Planar MJX obstacle walker")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train a small PPO or DreamerV3 agent")
    train.add_argument("--algorithm", choices=("ppo", "dreamer"), default="ppo")
    train.add_argument("--config", type=Path, default=Path("configs/walker.toml"))
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--seed", type=int)
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
    args = parser.parse_args(argv)

    if args.command == "train":
        config = Config.load(args.config)
        if args.seed is not None:
            config.seed = args.seed
        env = config.environment()
        save_run(config, args.algorithm, args.output)
        training = TrainingConfig(**config.training)
        if args.algorithm == "ppo":
            from maton.ppo import PPOConfig, train as train_ppo
            result = train_ppo(env, PPOConfig(**config.ppo), training, args.output, config.seed)
        else:
            from maton.dreamer import DreamerConfig, train as train_dreamer
            result = train_dreamer(env, DreamerConfig(**config.dreamer), training, args.output, config.seed)
        print(json.dumps(result, indent=2))
    elif args.command == "rollout":
        from maton.replay import record, summarize_episodes
        config, algorithm = load_run(args.run)
        if args.episodes < 1:
            parser.error("--episodes must be positive")
        if algorithm == "ppo":
            from maton.ppo import load_policy, action
            policy = load_policy(args.checkpoint or args.run / "policy.eqx")
            level = policy.curriculum_level
            act = lambda obs: action(policy, obs)
        else:
            from maton.dreamer import load_policy
            act = load_policy(args.checkpoint or args.run / "dreamer.pkl")
            level = act.curriculum_level
        env = config.environment()
        level = args.level - 1 if args.level is not None else level
        if not 0 <= level < len(env.curriculum.levels):
            parser.error("--level is outside this curriculum")
        episodes = []
        for index in range(args.episodes):
            if hasattr(act, "reset"):
                act.reset()
            episodes.append(record(env, act, args.output / f"episode-{index:04d}", args.seed + index, level))
        print(json.dumps(summarize_episodes(episodes, args.output / "summary.json"), indent=2))
    else:
        from maton.replay import render_episode
        print(render_episode(args.episode, args.output))
