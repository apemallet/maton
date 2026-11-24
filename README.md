# Maton

A planar, four-motor MJX biped learning procedural obstacle courses with upstream
DreamerV3 or an Equinox PPO baseline. Observations combine lidar, joint/torso
kinematics, previous actions, and foot contacts.

```sh
uv sync
just test

# Tiny integration checks; choose fresh output directories.
just smoke ppo data/ppo-smoke
just smoke dreamer data/dreamer-smoke

# Small development budgets; neither is enough to learn robust walking.
just train ppo data/ppo-dev
just train dreamer data/dreamer-dev

# Evaluate a saved checkpoint and render its trajectory.
just rollout data/ppo-dev data/ppo-eval
just render data/ppo-eval/episode-0000
uv run maton rollout data/ppo-dev --checkpoint data/ppo-dev/checkpoints/step-00000128.eqx \
  --output data/ppo-checkpoint-eval --episodes 3 --seed 2000
```

Edit [configs/walker.toml](configs/walker.toml) for course difficulty, sensors,
simulation, network sizes, and training budgets. Jobs reference the separate
[six-level curriculum](configs/curricula/walker.toml), starting with flat walking;
see [advancement rules](docs/curriculum.md). [configs/smoke.toml](configs/smoke.toml)
uses one terrain slot and minimal updates. Dreamer defaults to CPU; set
`dreamer.platform = "gpu"` for a GPU run.
See the [configuration reference](docs/config.md) for every option,
default, unit, and constraint.

Runs live in `data/` and retain resolved configuration, metrics, and numbered
checkpoints. Evaluation stores exact physics trajectories and the sampled course
as MuJoCo XML; videos can be rebuilt without loading the agent. Training is an
explicit CLI/just recipe, not a Snakemake artifact.

See [environment and replay](docs/overview.md), [DreamerV3](docs/dreamer.md), and
[PPO](docs/ppo.md).
