# Maton

Configurable planar MJX walkers learning procedural obstacle courses with upstream
DreamerV3 or an Equinox PPO baseline. Observations combine lidar, Fourier-encoded
joint/torso angles and rates, root velocity, and foot contacts.

```sh
uv sync
just test

# Default: PPO smoke training → held-out evaluation → video.
just smoke
just job smoke-dreamer

# Build the full experiment, or request one asset and its dependencies.
just job biped-ppo-s0
just build data/jobs/biped-ppo-s0/render/level-1/episode-0000.mp4 -n
just view-walker quadruped neutral
```

Experiments live in [configs/jobs](configs/jobs). A job selects an algorithm,
seed, [walker](docs/walker.md), [curriculum](docs/curriculum.md), training settings,
and named evaluations. Dreamer defaults to CPU; set `dreamer.platform = "gpu"`
for a GPU job.

Snakemake owns all generated artifacts under `data/`. Completed training is
protected; changed experiments receive new job names. Evaluations save exact
physics trajectories, sampled course XML, and summary metrics. Videos rebuild
from those trajectories without loading a policy.

See [workflow and naming](docs/workflow.md) for targets, rebuild policy, and
execution profiles; [configuration](docs/config.md) lists model and environment options.

See [environment and replay](docs/overview.md), [DreamerV3](docs/dreamer.md), and
[PPO](docs/ppo.md).
