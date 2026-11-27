# Maton

Configurable planar MJX walkers learning procedural obstacle courses with TD-MPC2,
upstream DreamerV3, or Equinox SAC and PPO baselines. Observations combine lidar, Fourier-encoded
joint/torso angles and rates, root velocity, and foot contacts.

```sh
uv sync
just test

# Default: PPO smoke training → held-out evaluation → video.
just smoke
just job smoke-dreamer

# Train in sessions: 2 hours now, resume later or on another machine.
just train biped-tdmpc2-s0 2
just status
just share biped-tdmpc2-s0

# Build the full experiment, or request one asset and its dependencies.
just job biped-ppo-s0
just build data/jobs/biped-ppo-s0/render/level-1/episode-0000.mp4 -n
just view-walker quadruped neutral
just talk                       # status deck: talk/slides.md → data/talk/
```

Experiments live in [configs/jobs](configs/jobs). A job selects an algorithm,
seed, [walker](docs/walker.md), [curriculum](docs/curriculum.md), training settings,
and named evaluations. On Linux, `uv sync` installs CUDA 12 JAX (only an NVIDIA driver is
needed); `just devices` should list the GPU. PPO, SAC, and TD-MPC2 use it automatically;
Dreamer defaults to CPU, so set `dreamer.platform = "gpu"`.

Snakemake owns all generated artifacts under `data/`. Completed training is
protected; changed experiments receive new job names. Evaluations save exact
physics trajectories, sampled course XML, and summary metrics. Videos rebuild
from those trajectories without loading a policy.

See [workflow and naming](docs/workflow.md) for targets, rebuild policy, and
execution profiles; [configuration](docs/config.md) lists model and environment options.

See [environment and replay](docs/overview.md) and the [trainer registry](docs/algorithms.md):
[TD-MPC2](docs/tdmpc2.md), [SAC](docs/sac.md), [DreamerV3](docs/dreamer.md), and [PPO](docs/ppo.md).
