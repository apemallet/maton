# Initial CPU checks

Using `configs/walker.toml`, seed 0:

| Agent | Budget | Updates | Result |
| --- | --- | --- | --- |
| PPO | 256 transitions | 4 | Finite loss/gradients; five completed training episodes |
| DreamerV3 | 128 decisions | 8 | Finite metrics/parameters; two completed training episodes |

Artifacts: `data/ppo-dev`, `data/dreamer-dev`. Fresh checkpoint evaluations at
course seed 1000 are saved under `data/ppo-eval` and `data/dreamer-eval`.
PPO displaced −0.72 m before falling; Dreamer displaced +0.15 m before falling.
These short runs validate integration and stability, not learned walking.

Also checked one-slot smoke runs, batched PPO time-limit resets, seeded Dreamer
checkpoint reload, exact saved course geometry, and headless MP4 rendering.
Increase training budgets only after reviewing losses, returns, and videos.
