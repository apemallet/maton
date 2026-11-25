# Historical experiments

Paths below refer to pre-workflow runs and their original configuration names.
New runs use [named jobs](workflow.md).

# Initial CPU checks

Original `configs/walker.toml`, seed 0, before shared `[training]` settings:

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

# Level-1 tuning (walker.toml, CPU)

Success = deterministic policy finishes the 4 m flat course upright; 10 held-out
seeds (`rollout --level 1`). Shared env: position control, `progress_scale=20`,
`upright_reward=0.1`, `finish_reward=25`.

| Trainer | Change | Seeds | Level-1 finishes |
| --- | --- | --- | --- |
| PPO | original (`upright_reward=1`, `initial_std=0.03`) | 0 | 0/5 — stood still (return ≈ standing ceiling) |
| PPO | `initial_std=0.2`, `entropy=0`, `upright_reward=0.1`, `episode_length=512` | 0, 1, 2 | 9/10, 6/10, 10/10 |
| PPO | same, `initial_std=0.4` | 1, 2 | 0/10, 0/10 (relies on noise) |
| PPO | same, `episode_length` 640 / 800 | 0–2 / 1–2 | 0/10 on 5 of 6 runs; collapses prone |
| Dreamer | upstream std (0.1–1.0), lr 3e-4 | 0 | 0/10 — flails, ≈3% upright |
| Dreamer | `policy_maxstd=0.3`, `learning_rate=1e-4`, `episode_length=800` | 0 (×2) | 7/10, 9/10 |
| Dreamer | same, `learning_rate=4e-5` | 0 | 0/10 at 262k |
| Dreamer | same as 1e-4 row, `episode_length=512` | 0 | 0/10 — upright but ~0.09 m/s, too slow |
| Dreamer | `deter=64`, `stoch=classes=16`, 512 | 0 | 0/10 — falls backward |

Friction / motors (512 steps, Dreamer seed 0, 393k):

| Change | Level-1 finishes |
| --- | --- |
| `friction=1.6` | 10/10 |
| `torque_limit=40, kp=60, kd=4` | 4/10 |
| both | 0/10 |

Seed check of `friction=1.6` from `walker.toml` (524k steps):

| Trainer | Seeds 0 / 1 / 2 |
| --- | --- |
| PPO | 10/10, 0/10, 0/10 |
| Dreamer | 10/10 (scratch run), 0/10, 0/10 |

`friction=2.5` / `4.0` (seed 1): 0/10 for PPO and Dreamer. Results so far are
seed-0 specific; judge changes on at least three seeds.

Fourier observation (`angle_harmonics=2`; no height, course position, or previous
action), `walker.toml` as committed (512 steps, `friction=1.6`), 524k steps:

| Trainer | Seeds 0 / 1 / 2 |
| --- | --- |
| PPO | 0/10, 0/10, 0/10 (falls; 8–22% upright) |
| Dreamer | 10/10 (finishes in ~154 steps), 0/10, 0/10 (upright but slow, 1.4 / 2.8 m) |

Open issue: PPO needs `episode_length=512`; Dreamer needs ≈800. Falls never
terminate, so fallen walkers waste most of a long episode. `configs/walker.toml`
currently uses 512 (PPO passes; Dreamer does not).
