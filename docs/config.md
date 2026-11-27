# Configuration reference

Jobs live in `configs/jobs/`: `basic-{ppo,dreamer}-s0` preserves the original small
job; `biped-{ppo,dreamer,tdmpc2,sac}-s0` trains walking; `smoke-*` checks integration.
See [workflow](workflow.md) for names, artifact paths, evaluations, and rebuild policy.

## Root

| Option | Default | Meaning                                                                                                  |
| ------ | ------- | -------------------------------------------------------------------------------------------------------- |
| `algorithm` | required | A registered [trainer](algorithms.md); include only that trainer’s table. |
| `walker` | required | Walker TOML path, relative to the job TOML. Resolved structure is embedded in `run.json`; see [walker structure](walker.md). |
| `curriculum` | required | Curriculum TOML path, relative to the job TOML. Resolved levels are embedded in `run.json`. |
| `seed` | `0`     | Nonnegative integer seed for policy initialization, sampling, and training courses. |

## `[env]`: physics, observations, reward

| Option           | Default | Meaning / constraints                                                                                                                                                     |
| ---------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `timestep`       | `0.01`  | Seconds per physics step; positive.                                                                                                                                       |
| `frame_skip`     | `2`     | Physics steps per action; positive integer.                                                                                                                               |
| `episode_length` | `500`   | Maximum actions per episode before time-limit truncation; positive integer.                                                                                               |
| `lidar_count`    | `9`     | Number of torso-relative range rays; positive integer. Changes observation size.                                                                                          |
| `lidar_range`    | `3.0`   | Maximum range in metres; positive. Returned ranges are normalized to `[0, 1]`.                                                                                            |
| `reset_noise`    | `0.01`  | Uniform initial position perturbation in `[-reset_noise, +reset_noise]` for all coordinates; metres for slides, radians for hinges. Nonnegative. Initial velocities zero. |
| `progress_mode` | `"distance"` | `distance`: forward position potential; `upright`: position potential multiplied by height and uprightness scores. |
| `progress_scale` | `1.0`   | Positive multiplier of forward displacement reward.                                                                                                                       |
| `control_cost`   | `0.001` | Nonnegative penalty multiplier for final-substep mean squared normalized motor torque.                                                                                                 |
| `control_mode` | `"torque"` | `torque` commands normalized torque; `position` commands target angles with PD control each physics step. |
| `finish_reward` | `0.0` | Nonnegative bonus paid once for upright course completion. |
| `angle_harmonics` | `2` | Fourier harmonics `K` for angle and angular-rate observations; observation size is `4 * (1 + hinges) * K + 2 + contacts + lidar_count`. Positive integer. |
| `friction` | `0.9` | MuJoCo sliding friction of every geom (contact uses the larger of the pair); positive, below 10. |
| `upright_reward` | `0.0` | Optional posture reward per simulated second; nonnegative. |

Position mode uses each joint's motor target range, neutral, gains, and torque cap
from the walker file. Negative/positive actions interpolate from neutral toward
the lower/upper target. Torque is `clip(kp * (target - angle) - kd * velocity, ±torque_limit)`.

Action duration is `timestep * frame_skip`; maximum simulated episode duration is
`episode_length * timestep * frame_skip`. Successful course completion
can end an episode earlier; falls may recover until the time limit. Increasing `frame_skip` holds each command longer.

Reward per action is `Phi(next) - Phi(previous) - control_cost * mean(normalized_torque²)`
plus `upright_reward * action_duration * posture_score` and a successful `finish_reward`.
The posture score multiplies `clip((root_height / spawn_height - 0.4) / 0.4, 0, 1)` and
`clip(cos(pitch), 0, 1)`.

- `distance`: `Phi = progress_scale * min(x, goal_x)`.
- `upright`: the same field multiplied by `posture_score`. Falling loses accumulated
  potential; recovery restores it. At a fixed posture, moving forward increases potential.

Both fields saturate at the finish to prevent runoff rewards. The effort penalty
is per action. Observation content is the same in both progress modes.
Rays span downward to 15° above forward.

## Curriculum TOML: levels and procedural course

Jobs reference a separate curriculum instead of declaring `[terrain]`. See
[curricula/walker.toml](../configs/curricula/walker.toml) and [curriculum rules](curriculum.md).
The curriculum has shared `[terrain]` defaults and ordered `[[levels]]` entries
with a `name`, optional `successes_required`, and `[levels.terrain]` overrides.
Root `successes_required` defaults to five per level.

All lengths are in metres. Each slot independently selects a block, gap, or flat
floor. Frequencies are probabilities **per slot**, not obstacles per metre.
Actual event height/depth and width are independently sampled from **50–100%** of
their configured maxima. Terrain is resampled on each episode reset.

| Option            | Default | Meaning / constraints                                                                                           |
| ----------------- | ------- | --------------------------------------------------------------------------------------------------------------- |
| `slots`           | `6`     | Number of potential obstacle locations; positive integer. More slots increase course length and collision cost. |
| `spacing`         | `1.5`   | Length of each slot; positive. Obstacles are centred within slots.                                              |
| `start_length`    | `2.0`   | Clear floor from the start to the first slot; at least `1.0`.                                                   |
| `block_frequency` | `0.2`   | Probability of a raised block in a slot; `[0, 1]`.                                                              |
| `block_height`    | `0.12`  | Maximum block elevation above the floor; nonnegative.                                                           |
| `block_width`     | `0.35`  | Maximum block width; strictly between zero and `spacing`.                                                       |
| `gap_frequency`   | `0.1`   | Probability of a floor recess in a slot; `[0, 1]`.                                                              |
| `gap_height`      | `0.15`  | Maximum recess depth below the floor; finite and nonnegative; floor geometry extends below the deepest configured gap.                         |
| `gap_width`       | `0.3`   | Maximum recess width; strictly between zero and `spacing`.                                                      |

`block_frequency + gap_frequency` must be at most one; the remaining probability
produces flat floor. Gaps have physical bottoms and vertical sides; entering a gap
does not trigger termination. Deep gaps rely on physical difficulty and timeout. Levels share `slots`, `spacing`, and `start_length`
to retain fixed MJX shapes. Course length forward
from the start is `start_length + slots * spacing`.

## `[training]`: shared trainer settings

| Option | Default | Meaning / constraints |
| --- | --- | --- |
| `steps` | `256` | Total actual environment transitions; positive integer. PPO requires divisibility by `ppo.num_envs`. |
| `learning_rate` | `0.0001` | Finite positive optimizer learning rate for either agent. |
| `checkpoint_every` | `64` | Checkpoint interval in actual environment transitions; positive integer. Final weights are always saved. |

One transition is one action held for `env.frame_skip` physics steps. Counts sum
across PPO environments; Dreamer reset bookkeeping does not consume the budget.
PPO shortens its final rollout to meet the budget exactly, allowing a smaller
final minibatch. Checkpoints are written after an update crosses an interval
threshold; PPO may save later than the threshold by up to one rollout. Filenames
and checkpoint metadata record the actual transition count. Multiple thresholds
crossed in one update produce one snapshot.

## `[ppo]`: baseline trainer

| Option | Default | Meaning / constraints |
| --- | --- | --- |
| `num_envs` | `2` | Batched MJX environments collecting transitions; positive integer. |
| `rollout_steps` | `32` | Maximum actions per environment per rollout/update; positive integer. |
| `epochs` | `2` | Optimization passes through each collected batch; positive integer. |
| `minibatch_size` | `32` | Samples per optimizer step; positive integer dividing a full `num_envs * rollout_steps` batch. |
| `hidden_size` | `64` | Width of the actor's and critic's two hidden layers; positive integer. |
| `gamma` | `0.99` | Reward discount per action; `[0, 1]`. |
| `gae_lambda` | `0.95` | Advantage trace factor; `[0, 1]`. Larger values use longer traces. |
| `clip_ratio` | `0.2` | Positive probability-ratio clipping radius around one. |
| `value_coefficient` | `0.5` | Critic loss weight; use nonnegative values. |
| `entropy_coefficient` | `0.001` | Pre-tanh Gaussian entropy bonus weight; use nonnegative values. |
| `max_grad_norm` | `0.5` | Positive global gradient-norm clipping threshold. |

Rollout updates: `ceil(training.steps / (num_envs * rollout_steps))`. Optimizer
steps per full rollout: `epochs * num_envs * rollout_steps / minibatch_size`.
Time limits bootstrap the critic; true terminations do not. See [PPO](ppo.md).

## `[dreamer]`: upstream DreamerV3 trainer

One MJX environment trains the upstream world model, imagination policy, and
value function. Unexposed upstream settings use the bundled, pinned defaults.

| Option | Default | Meaning / constraints |
| --- | --- | --- |
| `batch_size` | `1` | Replay sequences per training update; positive integer. |
| `batch_length` | `8` | Replay records per sampled sequence, including episode boundaries; integer at least two. |
| `train_every` | `16` | Actual environment transitions between updates once replay has enough sequences; positive integer. |
| `replay_size` | `2048` | Replay record capacity; at least `batch_size * batch_length`. |
| `hidden` | `32` | RSSM/encoder/decoder/head width; positive integer divisible by four. |
| `deter` | `64` | Deterministic recurrent state size; positive integer divisible by four. |
| `stoch` | `4` | Categorical stochastic latent variables; positive integer. |
| `classes` | `4` | Categories per latent variable; positive integer. |
| `imag_length` | `5` | Imagined rollout horizon; positive integer. |
| `policy_minstd` / `policy_maxstd` | `0.1` / `1.0` | Bounds on the actor's action standard deviation; positive, ordered. Position control needs far less than upstream's range. |
| `optimizer_warmup` | `0` | Optimizer warmup updates; nonnegative integer. |
| `learning_rate` | unset | Overrides `[training].learning_rate` for Dreamer only; upstream uses `4e-5`. |
| `platform` | `"cpu"` | Dreamer JAX platform, typically `"cpu"` or `"gpu"`; hardware must be available. |

Larger batches, sequences, latent states, and imagination horizons cost more
memory and computation. `platform` configures Dreamer; MJX uses JAX's default
device selection. See [Dreamer integration](dreamer.md).

## Off-policy trainers: shared options

`[tdmpc2]` and `[sac]` accept these, plus their own options below. See the [driver](offpolicy.md).

| Option | Default | Meaning |
| --- | --- | --- |
| `num_envs` | `4` | Parallel environments; must divide `training.steps`. |
| `batch_size` | `256` | Replay windows per update. |
| `buffer_size` | `1_000_000` | Replay capacity in transitions (capped at the budget). |
| `seed_steps` | unset | Random-action transitions before learning; unset: `max(1000, 5 × episode_length)`. |
| `update_ratio` | `1.0` | Gradient updates per transition. |
| `log_every` | `2048` | Transitions between `metrics.jsonl` rows. |

## `[tdmpc2]`

Defaults are the official 5M configuration; see [TD-MPC2](tdmpc2.md).

| Group | Options (defaults) |
| --- | --- |
| Model | `latent_dim` 512, `mlp_dim` 512, `enc_dim` 256, `num_enc_layers` 2, `num_q` 5, `simnorm_dim` 8, `dropout` 0.01, `num_bins` 101, `vmin`/`vmax` ±10 (symlog), `episodic` true |
| Planning | `mpc` true, `horizon` 3, `iterations` 6, `num_samples` 512, `num_elites` 64, `num_pi_trajs` 24, `min_std` 0.05, `max_std` 2, `temperature` 0.5 |
| Losses | `rho` 0.5, `consistency_coef` 20, `reward_coef` 0.1, `value_coef` 0.1, `termination_coef` 1, `entropy_coef` 1e-4, `log_std_min`/`log_std_max` −10/2 |
| Optimizer | `[training].learning_rate` (official 3e-4), `enc_lr_scale` 0.3, `grad_clip_norm` 20, `tau` 0.01, `discount` unset (heuristic) |

## `[sac]`

| Option | Default | Meaning |
| --- | --- | --- |
| `hidden_size` / `hidden_layers` | `256` / `2` | Actor and critic MLP shape. |
| `num_q` | `2` | Critic ensemble size; targets use the minimum. |
| `critic_norm` | `true` | LayerNorm in critic hidden layers. |
| `initial_alpha` | `1.0` | Starting entropy temperature; tuned automatically. |
| `target_entropy` | unset | Defaults to `-action_size`. |
| `log_std_min` / `log_std_max` | `-5` / `2` | Actor log-std bounds. |
| `tau` | `0.005` | Target critic update rate. |
| `discount` | unset | Episode-length heuristic, as TD-MPC2. |

## Low-level CLI

Snakemake calls these commands. Use workflow targets for managed artifacts;
CLI output directories must be new. Evaluation settings belong in job TOML
for managed runs.


| Command   | Option         | Default / purpose                                                               |
| --------- | -------------- | ------------------------------------------------------------------------------- |
| `train`   | `--config`     | Required job TOML path; algorithm and seed come from this file.                                         |
| `train`   | `--output`     | Required; new run directory.                                                    |
| `rollout` | `run`          | Required positional run directory containing `run.json`.                        |
| `rollout` | `--checkpoint` | Optional specific snapshot; otherwise final run weights.                        |
| `rollout` | `--output`     | Required evaluation directory; each episode subdirectory must be new.           |
| `rollout` | `--seed`       | `1000`; course seed for the first episode, incremented for subsequent episodes. |
| `rollout` | `--level` | Optional 1-based curriculum level; defaults to the saved checkpoint level. |
| `rollout` | `--episodes`   | `1`; positive number of episodes.                                               |
| `render`  | `episode`      | Required positional saved episode directory.                                    |
| `render`  | `--output`     | Defaults to `episode/episode.mp4`; destination video path.                      |

Rollout writes `summary.json` with success rate, distance, return, upright fraction,
and individual seeded results. Upright means root height ≥ half the configured spawn height and pitch within
1.4 radians, matching the finish condition.
