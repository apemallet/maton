# Configuration reference

## Root

| Option | Default | Meaning                                                                                                  |
| ------ | ------- | -------------------------------------------------------------------------------------------------------- |
| `seed` | `0`     | Integer seed for policy initialization, sampling, and training courses. CLI `train --seed` overrides it. |

## `[env]`: physics, observations, reward

| Option           | Default | Meaning / constraints                                                                                                                                                     |
| ---------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `timestep`       | `0.01`  | Seconds per physics step; positive.                                                                                                                                       |
| `frame_skip`     | `2`     | Physics steps per action; positive integer.                                                                                                                               |
| `episode_length` | `500`   | Maximum actions per episode before time-limit truncation; positive integer.                                                                                               |
| `lidar_count`    | `9`     | Number of torso-relative range rays; positive integer. Changes observation size.                                                                                          |
| `lidar_range`    | `3.0`   | Maximum range in metres; positive. Returned ranges are normalized to `[0, 1]`.                                                                                            |
| `reset_noise`    | `0.01`  | Uniform initial position perturbation in `[-reset_noise, +reset_noise]` for all coordinates; metres for slides, radians for hinges. Nonnegative. Initial velocities zero. |
| `progress_scale` | `1.0`   | Positive multiplier of forward displacement reward.                                                                                                                       |
| `control_cost`   | `0.001` | Nonnegative penalty multiplier for mean squared normalized motor actions.                                                                                                 |

Action duration is `timestep * frame_skip`; maximum simulated episode duration is
`episode_length * timestep * frame_skip`. Falls and course completion can end an
episode earlier. Increasing `frame_skip` holds each command longer.

Reward per action is `progress_scale * (next_x - x) - control_cost * mean(action²)`.
The effort penalty is per action, without an additional duration multiplier.
Observation size is `20 + lidar_count`. Rays span downward to 15° above forward.

## `[terrain]`: procedural course

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
| `gap_height`      | `0.15`  | Maximum recess depth below the floor; `[0, 1)` because the floor bed is one metre deep.                         |
| `gap_width`       | `0.3`   | Maximum recess width; strictly between zero and `spacing`.                                                      |

`block_frequency + gap_frequency` must be at most one; the remaining probability
produces flat floor. Gaps have bottoms and vertical sides. Course length forward
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
| `platform` | `"cpu"` | Dreamer JAX platform, typically `"cpu"` or `"gpu"`; hardware must be available. |

Larger batches, sequences, latent states, and imagination horizons cost more
memory and computation. `platform` configures Dreamer; MJX uses JAX's default
device selection. See [Dreamer integration](dreamer.md).

## CLI options outside TOML

| Command   | Option         | Default / purpose                                                               |
| --------- | -------------- | ------------------------------------------------------------------------------- |
| `train`   | `--algorithm`  | `ppo`; choose `ppo` or `dreamer`.                                               |
| `train`   | `--config`     | `configs/walker.toml`; input TOML path.                                         |
| `train`   | `--output`     | Required; new run directory.                                                    |
| `train`   | `--seed`       | Optional override of root `seed`.                                               |
| `rollout` | `run`          | Required positional run directory containing `run.json`.                        |
| `rollout` | `--checkpoint` | Optional specific snapshot; otherwise final run weights.                        |
| `rollout` | `--output`     | Required evaluation directory; each episode subdirectory must be new.           |
| `rollout` | `--seed`       | `1000`; course seed for the first episode, incremented for subsequent episodes. |
| `rollout` | `--episodes`   | `1`; positive number of episodes.                                               |
| `render`  | `episode`      | Required positional saved episode directory.                                    |
| `render`  | `--output`     | Defaults to `episode/episode.mp4`; destination video path.                      |
