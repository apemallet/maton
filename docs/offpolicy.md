# Off-policy driver

[`offpolicy.py`](../src/maton/offpolicy.py) trains any learner exposing `init_model`, `init`,
`init_carry`, `act`, and `update` (see `Learner`). Used by [TD-MPC2](tdmpc2.md) and [SAC](sac.md).

- `num_envs` batched MJX environments; finished ones reset in place.
- The first `seed_steps` transitions use uniform random actions. One burst of
  `round(seed_steps × update_ratio)` updates follows; afterwards `update_ratio` updates per transition.
- Replay lives on device: `[rows, envs]` ring buffers of observation, action, reward,
  next observation, terminated, and done. Windows of `horizon` consecutive transitions
  never cross the write pointer; `mask` zeroes steps after an episode ends.
- Collection (acting, stepping, storing, resetting) and each update burst are single jitted calls.
- Curriculum promotions apply at the next reset; the level is a traced value, so no recompilation.
- Outputs: `metrics.jsonl` every `log_every` transitions (returns, successes, losses,
  throughput, curriculum), `summary.json`, final and periodic model checkpoints.
