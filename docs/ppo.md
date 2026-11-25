# PPO baseline

- Equinox actor and critic: two 64-unit tanh hidden layers; four tanh-normal joint actions.
- Clipped PPO, GAE, Adam, gradient clipping; normalized batch advantages.
- True terminations stop value bootstrap. Time limits bootstrap from the final observation; both stop the GAE trace.
- Entropy regularization uses the underlying Gaussian entropy; `initial_std` sets initial exploration around the actor mean.
- `configs/jobs/smoke-ppo.toml` checks integration; `biped-ppo-s0.toml` uses a larger walking budget.

`[training]` sets the transition budget, learning rate, and checkpoint interval.
PPO saves at completed rollout updates; numbered filenames use transition counts.

Falls and pit entries remain live until recovery, finish, or timeout.
Curriculum promotion counts upright course finishes across environments. In-flight
episodes finish on their assigned level; old-level completions cannot promote the next.
Metrics and checkpoint metadata include curriculum progress.

Training writes `metrics.csv`, `summary.json`, `policy.eqx`, and numbered inference
snapshots in `checkpoints/`. Every snapshot has JSON model metadata.
`load_policy(path)` accepts a run directory or a snapshot; `action(policy, obs)`
returns a deterministic action for replay. Snapshots contain model weights, not
optimizer state; training starts a fresh run.
