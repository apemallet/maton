# PPO baseline

- Equinox actor and critic: two 64-unit tanh hidden layers; four tanh-normal joint actions.
- Clipped PPO, GAE, Adam, gradient clipping; normalized batch advantages.
- True terminations stop value bootstrap. Time limits bootstrap from the final observation; both stop the GAE trace.
- Entropy regularization uses the underlying Gaussian entropy.
- Defaults are a stability smoke run (256 transitions), not a locomotion training budget.

`[training]` sets the transition budget, learning rate, and checkpoint interval.
PPO saves at completed rollout updates; numbered filenames use transition counts.

Training writes `metrics.csv`, `summary.json`, `policy.eqx`, and numbered inference
snapshots in `checkpoints/`. Every snapshot has JSON model metadata.
`load_policy(path)` accepts a run directory or a snapshot; `action(policy, obs)`
returns a deterministic action for replay. Snapshots contain model weights, not
optimizer state; training starts a fresh run.
