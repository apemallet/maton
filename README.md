# Automata

Learn local cellular-automaton rules that grow a target image from a simple seed.
A global SAC controller edits a gated convolutional rule; batched CA evolution
and a strong edit penalty encourage a stationary rule to emerge organically.
The rule has a nonlinear convolutional trunk and gated residual head. Initial rule
parameters are learned through the critic, without CA gradients or a target-loss
initialization objective. Actor and
critics encode the current grid and its constant-rule look-ahead as
`(E(x), E(x_future) − E(x))` with small convolutional networks. Training starts near
the target, adding more noise and cell dropout as performance improves. A central
latent marker preserves the target location during corruption.

```sh
uv sync
just test
just train --output data/disk-01
just plot data/disk-01
```

Small CPU smoke run with actual learning:

```sh
just train --output data/smoke --steps 64 --warmup 8 --batch-size 8 \
  --episode-length 16 --size 6 --channels 3 --width 32 --ca-steps 4
```

Defaults: `12×12` periodic grid, one visible plus three hidden channels, a central
goal seed, a disk target, and 16 CA steps per controller decision. Training uses
perturbed target states, gradually mixing in the goal seed as difficulty rises. Use
`uv run automata --help` for settings. Choose a fresh output directory per run.

Outputs: `config.json`, per-decision `metrics.csv`, `summary.json`, inference
snapshot `model.eqx` (actor, twin critics, learned reset rule), and `rollout.npz`
(final episode states at block boundaries, target, final rule, learned reset rule,
and reset rules actually used). `just plot` adds `overview.png`. Inference snapshots are also refreshed every 1024 decisions. Snapshots omit
optimizer/replay state. `--initialize-from RUN` warm-starts the learned networks
and reset rule with fresh optimizer/replay state. No forced stationary
evaluation is performed; inspect target error and edit size together.

See the concise [design overview](docs/overview.md).
Experiment findings are recorded in [experiments](docs/experiments.md).

Separate direct-rule control: `just bptt data/bptt-01`, then
`just bptt-evaluate data/bptt-01`. The working local-recovery control is
`just bptt-stable data/bptt-stable`. It uses 16-step BPTT without a controller;
SAC initialization remains strictly critic-based. See the [baseline protocol](docs/bptt.md).

To discover a recovery signal before pricing edits, use
`--edit-penalty 0 --stop-on-signal`. This saves at the first sustained improvement
window; `signal.json` reports the comparison against starting error. Then
warm-start with a strong penalty. SAC experiments have **not established
convergence**; low edit magnitude alone does not demonstrate target recovery.
