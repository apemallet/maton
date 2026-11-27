# DreamerV3

Uses Danijar Hafner's upstream DreamerV3 model and Embodied replay, pinned to
[e01491f](https://github.com/danijar/dreamerv3/tree/e01491fad6434b2245a3b8ca201dd7faedcc458c).
The local harness collects vector observations, samples replay sequences, and
trains the upstream world model, imagination policy, and value function.

- Defaults: one environment, 256 transitions, 8-step sequences, 64 recurrent units.
- `[training]` controls transition budget, learning rate, and checkpoint interval.
- `[dreamer]` controls replay, network size, device, optimizer warmup, and action
  noise bounds. Initial upstream action standard deviation is approximately
  `minstd + 0.881 * (maxstd - minstd)`; it learns within those bounds.
- `metrics.jsonl`: update losses (first update, then every 64th) and every completed episode return. Upstream returns
  training metrics one update asynchronously behind the update counter.
- `dreamer.pkl`: final checkpoint; `dreamer-XXXXXXXX.pkl`: periodic snapshots.
- Upright course finishes advance the curriculum; falls may recover before timeout.
- Checkpoints include curriculum progress and default evaluation level.
- Checkpoints contain model, optimizer, RNG counters, and architecture settings.
- Sessions resume weights, optimizer, counters, and curriculum; replay restarts empty
  and the interrupted episode is dropped.
- Evaluation restores a recurrent policy and uses the upstream action distribution
  prediction (mean). Training keeps the upstream stochastic policy. Reset between
  episodes to reset recurrent state and posterior sampling RNG.

Upstream wheel packaging omits its `embodied.envs` namespace and default YAML.
The adapter supplies the unused namespace and includes upstream defaults with
its MIT license. A module-scoped proxy translates legacy positional `jax.jit`
options to modern keyword arguments. Project JAX and the model stay unchanged.
`uv` dependency metadata selects only the runtime needed by this vector task,
replacing upstream's obsolete CUDA/JAX pins and unrelated game dependencies.

Only load trusted local `.pkl` checkpoints.
