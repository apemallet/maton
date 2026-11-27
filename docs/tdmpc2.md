# TD-MPC2

Equinox port of [TD-MPC2](https://github.com/nicklashansen/tdmpc2) (Hansen et al., 2024) on the
shared [off-policy driver](offpolicy.md). Defaults are the official single-task 5M-parameter model.

- **World model:** encoder → SimNorm latent; latent dynamics; two-hot reward head;
  termination head (`episodic`); ensemble of `num_q` two-hot critics with slow targets.
- **Acting:** MPPI over `horizon` steps: `num_pi_trajs` policy-prior rollouts plus Gaussian
  samples, `iterations` refits on the `num_elites` best returns, then a score-weighted elite.
  Previous means warm-start the next step. `mpc = false` acts with the policy prior only.
- **Learning:** latent consistency + reward + value + termination losses, weighted by `rho^t`;
  the policy maximizes running-scale-normalized Q plus scaled entropy.
- **Discount:** `clip((L/5 - 1) / (L/5), 0.95, 0.995)` for episode length `L`, unless set.
- **Evaluation** plans without exploration noise, seeded per episode.

## Departures from the official code

| Change | Why |
| --- | --- |
| `num_envs` MJX environments; planning is batched across them | Throughput; the official agent steps one environment |
| `update_ratio` updates per transition (official: 1) | Trade sample efficiency for wall-clock |
| Replay windows may span episode ends; later steps are masked | Fixed-shape device buffer; same losses as sampling within episodes |
| Stored `next_observation` for every transition | Time limits bootstrap from the true final state |
| Min/avg Q evaluates only the two sampled heads | Same estimator, fewer FLOPs in planning and targets |
| Scaled entropy as `-action_size · log N(ε)` | Equals the official ratio form without its division |

## Cost (M1 Pro CPU, biped observation)

| Model | Update | Planning per env step (N = 1 / 8) |
| --- | --- | --- |
| 5M (default) | 130 ms | 197 / 141 ms |
| 1M (`latent_dim=128, mlp_dim=384, num_q=2`) | 53 ms | 99 / 76 ms |
| 1M, `num_samples=256, iterations=4` | 60 ms | 41 / 29 ms |

Compute-bound: use a GPU for walking budgets. Checkpoints: `tdmpc2.eqx` and
`checkpoints/step-XXXXXXXX.eqx` (inference model only) with JSON metadata.
