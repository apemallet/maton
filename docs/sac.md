# SAC

Soft Actor-Critic on the shared [off-policy driver](offpolicy.md): the proposal's model-free
counterpart to [TD-MPC2](tdmpc2.md), with identical replay, seeding, and update ratio.

- Tanh-Gaussian actor (ReLU MLP); `num_q` critics as one stacked ensemble, clipped double-Q targets.
- Critic LayerNorm (`critic_norm`) by default: stabilizes high update ratios.
- Temperature `alpha` tunes toward `target_entropy` (default `-action_size`).
- Discount: same episode-length heuristic as TD-MPC2 unless `discount` is set.
- Evaluation uses the tanh of the actor mean.
- Checkpoints: `sac.eqx`, `checkpoints/step-XXXXXXXX.eqx`, with JSON metadata.
