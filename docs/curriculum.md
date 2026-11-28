# Curriculum

Jobs set `curriculum = "curricula/walker.toml"`, relative to the job file.
The separate [curriculum TOML](../configs/curricula/walker.toml) defines:

| Level | Course |
| --- | --- |
| 1 | Flat floor: learn to walk to the end |
| 2 | Sparse, small blocks |
| 3 | More frequent, larger blocks |
| 4 | Blocks and shallow, recoverable recesses |
| 5 | Deep (5–10 m), wider gaps |
| 6 | Dense mixed terrain and treacherous 10–20 m gaps |

Courses are 8 m: 1 m clear start, six 1 m obstacle slots, then 1 m clear
**run-out** before the finish line, so a lunge over the last obstacle cannot win.
Floor continues 2 m past the finish. Promotion needs **4 successes among the last
10 episodes** at a level (`successes_required`, `window`; global or per level), so
a few lucky finishes no longer promote. Success means crossing the finish line
upright. Finishing the last level marks the curriculum complete; training
continues there until its transition budget is exhausted.

PPO shares one progression counter across environments. Promotion affects new
episodes only. Already-running episodes retain their level, and their later
finishes cannot count toward a different level. Dreamer changes level on reset.
The policy, optimizer, and Dreamer replay are retained across promotions. TD-MPC2/SAC
restart all environments and clear replay when promotion changes the efficiency penalty.

Episodes end (terminate) when the walker:
- crosses the finish line: a success only if upright;
- stays fallen for `env.fall_timeout` (1 s), or sinks `env.sink_depth` (0.5 m) below
  the floor, e.g. into a deep gap: failures. Brief stumbles and shallow recesses recover;
- leaves the course backwards, or the simulation becomes invalid.

Otherwise the time limit truncates. Hopeless states no longer idle until timeout.

Shared `[terrain]` defaults are overlaid by `[levels.terrain]`. Keep `slots`,
`spacing`, `start_length`, and `run_out` identical across levels (fixed MJX shapes).
Obstacle height/depth and width still vary from 50–100% of configured maxima;
gap/block frequencies are probabilities, so not every sampled course has a gap.

Run snapshots embed the entire resolved curriculum. Checkpoints and summaries
include the current level and success count; metrics record progression.
Saved level indices are zero-based; the CLI uses level numbers starting at one.
`maton rollout RUN --output EVAL --level 6` explicitly tests the final level.
Omit `--level` to evaluate the checkpoint's current level. Episode metadata
records the level and successful completion.

# No Man’s Land

[`no-mans-land.toml`](../configs/curricula/no-mans-land.toml) has **two levels sharing identical terrain**:
8.25 m with hard-level random blocks and exactly one central pit, **0.85 m wide ×
0.8 m deep**. Other slots never contain gaps. Central pit dimensions stay fixed;
blocks retain the usual size randomization. The pit is wider than the previous
maximum gap (0.65 m), but shallower than the biped (~1.16 m tall). Jumping across
or climbing out are intended strategies; feasibility still needs training evidence.

The [advanced TD-MPC2 job](../configs/jobs/advanced/biped-tdmpc2-NML-s0.toml)
loads the v2 policy, then trains for **16,777,216 additional transitions (32×)**.
Episodes allow 40.96 s; falls allow 3 s to recover. Height checks use local terrain
(`env.terrain_relative_height`) so standing on the pit floor does not count as fallen.
Observations and policy shapes stay compatible with the parent checkpoint.

# Efficiency as a second level

```toml
[[levels]]
name = "no-mans-land"
successes_required = 4
window = 10
efficiency_penalty = 0.0

[[levels]]
name = "no-mans-land-efficient"
efficiency_penalty = 2.0
```

- **Level 1:** learn to cross, with no efficiency penalty. Four successes among the
  last ten episodes promote to level 2.
- **Level 2:** identical course, with the penalty active immediately. Passing this
  level marks completion; training continues with the penalty to the full budget.
- `efficiency_penalty` belongs to each level and defaults to zero. Every episode uses
  its own level's penalty; pause/resume preserves the current level. Curricula may
  start directly on hard terrain for fine-tuning.
- Work is `Σ |joint torque × joint velocity| × physics dt`, integrated at every
  physics substep, including braking. Reward subtracts `efficiency_penalty × work / (mass × gravity)`.
  No Man’s Land uses weight **2**; its earlier action-cost and survival bonuses are zero.
- TD-MPC2/SAC clear and refill replay with the learned policy when the penalty starts,
  restarting all environments on the new level, so old unpenalized episodes cannot
  refill the new buffer.
- Evaluations report success, work (J), and **cost of transport**: total work divided
  by `mass × gravity × distance`. Successful crossings alone enter mean transport cost
  and crossing time; failed episodes never appear as efficient successes.
  Evaluation returns use the selected level’s reward; the advanced job evaluates level 2.

Mechanical work measures motion efficiency; it is not an electrical battery model.
