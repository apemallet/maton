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
The policy, optimizer, and Dreamer replay are retained across promotions.

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
