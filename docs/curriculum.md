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

All courses are 4 m long. Five cumulative successful finishes are required **at
each level**, configurable globally or per level using `successes_required`.
Success means reaching the actual course end while upright. Timeouts and
simulation failures never count. Failures do not erase earlier successes. Finishing the last level marks the curriculum complete;
training continues there until its transition budget is exhausted.

PPO shares one progression counter across environments. Promotion affects new
episodes only. Already-running episodes retain their level, and their later
finishes cannot count toward a different level. Dreamer changes level on reset.
The policy, optimizer, and Dreamer replay are retained across promotions.

Gaps have physical sides and bottoms, with no gap-entry kill rule. Falling or
being below the original floor does not end an episode, so recovery remains
possible. Late-level depths are intended to trap the walker until its ordinary
episode timeout. Only upright course completion, leaving the backward course
boundary, or invalid dynamics terminate early. A fallen walker at the finish
must recover before the finish counts.

Shared `[terrain]` defaults are overlaid by `[levels.terrain]`. Keep `slots`,
`spacing`, and `start_length` identical across levels to avoid MJX recompilation.
Obstacle height/depth and width still vary from 50–100% of configured maxima;
gap/block frequencies are probabilities, so not every sampled course has a gap.

Run snapshots embed the entire resolved curriculum. Checkpoints and summaries
include the current level and success count; metrics record progression.
Saved level indices are zero-based; the CLI uses level numbers starting at one.
`maton rollout RUN --output EVAL --level 6` explicitly tests the final level.
Omit `--level` to evaluate the checkpoint's current level. Episode metadata
records the level and successful completion.
