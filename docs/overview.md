# Walker and course

- Planar root translation/pitch with a TOML-defined body tree, hinges, and capped motors.
  Jobs select a [walker structure](walker.md); examples include bipeds and a quadruped.
- Actions command torque directly or joint targets with per-physics-step PD control.
  The walking job uses position control; see [motor settings](config.md).
- Geometry supports spheres, capsules, and boxes; contact sensing is optional per geom.
- Default 100 Hz physics, two steps per action. Crossing the finish line (success only if
  upright), staying fallen 1 s, sinking 0.5 m below the floor, and leaving the backward
  course boundary terminate; episode time limits truncate. Brief falls may recover.
- Observation (`4 * (1 + hinges) * K + 2 + contacts + lidar` floats; 53 for the default biped): sin/cos at harmonics
  1..K of absolute torso pitch and all parent-relative hinge angles; the same Fourier
  features of each angular rate times the action period (angle change per action); world-frame
  root x/z velocity; configured contacts; nine normalized lidar rays. No height, course
  position, or previous action. Fourier angles are wrap-free.
- Lidar intersects the same terrain boxes used by MJX. Rays rotate with the torso,
  spanning straight down to 15° above forward.

Jobs reference the six-level [curriculum](curriculum.md), starting with flat walking.
Each terrain slot independently selects a block, gap, or flat floor. Frequencies
are probabilities per slot and must sum to at most one. Heights and widths are
sampled uniformly from 50–100% of their configured maxima; `gap_height` is depth
below the floor. Gaps have physical bottoms and vertical sides; later levels
use 5–20 m depths; sinking into one ends the episode. A clear starting platform
precedes the slots and a clear run-out follows them. Resetting randomizes geometry without changing JAX shapes.

Progress potential is `Phi(x) = progress_scale * min(x, course_length)`,
saturating at the finish so running off the course earns no further progress.
Optional `progress_mode="upright"` multiplies this potential by a height and
uprightness score: falling loses progress potential, and recovery restores it. Each action earns
`Phi(next) - Phi(previous) - control_cost * mean(normalized_torque²)`.
An optional `upright_reward` adds a posture score per simulated second, weighted
by torso height and uprightness. Moving backward loses progress reward;
standing still adds no progress. An optional finish bonus rewards upright completion.
Brief falls can recover; a fall held for `fall_timeout` ends the episode.

# Checkpoints and inspection

`maton train --config configs/jobs/biped-ppo-v2-s0.toml --output data/RUN`
saves resolved settings in `run.json`. Outputs must be new directories.
PPO checkpoints hold inference weights; Dreamer holds agent/optimizer state.
Training resumes from saved progress; see [sessions](workflow.md#sessions-pause-resume-share).

`maton rollout RUN --checkpoint FILE --output EVAL --episodes 3 --seed 1000`
evaluates fresh seeded courses with recurrent state reset between episodes.
Omit `--checkpoint` for final weights. Evaluation defaults to the checkpoint
curriculum level; `--level 6` selects the last level. Each episode contains:

- `trajectory.npz`: initial and subsequent qpos/qvel, actions, observations,
  rewards, and terrain.
- `model.xml`: the exact sampled course and robot.
- `episode.json`: seed, return, distance, duration, level, and success.

`maton render EVAL/episode-0000` produces `episode.mp4` from a side-on projection
of the saved MuJoCo geometry. It needs no display, GPU, or policy checkpoint.

Smoke tests validate finite dynamics, optimizer updates, checkpoint reload, and
video generation. These budgets establish integration, not locomotion performance.
