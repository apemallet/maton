# Walker and course

- Planar root translation/pitch; two hips and two knees, four capped torque motors.
- Actions command torque directly or joint targets with per-physics-step PD control.
  The walking job uses position control; see [motor settings](config.md).
- Optional short capsule feet widen the support footprint without extra joints;
  `foot_length=0` preserves the original spherical feet.
- Default 100 Hz physics, two steps per action. Upright course completion and leaving the backward
  course boundary terminate; episode time limits truncate. Falls may recover.
- Observation (`20K + 4 + lidar` floats, `K = angle_harmonics`; 53 by default): sin/cos at harmonics
  1..K of absolute torso pitch and the four parent-relative joint angles; the same Fourier
  features of each angular rate times the action period (angle change per action); world-frame
  root x/z velocity; two foot contacts; nine normalized lidar rays. No height, course
  position, or previous action. Fourier angles are wrap-free.
- Lidar intersects the same terrain boxes used by MJX. Rays rotate with the torso,
  spanning straight down to 15° above forward.

Jobs reference the six-level [curriculum](curriculum.md), starting with flat walking.
Each terrain slot independently selects a block, gap, or flat floor. Frequencies
are probabilities per slot and must sum to at most one. Heights and widths are
sampled uniformly from 50–100% of their configured maxima; `gap_height` is depth
below the floor. Gaps have physical bottoms and vertical sides; later levels
use 5–20 m depths without any gap-entry or fall kill rule. A clear starting platform
precedes the slots. Resetting randomizes geometry without changing JAX shapes.

Progress potential is `Phi(x) = progress_scale * min(x, course_length)`,
saturating at the finish so running off the course earns no further progress.
Optional `progress_mode="upright"` multiplies this potential by a height and
uprightness score: falling loses progress potential, and recovery restores it. Each action earns
`Phi(next) - Phi(previous) - control_cost * mean(normalized_torque²)`.
An optional `upright_reward` adds a posture score per simulated second, weighted
by torso height and uprightness. Moving backward loses progress reward;
standing still adds no progress. An optional finish bonus rewards upright completion.
Falls never terminate an episode.

# Checkpoints and inspection

`maton train --algorithm {ppo,dreamer} --config configs/walker.toml --output data/RUN`
saves resolved settings in `run.json`. Outputs must be new directories.
PPO checkpoints hold inference weights; Dreamer holds agent/optimizer state.
Neither harness currently resumes training/replay from an existing run.

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
