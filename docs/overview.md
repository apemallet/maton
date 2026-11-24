# Walker and course

- Planar root translation/pitch; two hips and two knees, four ±20 Nm motors.
- Default 100 Hz physics, two steps per action. Falling, leaving the course, or
  reaching the finish terminates; episode time limits truncate.
- Nine normalized lidar rays, torso height/orientation, four joint angles,
  seven velocities, two foot contacts, and previous actions: 29 floats.
- Lidar intersects the same terrain boxes used by MJX. Rays rotate with the torso,
  spanning straight down to 15° above forward.

Each terrain slot independently selects a block, gap, or flat floor. Frequencies
are probabilities per slot and must sum to at most one. Heights and widths are
sampled uniformly from 50–100% of their configured maxima; `gap_height` is depth
below the floor. Gaps have bottoms and vertical sides. A clear starting platform
precedes the slots. Resetting randomizes geometry without changing JAX shapes.

Progress potential is `Phi(x) = progress_scale * x`. Each action earns
`Phi(next) - Phi(previous) - control_cost * mean(action²)`. Moving backward loses
reward; standing still adds no progress. There is no survival bonus.

# Checkpoints and inspection

`maton train --algorithm {ppo,dreamer} --config configs/walker.toml --output data/RUN`
saves resolved settings in `run.json`. Outputs must be new directories.
PPO checkpoints hold inference weights; Dreamer holds agent/optimizer state.
Neither harness currently resumes training/replay from an existing run.

`maton rollout RUN --checkpoint FILE --output EVAL --episodes 3 --seed 1000`
evaluates fresh seeded courses with recurrent state reset between episodes.
Omit `--checkpoint` for final weights. Each episode contains:

- `trajectory.npz`: initial and subsequent qpos/qvel, actions, observations,
  rewards, and terrain.
- `model.xml`: the exact sampled course and robot.
- `episode.json`: seed, return, distance, duration, and ending reason.

`maton render EVAL/episode-0000` produces `episode.mp4` from a side-on projection
of the saved MuJoCo geometry. It needs no display, GPU, or policy checkpoint.

Smoke tests validate finite dynamics, optimizer updates, checkpoint reload, and
video generation. These budgets establish integration, not locomotion performance.
