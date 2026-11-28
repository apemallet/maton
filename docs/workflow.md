# Jobs and artifacts

`configs/jobs/{job}.toml` owns `data/jobs/{job}/`. Job names use alphanumeric,
hyphen-separated segments; evaluations use lowercase kebab-case. Job names may include directory segments (for example `advanced/biped-tdmpc2-NML-s0`); artifact paths mirror those segments. Prefer `{walker}-{algorithm}-s{seed}` plus a descriptive variant.
Names identify experiments; settings are read from TOML, never inferred from names.

```text
job + walker + curriculum → train/
train/ + evaluation settings → eval/{evaluation}/
train/ metrics → plots/training.png   (return and curriculum level vs transitions)
saved episode → render/{evaluation}/episode-0000.mp4
walker → data/walkers/{walker}/{zero,neutral}.png
talk/slides.md + referenced data/ media → data/talk/index.html
```

Each job selects exactly one algorithm and contains only its trainer table.
Walker and curriculum paths are relative to the job file (`../walkers/biped.toml`).
Algorithm, training seed, and evaluation options belong in TOML; recipes supply no overrides.

```toml
algorithm = "ppo"
seed = 0
walker = "../walkers/biped.toml"
curriculum = "../curricula/walker.toml"
# [env], [training], [ppo] ...

[evaluations.level-1]
checkpoint = "final"  # or "step-00008192": actual saved transition count
level = 1            # optional; defaults to checkpoint's curriculum level
seed = 1000          # subsequent episodes increment this seed
episodes = 10
render = [0, 1, 2]   # zero-based indices; defaults to no videos
```

Evaluation defaults: final checkpoint, checkpoint level, seed 1000, one episode.
Named evaluations share training. Changing `render` only selects videos; it does
not invalidate trajectories. An explicitly requested video can use any valid
episode index, including those outside the default `render` selection.

# Inheritance and fine-tuning

A child job can inherit another job in `configs/jobs/` and train on a harder curriculum:

```toml
# configs/jobs/biped-ppo-harder-s0.toml
extends = "biped-ppo-v2-s0"
curriculum = "../curricula/harder.toml"  # create this curriculum first
continue_from = { job = "biped-ppo-v2-s0", checkpoint = "final" }

[training]
steps = 524_288          # additional transitions for this child
learning_rate = 0.00003
```

- `extends` names one job relative to `configs/jobs/`, without `.toml`. Tables merge recursively; values and arrays
  replace inherited values. Evaluations and `continue_from` inherit too. References
  stay relative to the TOML that declares them. Set `inherit_evaluations = false`
  to replace inherited protocols (useful for a shorter curriculum). Cycles are rejected.
- Inheritance alone reuses settings. `continue_from` independently selects source
  weights; omit `checkpoint` for `final`, or select `step-XXXXXXXX` at a saved count.
  `final` is the completed source job's latest checkpoint, published under `train/`.
- PPO, SAC, TD-MPC2, and Dreamer load model weights. Optimizers, replay, counters,
  environment state, and curriculum progress start fresh; the child starts at level 1.
  Use the same algorithm and compatible model/observation/action shapes.
- Snakemake tracks ancestor TOMLs and resolved settings, and builds source `train/`
  before child `train/`. Inheritance alone does not build the parent. Source identity
  and checkpoint selection enter the child's training fingerprint.
- `just job biped-ppo-harder-s0 -n` previews the chain. `just job` builds it.
  For `just train` sessions, build the source `train/` first. Paused children resume
  their own exact state; they do not reload source weights.

# Build

```sh
just smoke                       # default: smoke-ppo, including one video
just job smoke-dreamer
just job biped-ppo-v2-s0 -n          # review work without executing it
just build data/jobs/biped-ppo-v2-s0/train
just build data/jobs/biped-ppo-v2-s0/eval/level-1
just build data/jobs/biped-ppo-v2-s0/render/level-1/episode-0000.mp4
just view-walker biped neutral
```

Equivalent: `uv run snakemake --config job=biped-ppo-v2-s0` builds all configured job
outputs. A concrete path builds only that artifact and its dependencies.

# Sessions: pause, resume, share

Training happens in `data/jobs/{job}/progress/`, resumable on any machine.

```sh
just train biped-tdmpc2-v2-s0 2   # train for 2 hours (blank: to completion); Ctrl-C pauses
just status                    # transitions, %, hours, sessions, level, return, machines
just share biped-tdmpc2-v2-s0     # commit progress, plus built eval/, render/, plots/ (binaries via Git LFS)
just job biped-tdmpc2-v2-s0       # resume to the full budget, then publish train/, evaluate, render
```

- Trainers save exact state in `progress/resume/` at every checkpoint interval and on pause.
  Same-machine resumes match uninterrupted training; PPO, SAC, and TD-MPC2 are bit-identical.
- `progress.json` logs each session: steps, hours, machine, device, git revision.
- Shared: progress log, metrics, run settings, resume state, latest model, evaluations, videos, plots. Local only: replay
  buffers, numbered snapshots, published `train/`. Off-policy resumes elsewhere refill replay
  with the current policy for `seed_steps` before updating; Dreamer always restarts replay.
- One machine trains a job at a time: `just share` and push when you stop; `git pull` before resuming.
- Changed training settings refuse to resume; use a new job name.
- Run `just install` once per clone to enable Git LFS. LFS stores every shared version:
  per share, biped PPO ≈ 1 MB, SAC ≈ 5 MB, TD-MPC2 (5M) ≈ 100 MB. Check your LFS quota.

# Rebuild policy

- `train/` publishes completed progress; it is `protected(directory(...))`. Configuration fingerprints
  reject changed training settings under an existing name, even without local
  Snakemake metadata. Create a new job name for a changed experiment.
- Job/walker/curriculum inputs are `ancient()` **only for timestamp purposes**;
  a semantic training fingerprint in rule parameters tracks their contents.
  Formatting, evaluation edits, and video selection do not retrain models.
- Source files are not workflow inputs. After relevant implementation changes,
  use `-f` on the artifact to rebuild, e.g.
  `just build data/jobs/biped-tdmpc2-v2-s0/render/level-6/episode-0000.mp4 -f`.
  With current dependencies, this rerenders saved trajectories without evaluation.
  Missing or stale data dependencies still rebuild; inspect with `-n` first.
- Lockfile changes remain normal dependencies. Training reruns fail on protected
  outputs. Use a new job name, or deliberately remove the old protected training
  directory before rebuilding. Training resumes from `progress/`, never `train/`.
- After rule input definitions change, clear outdated dependency metadata with
  `uv run snakemake --cleanup-metadata PATH ...` for affected outputs. This keeps
  artifacts intact; training identity checks still apply.
- Evaluation and video outputs rebuild normally. Each directory belongs to one
  rule; evaluation and rendering live outside the protected training directory.
- Builders stage results beside their destination and publish after success.
  Failed training/evaluation leaves no partial final directory.
- Training retains resolved `run.json`, metrics, checkpoints, and `provenance.json`
  with training settings/fingerprint, Git revision/dirty status, source digest,
  and lockfile digest. Keep `.snakemake/` locally for dependency metadata.

Existing `data/` runs are historical artifacts and are not moved or adopted.

# Execution

`profiles/default/config.yaml` defaults to one local core and one training slot.
Rules declare threads, memory (MB), runtime (minutes), and training slots. Override
resources with `--set-resources train:mem_mb=16384 train:runtime=1440` and threads
with `--set-threads train=4`; `--cores` controls total local concurrency.

For remote execution, install the appropriate Snakemake executor plugin and pass
an executor profile with `--profile PATH`. Put scheduler/GPU allocation there;
set `dreamer.platform` in the job for the model's compute platform. No remote
provider is assumed or configured by this repository.

# Advanced training

The long TD-MPC2 child lives in `configs/jobs/advanced/`; its [course and efficiency gate](curriculum.md#no-mans-land)
are separate from the baseline curriculum. The profile supplies 16 GB memory and
72 h runtime allowances; use a GPU host or add an executor profile for your scheduler.

```sh
just job advanced/biped-tdmpc2-NML-s0 --profile profiles/advanced -n
just job advanced/biped-tdmpc2-NML-s0 --profile profiles/advanced
# After the parent train/ exists, bounded sessions also work:
just train advanced/biped-tdmpc2-NML-s0 2
```

The DAG includes `biped-tdmpc2-v2-s0/train` → child `train` → 100 held-out evaluations on level 2,
three videos, and training plots. Status discovers nested jobs; share uses their full names.
