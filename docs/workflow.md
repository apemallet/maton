# Jobs and artifacts

`configs/jobs/{job}.toml` owns `data/jobs/{job}/`. Job/evaluation names use lowercase
kebab-case. Prefer `{walker}-{algorithm}-s{seed}` plus a descriptive variant.
Names identify experiments; settings are read from TOML, never inferred from names.

```text
job + walker + curriculum → train/
train/ + evaluation settings → eval/{evaluation}/
saved episode → render/{evaluation}/episode-0000.mp4
walker → data/walkers/{walker}/{zero,neutral}.png
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

# Build

```sh
just smoke                       # default: smoke-ppo, including one video
just job smoke-dreamer
just job biped-ppo-s0 -n          # review work without executing it
just build data/jobs/biped-ppo-s0/train
just build data/jobs/biped-ppo-s0/eval/level-1
just build data/jobs/biped-ppo-s0/render/level-1/episode-0000.mp4
just view-walker biped neutral
```

Equivalent: `uv run snakemake --config job=biped-ppo-s0` builds all configured job
outputs. A concrete path builds only that artifact and its dependencies.

# Rebuild policy

- Completed training is `protected(directory(...))`. Configuration fingerprints
  reject changed training settings under an existing name, even without local
  Snakemake metadata. Create a new job name for a changed experiment.
- Job/walker/curriculum inputs are `ancient()` **only for timestamp purposes**;
  a semantic training fingerprint in rule parameters tracks their contents.
  Formatting, evaluation edits, and video selection do not retrain models.
- Source and lockfile changes remain normal dependencies. A scheduled training
  rerun fails on protected outputs rather than overwriting the model. Use a new
  job name, or deliberately remove the old protected training directory before
  requesting a rebuild; there is no automatic training resume.
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
