# Algorithms

[`algorithms.py`](../src/maton/algorithms.py) registers each trainer without importing JAX.
Jobs, the CLI, run snapshots, and Snakemake read names, checkpoint paths, and source
dependencies from it.

| Name | Module | Final checkpoint | Snapshots |
| --- | --- | --- | --- |
| `ppo` | [ppo](ppo.md) | `policy.eqx` | `checkpoints/step-XXXXXXXX.eqx` |
| `dreamer` | [dreamer](dreamer.md) | `dreamer.pkl` | `dreamer-XXXXXXXX.pkl` |

## Adding a trainer

1. Create `src/maton/<name>.py` exposing:
   - `Config`: frozen dataclass for the job's `[<name>]` table; validate in `__post_init__`.
   - `train(env, config, training, output, seed) -> dict`: honour `training.steps` and
     `training.checkpoint_due`; record curriculum progress; write a JSON-serialisable summary.
   - `load_agent(checkpoint) -> Agent`: `reset()`, `__call__(obs) -> action`, `curriculum_level`.
2. Add one `Algorithm(...)` entry: checkpoint layout, shared modules, and non-Python assets.
   These become Snakemake training inputs.
3. Add `configs/jobs/smoke-<name>.toml`, tests, and a docs page.

`run.json` stores only the selected trainer's resolved table.
