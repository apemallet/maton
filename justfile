open := if os() == "macos" { "open" } else { "xdg-open" }

_default:
	@just --list

alias i := install
# Install dependencies and Git LFS hooks (once per clone; needs git-lfs).
install:
	uv sync
	git lfs install --local

alias t := test
test *args:
	just typecheck
	uv run pytest {{args}}

typecheck path='src':
	uv run basedpyright {{path}}

# Train a job for HOURS (default: to completion). Ctrl-C pauses; rerun to resume.
train name hours='':
	uv run python -m maton.workflow session --config configs/jobs/{{name}}.toml {{ if hours == "" { "" } else { "--hours " + hours } }}

# Training progress of every job: transitions, hours, sessions, machines.
status:
	uv run python -m maton.workflow status

# Commit one job's progress (weights via Git LFS) so another machine can resume it.
share name:
	git add data/jobs/{{name}}/progress
	git commit -m "$(uv run python -m maton.workflow status --job {{name}} --oneline)" -- data/jobs/{{name}}/progress

# Build the job's training (resuming its progress), named evaluations, and selected videos.
job name='smoke-ppo' *args:
	uv run snakemake --config job={{name}} {{args}}

# Build any artifact by its canonical path. Extra args include -n and --cores.
build target *args:
	uv run snakemake {{target}} {{args}}

smoke *args:
	uv run snakemake {{args}}

# Preview a walker through the workflow, then open its PNG.
view-walker walker='biped' pose='zero':
	uv run snakemake data/walkers/{{walker}}/{{pose}}.png
	{{open}} data/walkers/{{walker}}/{{pose}}.png

# Build the talk deck (and any data/ media it references), then open it.
talk *args:
	uv run snakemake data/talk {{args}}
	{{open}} data/talk/index.html

# List JAX devices; on a CUDA machine this should show the GPU.
devices:
	uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
