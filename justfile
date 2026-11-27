open := if os() == "macos" { "open" } else { "xdg-open" }

_default:
	@just --list

alias i := install
install:
	uv sync

alias t := test
test *args:
	just typecheck
	uv run pytest {{args}}

typecheck path='src':
	uv run basedpyright {{path}}

# Build the job's training, named evaluations, and selected videos.
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
