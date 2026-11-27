# The justfile stores all the useful commands for the project
#
# Everything runs through uv, so recipes behave the same on macOS, Linux and Windows.
# Recipe lines stay valid in both sh and PowerShell; anything fancier lives in Python.
# Snakemake owns generated artifacts under data/; see docs/workflow.md.

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

# windows otherwise defaults python text i/o to cp1252
export PYTHONUTF8 := "1"

open := if os() == "macos" { "open" } else if os() == "windows" { "Invoke-Item" } else { "xdg-open" }

# default recipe
[private]
default:
    @just --list

# INFO: ---------------------------------
#         environment & quality
# ---------------------------------------

# install python deps and git lfs hooks (once per clone; needs git-lfs)
[group("core")]
install:
    @just bprint "📦 Installing dependencies..."
    @uv sync
    @git lfs install --local

# typecheck, then run the test suite
[group("core")]
test *args: typecheck
    @just bprint "🧪 Running tests..."
    @uv run pytest {{ args }}

# typecheck sources with basedpyright
[group("core")]
typecheck path="src":
    @just bprint "🔎 Typechecking {{ path }}..."
    @uv run basedpyright {{ path }}

# list jax devices; on a cuda machine this should show the gpu
[group("core")]
devices:
    @uv run python -c "import jax; print(jax.default_backend(), jax.devices())"

alias i := install
alias t := test

# INFO: ---------------------------------
#         training sessions
# ---------------------------------------
# progress lives in data/jobs/<job>/progress and resumes on any machine

# train a job for `hours` (blank: to completion); ctrl-c pauses, rerun to resume
[group("training")]
train name hours="":
    @just bprint "🏋️ Training {{ name }}{{ if hours == "" { "" } else { " for " + hours + " h" } }}..."
    @uv run python -m maton.workflow session --config configs/jobs/{{ name }}.toml {{ if hours == "" { "" } else { "--hours " + hours } }}

# training progress of every job: transitions, hours, sessions, machines
[group("training")]
status:
    @uv run python -m maton.workflow status

# commit one job's progress (weights via git lfs) so another machine can resume it
[group("training")]
share name:
    @just bprint "📤 Committing {{ name }} progress..."
    @uv run python -m maton.workflow share --job {{ name }}

alias s := status

# INFO: ---------------------------------
#         workflow artifacts
# ---------------------------------------
# snakemake targets: training resumes its progress, then publishes train/

# build a job's training, named evaluations, and selected videos
[group("workflow")]
job name="smoke-ppo" *args:
    @just bprint "🐍 Building {{ name }}..."
    @uv run snakemake --config job={{ name }} {{ args }}

# build any artifact by its canonical path; extra args include -n and --cores
[group("workflow")]
build target *args:
    @uv run snakemake {{ target }} {{ args }}

# smoke-test the default job end to end: training, evaluation, video
[group("workflow")]
smoke *args:
    @just bprint "💨 Running the smoke job..."
    @uv run snakemake {{ args }}

alias j := job

# INFO: --------------
#         tools
# --------------------

# preview a walker structure, then open its png
[group("tools")]
view-walker walker="biped" pose="zero":
    @uv run snakemake data/walkers/{{ walker }}/{{ pose }}.png
    @{{ open }} data/walkers/{{ walker }}/{{ pose }}.png

# build the talk deck (and any data/ media it references), then open it
[group("tools")]
talk *args:
    @just bprint "🎤 Building the talk deck..."
    @uv run snakemake data/talk {{ args }}
    @{{ open }} data/talk/index.html

# INFO: --------------
#         util
# --------------------

# print a message
[private]
print message:
    @echo "{{ message }}"

# print a message in bold
[private]
bprint message:
    @echo "{{ BOLD }}{{ message }}{{ NORMAL }}"
