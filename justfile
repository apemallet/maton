_default:
	@just --list

alias i := install
install:
	uv sync

alias t := test
test *args:
	just typecheck && echo ''
	uv run pytest {{args}}

typecheck path='src':
	uv run basedpyright {{path}}

train algorithm='ppo' output='data/ppo-dev' config='configs/walker.toml':
	uv run maton train --algorithm {{algorithm}} --config {{config}} --output {{output}}

smoke algorithm='ppo' output='data/ppo-smoke':
	uv run maton train --algorithm {{algorithm}} --config configs/smoke.toml --output {{output}}

rollout run output seed='1000':
	uv run maton rollout {{run}} --output {{output}} --seed {{seed}}

render episode:
	uv run maton render {{episode}}
