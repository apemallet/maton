"""Atomic artifact builders called by Snakemake."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from collections.abc import Iterator

from maton.jobs import Job, fingerprint


@contextmanager
def staged_directory(output: Path) -> Iterator[Path]:
    """Publish only successful results; retries never see partial run directories."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'artifact'
        yield stage
        if output.exists():
            raise FileExistsError(output)
        # Training summaries contain their checkpoint's path.
        summary = stage / 'summary.json'
        if summary.exists():
            summary.write_text(summary.read_text().replace(str(stage), str(output)))
        stage.rename(output)


def provenance(job: Job) -> dict:
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=normal'], capture_output=True, text=True)
    return dict(job=job.path.stem, training_fingerprint=fingerprint(job.training_spec()),
                training=job.training_spec(), source_revision=revision.stdout.strip() or None,
                source_dirty=bool(dirty.stdout.strip()),
                source_fingerprint=fingerprint({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                                for p in sorted(Path('src/maton').rglob('*.py'))}),
                lock_fingerprint=hashlib.sha256(Path('uv.lock').read_bytes()).hexdigest())


def build_train(job: Job, output: Path) -> None:
    from maton.cli import main
    with staged_directory(output) as stage:
        main(['train', '--config', str(job.path), '--output', str(stage)])
        (stage / 'provenance.json').write_text(json.dumps(provenance(job), indent=2) + '\n')


def build_evaluation(job: Job, name: str, train: Path, output: Path) -> None:
    from maton.cli import main
    evaluation = job.evaluations[name]
    checkpoint = job.checkpoint(train, evaluation)
    if not checkpoint.is_file():
        raise ValueError(f'Checkpoint does not exist: {checkpoint}; use an actual saved transition count')
    with staged_directory(output) as stage:
        args = ['rollout', str(train), '--checkpoint', str(checkpoint), '--output', str(stage),
                '--seed', str(evaluation.seed), '--episodes', str(evaluation.episodes)]
        if evaluation.level is not None:
            args += ['--level', str(evaluation.level)]
        main(args)
        (stage / 'evaluation.json').write_text(json.dumps(evaluation.spec(), indent=2) + '\n')


def build_render(episode: Path, output: Path) -> None:
    from maton.replay import render_episode
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.render-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'episode.mp4'
        render_episode(episode, stage)
        stage.replace(output)


def build_preview(walker: Path, pose: str, output: Path) -> None:
    from maton.preview import preview_walker
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.preview-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'preview.png'
        preview_walker(walker, stage, neutral=pose == 'neutral')
        stage.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage', required=True)
    for name in ('train', 'evaluate'):
        command = commands.add_parser(name)
        command.add_argument('--config', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        if name == 'evaluate':
            command.add_argument('--evaluation', required=True)
            command.add_argument('--train', type=Path, required=True)
    render = commands.add_parser('render')
    render.add_argument('--episode', type=Path, required=True)
    render.add_argument('--output', type=Path, required=True)
    preview = commands.add_parser('preview')
    preview.add_argument('--walker', type=Path, required=True)
    preview.add_argument('--pose', choices=('zero', 'neutral'), required=True)
    preview.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'train':
        build_train(Job.load(args.config), args.output)
    elif args.stage == 'evaluate':
        build_evaluation(Job.load(args.config), args.evaluation, args.train, args.output)
    elif args.stage == 'render':
        build_render(args.episode, args.output)
    else:
        build_preview(args.walker, args.pose, args.output)


if __name__ == '__main__':
    main()
