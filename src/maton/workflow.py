"""Atomic artifact builders called by Snakemake."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from typing import Any

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
            # Match the JSON-escaped form: Windows separators are written as `\\`.
            escape = lambda path: json.dumps(str(path))[1:-1]
            summary.write_text(summary.read_text().replace(escape(stage), escape(output)))
        stage.rename(output)


def provenance(job: Job) -> dict:
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=normal'], capture_output=True, text=True)
    return dict(job=job.name, training_fingerprint=fingerprint(job.training_spec()),
                training=job.training_spec(), source_revision=revision.stdout.strip() or None,
                source_dirty=bool(dirty.stdout.strip()),
                source_fingerprint=fingerprint({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                                for p in sorted(Path('src/maton').rglob('*.py'))}),
                lock_fingerprint=hashlib.sha256(Path('uv.lock').read_bytes()).hexdigest())


def progress_directory(job: Job) -> Path:
    """Resumable training for a job; shared between machines through Git LFS."""
    return Path('data/jobs') / job.name / 'progress'


def read_progress(job: Job) -> dict[str, Any] | None:
    path = progress_directory(job) / 'progress.json'
    return json.loads(path.read_text()) if path.exists() else None


def run_session(job: Job, hours: float | None = None, steps: int | None = None) -> dict:
    """Train `job` from its saved progress until done, `hours` elapse, `steps` is reached, or Ctrl-C."""
    import jax
    from maton.cli import main
    progress = progress_directory(job)
    signature = fingerprint(job.training_spec())
    log: dict[str, Any] = read_progress(job) or dict(job=job.name, algorithm=job.algorithm, training_fingerprint=signature,
                                     total=job.training_spec()['training'].get('steps', 256), steps=0,
                                     complete=False, hours=0.0, sessions=[])
    if log['training_fingerprint'] != signature:
        raise ValueError(f'{progress}: training settings changed since this progress began; '
                         'use a new job name or delete the progress directory')
    started, clock, before = datetime.now(timezone.utc), time.perf_counter(), log['steps']
    args = ['train', '--config', str(job.path), '--output', str(progress)]
    args += ['--hours', str(hours)] if hours is not None else []
    args += ['--steps', str(steps)] if steps is not None else []
    try:
        main(args)
    finally:
        summary_path = progress / 'summary.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary['steps'] != before:
                revision = provenance(job)
                log['sessions'].append(dict(started=started.isoformat(timespec='seconds'),
                                            hours=round((time.perf_counter() - clock) / 3600, 4),
                                            steps=[before, summary['steps']], machine=platform.node(),
                                            device=jax.default_backend(), revision=revision['source_revision'],
                                            dirty=revision['source_dirty']))
            log.update(steps=summary['steps'], complete=summary['complete'],
                       hours=round(sum(session['hours'] for session in log['sessions']), 4),
                       curriculum=summary['curriculum'],
                       latest={key: value for key, value in (summary.get('last') or {}).items()
                               if key in ('episode_return', 'episode_return_mean', 'successes', 'steps')})
            (progress / 'progress.json').write_text(json.dumps(log, indent=2) + '\n')
    return log


def share(job: Job) -> None:
    """Commit the job's progress (weights through Git LFS) so another machine can resume it,
    plus any built evaluations, videos and plots."""
    git = lambda *args, **kw: subprocess.run(['git', *args], **kw)
    if git('config', '--get', 'filter.lfs.clean', capture_output=True).returncode != 0:
        raise SystemExit("Git LFS is not enabled here; run 'just install' first.")
    root = progress_directory(job).parent
    paths = [str(root / name) for name in ('progress', 'eval', 'render', 'plots') if (root / name).exists()]
    git('add', *paths, check=True)
    git('commit', '-m', summary_line(status([job])[0]), '--', *paths, check=True)


def build_train(job: Job, output: Path) -> None:
    """Finish the job's progress (resuming any earlier sessions), then publish it."""
    log = run_session(job)
    if not log['complete']:
        raise RuntimeError(f'{job.name}: training paused at {log["steps"]}/{log["total"]}; rerun to resume')
    with staged_directory(output) as stage:
        shutil.copytree(progress_directory(job), stage, ignore=shutil.ignore_patterns('resume'))
        (stage / 'provenance.json').write_text(json.dumps(provenance(job) | dict(sessions=log['sessions']),
                                                          indent=2) + '\n')


def status(jobs: list[Job]) -> list[dict]:
    rows = []
    for job in jobs:
        log = read_progress(job)
        published = (Path('data/jobs') / job.name / 'train').exists()
        total = job.training_spec()['training'].get('steps', 256)
        steps = log['steps'] if log else 0
        state = ('published' if published else 'complete' if log and log['complete']
                 else 'paused' if steps else 'not started')
        latest = (log or {}).get('latest', {})
        reward = latest.get('episode_return', latest.get('episode_return_mean'))
        rows.append(dict(job=job.name, algorithm=job.algorithm, state=state, steps=steps, total=total,
                         percent=100 * steps / total, hours=log['hours'] if log else 0.0,
                         sessions=len(log['sessions']) if log else 0,
                         machines=sorted({s['machine'] for s in log['sessions']}) if log else [],
                         level=log['curriculum']['level'] + 1 if log and 'curriculum' in log else None,
                         episode_return=reward))
    return rows


def format_status(rows: list[dict]) -> str:
    header = ('job', 'algorithm', 'state', 'transitions', '%', 'hours', 'sessions', 'level', 'return', 'machines')
    table = [header] + [(row['job'], row['algorithm'], row['state'], f"{row['steps']:,}/{row['total']:,}",
                         f"{row['percent']:.1f}", f"{row['hours']:.2f}", str(row['sessions']),
                         str(row['level'] or '-'),
                         '-' if row['episode_return'] is None else f"{row['episode_return']:.2f}",
                         ', '.join(row['machines']) or '-') for row in rows]
    widths = [max(len(line[i]) for line in table) for i in range(len(header))]
    return '\n'.join('  '.join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip() for line in table)


def summary_line(row: dict) -> str:
    return (f"progress({row['job']}): {row['steps']:,}/{row['total']:,} transitions ({row['percent']:.1f}%), "
            f"{row['hours']:.2f} h over {row['sessions']} session{'s' if row['sessions'] != 1 else ''}")


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
    from maton.rendering import render_episode
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
    session = commands.add_parser('session', help='Train a job from its progress; pause on --hours/--steps/Ctrl-C')
    session.add_argument('--config', type=Path, required=True)
    session.add_argument('--hours', type=float)
    session.add_argument('--steps', type=int)
    report = commands.add_parser('status', help='Training progress of every job')
    report.add_argument('--job', help='Only this job name')
    report.add_argument('--oneline', action='store_true', help='Commit-message summary')
    publish = commands.add_parser('share', help="Commit a job's progress so another machine can resume it")
    publish.add_argument('--job', required=True)
    preview = commands.add_parser('preview')
    preview.add_argument('--walker', type=Path, required=True)
    preview.add_argument('--pose', choices=('zero', 'neutral'), required=True)
    preview.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'train':
        build_train(Job.load(args.config), args.output)
    elif args.stage == 'evaluate':
        build_evaluation(Job.load(args.config), args.evaluation, args.train, args.output)
    elif args.stage == 'session':
        log = run_session(Job.load(args.config), args.hours, args.steps)
        print(summary_line(status([Job.load(args.config)])[0]) + (' (complete)' if log['complete'] else ''))
    elif args.stage == 'status':
        paths = [Path(f'configs/jobs/{args.job}.toml')] if args.job else sorted(Path('configs/jobs').rglob('*.toml'))
        rows = status([Job.load(path) for path in paths])
        print(summary_line(rows[0]) if args.oneline else format_status(rows))
    elif args.stage == 'share':
        share(Job.load(Path(f'configs/jobs/{args.job}.toml')))
    elif args.stage == 'render':
        build_render(args.episode, args.output)
    else:
        build_preview(args.walker, args.pose, args.output)


if __name__ == '__main__':
    main()
