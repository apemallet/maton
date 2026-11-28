"""Exercise Snakemake's DAG and rerun policy with cheap stand-in builders."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def workflow(tmp_path):
    shutil.copy('Snakefile', tmp_path)
    shutil.copy('uv.lock', tmp_path)
    shutil.copytree('configs', tmp_path / 'configs')
    shutil.copytree('profiles', tmp_path / 'profiles')
    # Isolate source edits used to exercise the manual rebuild policy.
    shutil.copytree('src', tmp_path / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    binary = tmp_path / 'bin'
    binary.mkdir()
    runner = binary / 'uv'
    runner.write_text(f'#!{sys.executable}\n' + '''
import json
from pathlib import Path
import sys
from maton.jobs import Job, fingerprint

# `uv run python -m maton.workflow <stage> ...` or `uv run python -m maton.<module> ...`
workflow = sys.argv[4] == 'maton.workflow'
stage = sys.argv[5] if workflow else sys.argv[4]
rest = sys.argv[6:] if workflow else sys.argv[5:]
args = dict(zip(rest[::2], rest[1::2]))
output = Path(args['--output'])
if stage == 'train':
    job = Job.load(args['--config'])
    output.mkdir(parents=True)
    (output / 'provenance.json').write_text(json.dumps({'training_fingerprint': fingerprint(job.training_spec())}))
elif stage == 'evaluate':
    output.mkdir(parents=True)
    (output / 'summary.json').write_text('{}')
else:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('stand-in artifact')
''')
    runner.chmod(0o755)
    environment = dict(os.environ, PATH=f'{binary}{os.pathsep}{os.environ["PATH"]}',
                       XDG_CACHE_HOME=str(tmp_path / 'cache'))
    environment.pop('PYTHONPATH', None)

    def run(*args, success=True):
        result = subprocess.run([sys.executable, '-m', 'snakemake', '--cores', '1',
                                 '--runtime-source-cache-path', str(tmp_path / 'source-cache'), *args],
                                cwd=tmp_path, env=environment, capture_output=True, text=True)
        output = result.stdout + result.stderr
        if success:
            assert result.returncode == 0, output
        else:
            assert result.returncode != 0, output
        return output

    return tmp_path, run


def test_dag_and_stage_invalidation(workflow):
    root, run = workflow
    first = run()
    assert 'localrule train:' in first and 'localrule evaluate:' in first and 'localrule render:' in first
    assert 'localrule plot:' in first
    assert 'Nothing to be done' in run('-n')
    path = root / 'configs/jobs/smoke-ppo.toml'
    path.write_text(path.read_text().replace('render = [0]', 'render = []'))
    assert 'Nothing to be done' in run('-n')
    path.write_text(path.read_text().replace('episodes = 1', 'episodes = 2'))
    planned = run('-n')
    assert 'rule evaluate:' in planned and 'rule train:' not in planned
    run()
    # Requesting an additional video uses existing trajectories only.
    planned = run('data/jobs/smoke-ppo/render/level-1/episode-0001.mp4', '-n')
    assert 'rule render:' in planned and 'rule evaluate:' not in planned and 'rule train:' not in planned
    run('data/jobs/smoke-ppo/render/level-1/episode-0002.mp4', '-n', success=False)
    # Configuration identity survives loss of local Snakemake metadata.
    shutil.rmtree(root / '.snakemake')
    path.write_text(path.read_text().replace('steps = 16', 'steps = 32'))
    assert 'new job name' in run('-n', success=False)


def test_protected_training_and_preview_target(workflow):
    root, run = workflow
    run('data/jobs/smoke-ppo/train')
    assert 'protected' in run('data/jobs/smoke-ppo/train', '--forcerun', 'train', success=False).lower()
    source = root / 'src/maton/ppo.py'
    source.write_text(source.read_text() + '\n# changed training implementation\n')
    assert 'Nothing to be done' in run('data/jobs/smoke-ppo/train')
    preview = 'data/walkers/quadruped/neutral.png'
    run(preview)
    assert (root / preview).is_file()
    assert 'Nothing to be done' in run(preview, '-n')


def test_source_changes_need_explicit_rebuilds(workflow):
    root, run = workflow
    run()
    preview = 'data/walkers/quadruped/neutral.png'
    run(preview)
    for source in (root / 'src').rglob('*.py'):
        source.write_text(source.read_text() + '\n# implementation changed\n')
    assert 'Nothing to be done' in run('-n')
    assert 'Nothing to be done' in run(preview, '-n')

    video = 'data/jobs/smoke-ppo/render/level-1/episode-0000.mp4'
    episode = root / 'data/jobs/smoke-ppo/eval/level-1/summary.json'
    original = episode.read_bytes(), episode.stat().st_mtime_ns
    rebuilt = run(video, '-f')
    assert 'localrule render:' in rebuilt
    assert 'localrule train:' not in rebuilt and 'localrule evaluate:' not in rebuilt
    assert (episode.read_bytes(), episode.stat().st_mtime_ns) == original
    assert 'Nothing to be done' in run('-n')

    # Data dependencies still invalidate the video without a force flag.
    newer = (root / video).stat().st_mtime + 10
    os.utime(episode.parent / '.snakemake_timestamp', (newer, newer))
    planned = run(video, '-n')
    assert 'rule render:' in planned
    assert 'rule train:' not in planned and 'rule evaluate:' not in planned


def test_inherited_continuation_dag_and_semantic_changes(workflow):
    root, run = workflow
    jobs = root / 'configs/jobs'
    child = jobs / 'harder.toml'
    child.write_text('extends = "smoke-ppo"\ncontinue_from = {job = "smoke-ppo"}\n[training]\nsteps = 32\n')
    target = 'data/jobs/harder/train'
    planned = run(target, '-n')
    assert 'wildcards: job=smoke-ppo' in planned and 'wildcards: job=harder' in planned
    built = run(target)
    assert built.index('wildcards: job=smoke-ppo') < built.index('wildcards: job=harder')
    assert 'Nothing to be done' in run(target, '-n')
    base = jobs / 'smoke-ppo.toml'
    base.write_text(base.read_text().replace('render = [0]', 'render = []'))
    assert 'Nothing to be done' in run(target, '-n')
    base.write_text(base.read_text().replace('steps = 16', 'steps = 24'))
    assert 'new job name' in run(target, '-n', success=False)


def test_inheritance_alone_does_not_train_parent(workflow):
    root, run = workflow
    (root / 'configs/jobs/variant.toml').write_text('extends = "smoke-ppo"\nseed = 3\n')
    planned = run('data/jobs/variant/train', '-n')
    assert 'wildcards: job=variant' in planned
    assert 'wildcards: job=smoke-ppo' not in planned


def test_advanced_job_and_profile_include_parent_training(workflow):
    _, run = workflow
    planned = run('--config', 'job=advanced/biped-tdmpc2-NML-s0',
                  '--profile', 'profiles/advanced', '-n')
    assert 'wildcards: job=biped-tdmpc2-v2-s0' in planned
    assert 'wildcards: job=advanced/biped-tdmpc2-NML-s0' in planned
    assert 'mem_mb=16384' in planned and 'runtime=4320' in planned
    assert 'evaluation=level-6' not in planned
    assert 'evaluation=no-mans-land' in planned
