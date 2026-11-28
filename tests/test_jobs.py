import json
from pathlib import Path

import pytest

from maton.algorithms import ALGORITHMS
from maton.jobs import Evaluation, Job, fingerprint
from maton.workflow import staged_directory


def copy_job(tmp_path):
    source = Path('configs/jobs/smoke-ppo.toml')
    contents = source.read_text()
    for kind in ('walker', 'curriculum'):
        job = Job.load(source)
        contents = contents.replace(str(job.settings[kind]), job.reference(kind).resolve().as_posix())
    path = tmp_path / 'test-job.toml'
    path.write_text(contents)
    return path


def test_all_jobs_resolve_with_only_their_selected_trainer():
    for path in Path('configs/jobs').glob('*.toml'):
        job = Job.load(path)
        assert job.config().resolved()['training']['steps'] > 0
        assert (set(ALGORITHMS) - {job.algorithm}).isdisjoint(job.settings)


def test_stage_fingerprints_and_immutable_training(tmp_path):
    path = copy_job(tmp_path)
    original = Job.load(path)
    train = tmp_path / 'train'
    train.mkdir()
    signature = fingerprint(original.training_spec())
    (train / 'provenance.json').write_text(json.dumps({'training_fingerprint': signature}))
    path.write_text(path.read_text().replace('render = [0]', 'render = []'))
    rendered = Job.load(path)
    assert rendered.evaluations['level-1'].spec() == original.evaluations['level-1'].spec()
    rendered.assert_training_identity(train)
    path.write_text(path.read_text().replace('episodes = 1', 'episodes = 2'))
    evaluated = Job.load(path)
    assert evaluated.evaluations['level-1'].spec() != original.evaluations['level-1'].spec()
    evaluated.assert_training_identity(train)
    path.write_text(path.read_text().replace('steps = 16', 'steps = 32'))
    with pytest.raises(ValueError, match='new job name'):
        Job.load(path).assert_training_identity(train)


def test_shared_structure_changes_training_identity(tmp_path):
    path = copy_job(tmp_path)
    job = Job.load(path)
    structure = tmp_path / 'walker.toml'
    structure.write_text(job.reference('walker').read_text())
    path.write_text(path.read_text().replace(job.reference('walker').as_posix(), structure.as_posix()))
    before = fingerprint(Job.load(path).training_spec())
    # Formatting changes do not invalidate training; semantic changes do.
    structure.write_text(structure.read_text() + '\n# comment\n')
    assert fingerprint(Job.load(path).training_spec()) == before
    structure.write_text(structure.read_text().replace('name = "basic"', 'name = "renamed"'))
    assert fingerprint(Job.load(path).training_spec()) != before


@pytest.mark.parametrize('options', [dict(episodes=0), dict(episodes=True), dict(level=0),
                                     dict(seed=-1), dict(render=[1]), dict(render=[0, 0]),
                                     dict(checkpoint='../weights'), dict(checkpoint='step-12')])
def test_bad_evaluations(options):
    with pytest.raises(ValueError):
        Evaluation(**options)


@pytest.mark.parametrize('change', [lambda s: s.replace('algorithm = "ppo"', 'algorithm = "unknown"'),
                                   lambda s: s + '\n[typo]\nx = 1\n',
                                   lambda s: s.replace('level = 1', 'level = 999'),
                                   lambda s: s.replace('[ppo]', '[dreamer]')])
def test_bad_jobs(tmp_path, change):
    path = copy_job(tmp_path)
    path.write_text(change(path.read_text()))
    with pytest.raises((ValueError, TypeError)):
        Job.load(path)


def test_checkpoint_paths():
    train = Path('data/train')
    ppo = Job.load('configs/jobs/smoke-ppo.toml')
    dreamer = Job.load('configs/jobs/smoke-dreamer.toml')
    assert ppo.checkpoint(train, Evaluation()) == train / 'policy.eqx'
    assert dreamer.checkpoint(train, Evaluation()) == train / 'dreamer.pkl'
    checkpoint = Evaluation(checkpoint='step-00000008')
    assert ppo.checkpoint(train, checkpoint) == train / 'checkpoints/step-00000008.eqx'
    assert dreamer.checkpoint(train, checkpoint) == train / 'dreamer-00000008.pkl'


def test_staging_publishes_only_successful_artifacts(tmp_path):
    output = tmp_path / 'train'
    with pytest.raises(RuntimeError):
        with staged_directory(output) as stage:
            stage.mkdir()
            (stage / 'partial').write_text('partial')
            raise RuntimeError('failed job')
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []
    with staged_directory(output) as stage:
        stage.mkdir()
        (stage / 'summary.json').write_text(json.dumps({'checkpoint': str(stage / 'policy.eqx')}))
    assert json.loads((output / 'summary.json').read_text())['checkpoint'] == str(output / 'policy.eqx')
    with pytest.raises(FileExistsError):
        with staged_directory(output) as stage:
            stage.mkdir()
    assert (output / 'summary.json').exists()


def test_inheritance_merges_tables_and_tracks_declaring_paths(tmp_path):
    base = copy_job(tmp_path)
    child = tmp_path / 'harder.toml'
    child.write_text('extends = "test-job"\nseed = 7\n[training]\nsteps = 32\n'
                     '[evaluations.level-1]\nrender = []\n')
    leaf = tmp_path / 'hardest.toml'
    leaf.write_text('extends = "harder"\n[training]\nlearning_rate = 0.00002\n')
    job = Job.load(leaf)
    assert job.settings['seed'] == 7
    assert job.settings['training']['steps'] == 32
    assert job.settings['training']['checkpoint_every'] == Job.load(base).settings['training']['checkpoint_every']
    assert job.config().resolved()['training']['learning_rate'] == 0.00002
    assert job.evaluations['level-1'].render == ()
    assert job.config_paths == (base, child, leaf)
    assert job.reference('walker').resolve() == Job.load(base).reference('walker').resolve()
    before = fingerprint(job.training_spec())
    base.write_text(base.read_text().replace('steps = 16', 'steps = 64'))
    assert fingerprint(Job.load(leaf).training_spec()) == before  # overridden
    base.write_text(base.read_text().replace('checkpoint_every = 8', 'checkpoint_every = 4'))
    assert fingerprint(Job.load(leaf).training_spec()) != before


@pytest.mark.parametrize('field', ['extends = "test-job"', 'continue_from = {job = "test-job"}',
                                  'extends = "child"\n', 'continue_from = {job = "child"}'])
def test_job_dependency_cycles(tmp_path, field):
    base = copy_job(tmp_path)
    base.write_text(field + '\n' + base.read_text())
    (tmp_path / 'child.toml').write_text('extends = "test-job"\n')
    with pytest.raises(ValueError, match='cycle'):
        Job.load(base)


@pytest.mark.parametrize('field', ['extends = ["test-job"]', 'extends = "../test-job"',
                                  'continue_from = "test-job"', 'continue_from = {}',
                                  'continue_from = {job = "test-job", checkpoint = "step-12"}',
                                  'continue_from = {job = "test-job", typo = 1}'])
def test_invalid_job_dependencies(tmp_path, field):
    copy_job(tmp_path)
    child = tmp_path / 'child.toml'
    child.write_text(field + '\n')
    with pytest.raises(ValueError):
        Job.load(child)


def test_continuation_paths_and_identity(tmp_path):
    base = copy_job(tmp_path)
    path = tmp_path / 'child.toml'
    path.write_text('extends = "test-job"\ncontinue_from = {job = "test-job", checkpoint = "step-00000008"}\n')
    job = Job.load(path)
    assert job.source_train() == Path('data/jobs/test-job/train')
    assert job.initial_checkpoint() == Path('data/jobs/test-job/train/checkpoints/step-00000008.eqx')
    before = fingerprint(job.training_spec())
    base.write_text(base.read_text().replace('episodes = 1', 'episodes = 2'))
    assert fingerprint(Job.load(path).training_spec()) == before
    path.write_text(path.read_text().replace('step-00000008', 'final'))
    assert Job.load(path).initial_checkpoint() == Path('data/jobs/test-job/train/policy.eqx')
    assert fingerprint(Job.load(path).training_spec()) != before
    path.write_text(path.read_text().replace('extends = "test-job"', 'extends = "test-job"\nalgorithm = "sac"'))
    with pytest.raises(ValueError, match='same algorithm'):
        Job.load(path)
