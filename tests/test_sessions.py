from pathlib import Path
import shutil
import signal

import pytest

from maton.jobs import Job
from maton.sessions import Pause
from maton.workflow import format_status, run_session, status, summary_line


def test_pause_on_steps_and_first_interrupt():
    pause = Pause(steps=10)
    assert not pause(9) and pause(10)
    assert Pause(hours=0)(0)
    pause = Pause()
    with pause.catching_interrupt():
        signal.raise_signal(signal.SIGINT)  # os.kill terminates the process on Windows
        assert pause(0)


@pytest.fixture
def project(tmp_path, monkeypatch):
    shutil.copytree('configs', tmp_path / 'configs')
    shutil.copy('uv.lock', tmp_path)  # provenance digests it
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_sessions_accumulate_progress_and_reject_changed_settings(project):
    job = Job.load('configs/jobs/smoke-ppo.toml')
    log = run_session(job, steps=8)
    assert (log['steps'], log['complete'], len(log['sessions'])) == (8, False, 1)
    row = status([job])[0]
    assert row['state'] == 'paused' and summary_line(row).startswith('progress(smoke-ppo): 8/16 transitions (50.0%)')
    assert 'paused' in format_status([row])
    log = run_session(job)
    assert log['complete'] and [s['steps'] for s in log['sessions']] == [[0, 8], [8, 16]]
    path = Path('configs/jobs/smoke-ppo.toml')
    path.write_text(path.read_text().replace('steps = 16', 'steps = 32'))
    with pytest.raises(ValueError, match='settings changed'):
        run_session(Job.load(path))


def test_continued_job_uses_published_source_and_keeps_its_own_progress(project):
    from maton.workflow import build_train
    source = Job.load('configs/jobs/smoke-ppo.toml')
    build_train(source, Path('data/jobs/smoke-ppo/train'))
    path = Path('configs/jobs/harder.toml')
    path.write_text('extends = "smoke-ppo"\ncontinue_from = {job = "smoke-ppo"}\n'
                    '[training]\nsteps = 24\nlearning_rate = 0.00003\n')
    job = Job.load(path)
    first = run_session(job, steps=8)
    assert first['steps'] == 8 and not first['complete']
    assert first['total'] == 24
    final = run_session(job)
    assert final['steps'] == 24 and final['complete']
    assert [s['steps'] for s in final['sessions']] == [[0, 8], [8, 24]]
