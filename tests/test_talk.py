from pathlib import Path

import pytest

from maton.talk import PLACEHOLDER, build, media


def deck(tmp_path, body):
    slides = tmp_path / 'slides.md'
    slides.write_text(body)
    template = tmp_path / 'template.html'
    template.write_text(f'<textarea>{PLACEHOLDER}</textarea>')
    return slides, template


def test_media_ignores_prose_mentions(tmp_path):
    slides, _ = deck(tmp_path, '![w](data/walkers/a.png)\n<video src="data/jobs/x/e.mp4">\nsee data/jobs/y/z.mp4\n')
    assert media(slides) == ['data/jobs/x/e.mp4', 'data/walkers/a.png']


def test_build_copies_media_and_escapes_markdown(tmp_path):
    slides, template = deck(tmp_path, '![w](data/walkers/a.png) A & B </textarea>')
    (tmp_path / 'data/walkers').mkdir(parents=True)
    (tmp_path / 'data/walkers/a.png').write_bytes(b'png')
    build(slides, template, tmp_path / 'out', root=tmp_path)
    page = (tmp_path / 'out/index.html').read_text()
    assert '](media/walkers/a.png)' in page and 'A &amp; B &lt;/textarea&gt;' in page
    assert (tmp_path / 'out/media/walkers/a.png').read_bytes() == b'png'


def test_build_rejects_missing_media_and_bad_template(tmp_path):
    slides, template = deck(tmp_path, '![w](data/missing.png)')
    with pytest.raises(FileNotFoundError):
        build(slides, template, tmp_path / 'out', root=tmp_path)
    template.write_text('no placeholder')
    with pytest.raises(ValueError):
        build(slides, template, tmp_path / 'out2', root=tmp_path)


def test_repository_deck_is_buildable_shape():
    assert Path('talk/template.html').read_text().count(PLACEHOLDER) == 1
    assert all(path.startswith('data/') for path in media(Path('talk/slides.md')))
