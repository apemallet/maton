"""Build the reveal.js talk: inline Markdown slides, copy referenced data/ media."""

import html
from pathlib import Path
import re
import shutil

# Only real references (`src="data/..."`, `](data/...)`) count; prose mentions do not.
MEDIA = re.compile(r'(?<=src=")data/[^"]+|(?<=\]\()data/[^)\s]+')
PLACEHOLDER = '{{slides}}'


def media(slides: Path) -> list[str]:
    """Repository-relative data/ artifacts the deck needs; Snakemake builds them first."""
    return sorted(set(MEDIA.findall(slides.read_text(encoding='utf-8'))))


def build(slides: Path, template: Path, output: Path, root: Path = Path('.')) -> None:
    page = template.read_text(encoding='utf-8')
    if page.count(PLACEHOLDER) != 1:
        raise ValueError(f'{template} must contain {PLACEHOLDER} exactly once')
    output.mkdir(parents=True, exist_ok=True)
    for path in media(slides):
        source = root / path
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output / 'media' / Path(path).relative_to('data')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    text = MEDIA.sub(lambda match: 'media/' + match.group(0).removeprefix('data/'), slides.read_text(encoding='utf-8'))
    # The Markdown lives in a <textarea>; escaping keeps `</textarea>` and `&` literal.
    (output / 'index.html').write_text(page.replace(PLACEHOLDER, html.escape(text, quote=False)), encoding='utf-8')


def main() -> None:
    import argparse
    import tempfile
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('slides', 'template', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    # Stage beside the destination; publish only a complete deck.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.talk-', dir=args.output.parent) as temporary:
        stage = Path(temporary) / 'deck'
        build(args.slides, args.template, stage)
        stage.rename(args.output)


if __name__ == '__main__':
    main()
