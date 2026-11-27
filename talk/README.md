# Talk

Status deck: reveal.js, written in Markdown.

```sh
just talk        # builds data/talk/ via Snakemake, then opens index.html
```

- Edit `slides.md`. Slides are separated by a line containing only `---`; text after `Note:` is speaker notes (press `S`).
- Media: reference artifacts by path, e.g. `![w](data/walkers/biped/neutral.png)` or
  `<video src="data/jobs/biped-dreamer-s0/render/level-1/episode-0000.mp4" data-autoplay loop muted></video>`.
  Snakemake builds every referenced artifact first (a video triggers its job's training), then copies it into the deck.
  Paths in plain prose are ignored.
- `.todo` boxes mark visuals that still need runs.
- Layout helpers in `template.html`: `.cols`, `.stat`, `.card`, `.verdict`, `.pass`/`.part`/`.fail`.
- reveal.js loads from a CDN; present online, or export a PDF with `index.html?print-pdf` in Chrome.
