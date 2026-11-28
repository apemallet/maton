from pathlib import Path

from maton.jobs import Job, NAME, fingerprint
from maton.talk import media


wildcard_constraints:
    job=NAME,
    evaluation=NAME,
    walker=NAME,
    pose="zero|neutral",
    episode=r"[0-9][0-9][0-9][0-9]+"


def job_config(wildcards):
    return f"configs/jobs/{wildcards.job}.toml"


def job(wildcards):
    return Job.load(job_config(wildcards))


def training_signature(wildcards):
    experiment = job(wildcards)
    experiment.assert_training_identity(Path(f"data/jobs/{wildcards.job}/train"))
    return fingerprint(experiment.training_spec())


def evaluation_signature(wildcards):
    return job(wildcards).evaluations[wildcards.evaluation].spec()


def render_input(wildcards):
    experiment = job(wildcards)
    protocol = experiment.evaluations[wildcards.evaluation]
    index = int(wildcards.episode)
    if wildcards.episode != f"{index:04d}" or not 0 <= index < protocol.episodes:
        raise ValueError("Episode must be a canonical, existing evaluation index")
    root = f"data/jobs/{wildcards.job}/eval/{wildcards.evaluation}"
    return root


def targets(name):
    experiment = Job.load(f"configs/jobs/{name}.toml")
    return [f"data/jobs/{name}/train", f"data/jobs/{name}/plots/training.png"] + [
        f"data/jobs/{name}/eval/{evaluation}" for evaluation in experiment.evaluations
    ] + [
        f"data/jobs/{name}/render/{evaluation}/episode-{index:04d}.mp4"
        for evaluation, protocol in experiment.evaluations.items() for index in protocol.render
    ]


rule all:
    input:
        targets(config.get("job", "smoke-ppo"))


rule train:
    input:
        # The semantic signature below tracks training settings only.
        config=lambda w: ancient(job_config(w)),
        walker=lambda w: ancient(str(job(w).reference("walker"))),
        curriculum=lambda w: ancient(str(job(w).reference("curriculum"))),
        lock="uv.lock"
    output:
        protected(directory("data/jobs/{job}/train"))
    params:
        signature=training_signature
    threads: 1
    resources:
        mem_mb=4096,
        runtime=240,
        training_slots=1
    shell:
        "uv run python -m maton.workflow train --config {input.config:q} --output {output:q}"


rule evaluate:
    input:
        train="data/jobs/{job}/train",
        config=lambda w: ancient(job_config(w))
    output:
        directory("data/jobs/{job}/eval/{evaluation}")
    params:
        protocol=evaluation_signature
    threads: 1
    resources:
        mem_mb=2048,
        runtime=60
    shell:
        "uv run python -m maton.workflow evaluate --config {input.config:q} "
        "--evaluation {wildcards.evaluation:q} --train {input.train:q} --output {output:q}"


rule render:
    input:
        episode=render_input
    output:
        ensure("data/jobs/{job}/render/{evaluation}/episode-{episode}.mp4", non_empty=True)
    params:
        episode=lambda w: f"data/jobs/{w.job}/eval/{w.evaluation}/episode-{w.episode}"
    threads: 1
    resources:
        mem_mb=1024,
        runtime=30
    shell:
        "uv run python -m maton.workflow render --episode {params.episode:q} --output {output:q}"


rule plot:
    input:
        train="data/jobs/{job}/train"
    output:
        ensure("data/jobs/{job}/plots/training.png", non_empty=True)
    threads: 1
    resources:
        mem_mb=1024,
        runtime=5
    shell:
        "uv run python -m maton.plots --train {input.train:q} --output {output:q}"


rule preview:
    input:
        walker="configs/walkers/{walker}.toml"
    output:
        ensure("data/walkers/{walker}/{pose}.png", non_empty=True)
    threads: 1
    resources:
        mem_mb=1024,
        runtime=5
    shell:
        "uv run python -m maton.workflow preview --walker {input.walker:q} "
        "--pose {wildcards.pose:q} --output {output:q}"


rule talk:
    input:
        slides="talk/slides.md",
        template="talk/template.html",
        # Every src="data/..." or ](data/...) reference; Snakemake builds them first.
        media=lambda w: media(Path("talk/slides.md"))
    output:
        directory("data/talk")
    threads: 1
    resources:
        mem_mb=512,
        runtime=5
    shell:
        "uv run python -m maton.talk --slides {input.slides:q} "
        "--template {input.template:q} --output {output:q}"
