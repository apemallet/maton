<p class="kicker">Maton · status update · Nov 2026</p>

# Maton

### Teaching procedurally built walkers to cross obstacle courses with world models

<p class="small">DreamerV3 vs PPO on MuJoCo MJX · 15 minutes</p>

Note:
30 s. One sentence on why: can a learned world model make locomotion over a
curriculum cheaper and more general than model-free RL?

---

<p class="kicker">The question</p>

## Does a world model generalise better zero-shot than model-free RL?

- **Proposal:** TD-MPC2 vs SAC/PPO, ≥ 3 seeds, evaluated on unseen conditions
- **Unseen conditions, in order:** obstacle courses → physics (gravity, friction) → walker configurations
- **Built so far:** our own MJX environment (instead of BipedalWalker-v3), with DreamerV3 vs PPO

<div class="verdict">

Today: what works, what is fragile, and how to get back to the proposed comparison.

</div>

Note:
1 min. From docs/project-proposal.md. The MJX environment already covers all
three unseen-condition axes: procedural courses, friction, TOML walkers.

---

<p class="kicker">Setup</p>

## A planar walker, a lidar, and a six-level course

<div class="cols wide-left">
<div>

![biped walker preview](data/walkers/biped/neutral.png)

</div>
<div>

- MuJoCo **MJX**, 100 Hz, PD position control
- Bodies defined in TOML: biped, quadruped, …
- **53-float observation:** Fourier joint angles and rates, root velocity, foot contacts, 9 lidar rays
- **Reward:** upright-weighted progress, plus a finish bonus
- **Curriculum:** flat → blocks → recesses → 10–20 m gaps; 5 finishes to advance

</div>
</div>

Note:
1.5 min. Falls do not end an episode; the walker may recover until timeout.
That design choice comes back in the diagnosis.

---

<p class="kicker">Infrastructure</p>

## One TOML file produces every artifact

<div class="pipeline">
<span class="card">job + walker + curriculum</span><span class="arrow">→</span>
<span class="card">train/</span><span class="arrow">→</span>
<span class="card">eval/{name}/</span><span class="arrow">→</span>
<span class="card">render/*.mp4</span>
</div>

- Training outputs are **protected and fingerprinted**: changed settings need a new job name
- Evaluations save **exact physics trajectories**, so videos rebuild without the policy
- Provenance: git revision, source and lockfile digests
- 102 tests; both trainers run end to end

Note:
1 min. Keep this short. The point: experiments are now cheap to define and
reproducible, which is what the next steps need.

---

<p class="kicker">Result 1</p>

## It walks, but only on seed 0

<div class="cols">
<div>

<p class="stat">10 / 0 / 0</p>

Level-1 (flat) finishes out of 10 held-out courses, seeds 0 / 1 / 2, friction 1.6, 524k steps. **Identical for PPO and Dreamer.**

With the current Fourier observation: Dreamer 10 / 0 / 0, PPO 0 / 0 / 0.

No run has passed level 1.

</div>
<div>

<div class="todo">VIDEO: biped-dreamer-s0 · level-1 · episode 0 (success)<br>next to the seed-1 failure</div>

</div>
</div>

Note:
2 min. Both algorithms fail the same way. That points at the environment and
protocol, not at either algorithm.

---

<p class="kicker">Result 2</p>

## Small changes flip the outcome

| Change (level-1 finishes / 10) | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| PPO · std 0.2, ep 512, friction 0.9 | <span class="pass">9</span> | <span class="part">6</span> | <span class="pass">10</span> |
| PPO · std 0.2 → 0.4 | – | <span class="fail">0</span> | <span class="fail">0</span> |
| PPO · episode 512 → 640 / 800 | <span class="fail">0 on 5 of 6 runs</span> | | |
| Dreamer · maxstd 0.3, lr 1e-4, ep 800 | <span class="pass">7, 9</span> | – | – |
| Dreamer · same, ep 512 | <span class="fail">0</span> (too slow) | – | – |
| Dreamer · friction 1.6 | <span class="pass">10</span> | <span class="fail">0</span> | <span class="fail">0</span> |
| Both · friction 2.5 or 4.0 | – | <span class="fail">0</span> | – |
| PPO · Fourier observation | <span class="fail">0</span> | <span class="fail">0</span> | <span class="fail">0</span> |
| Dreamer · Fourier observation | <span class="pass">10</span> | <span class="fail">0</span> | <span class="fail">0</span> |

<p class="small">From docs/experiments.md. Dashes: not run. Training budgets are not recorded for every row.</p>

Note:
2 min. Exploration noise, episode length and friction each flip pass/fail.
Most cells were run on seed 0 only.

---

<p class="kicker">Result 3</p>

## Tuning one seed of one method hurt the other

- Best multi-seed result: **PPO 25/30** (std 0.2, episode 512, default friction)
- Friction 1.6 was picked from a **Dreamer seed-0** sweep → PPO fell to **10/30** (budgets may differ)
- The Fourier-observation commit also removed height, previous action and course position → PPO **0/30**. Two changes, one result: **confounded**

<div class="verdict">

Lesson: no setting changes on fewer than 5 seeds, and one variable at a time.

</div>

Note:
1.5 min. This is the most useful finding for the team: the process, not the
model, cost us the robust PPO result.

---

<p class="kicker">Diagnosis</p>

## The bottleneck is the experiment protocol, not yet the algorithm

1. **Under-scaled compute.** On an M1 Pro CPU, a 524k-step run takes **≈ 45 min for Dreamer** (≈ 200 steps/s, one environment, shrunk model) and **≈ 8 min for PPO** (16 environments). Runs are cheap, so the fragility comes from tiny budgets, not a lack of hardware. MJX-scale PPO uses thousands of environments.
2. **Falls never end episodes.** A fallen walker burns the rest of the episode, hence the episode-length tug of war (PPO wants 512, Dreamer 800).
3. **Single-seed decisions.** 3 seeds × 10 episodes; most changes judged on seed 0.

Note:
2 min. Each point maps to one step of the plan.

---

<p class="kicker">Algorithm choice</p>

## Back to the proposal: TD-MPC2

| | DreamerV3 | TD-MPC2 | PPO at scale | FastTD3 |
| --- | --- | --- | --- | --- |
| Built for state-based continuous control | ok | **strong** | standard | **strong** |
| Uses MJX parallelism | not in our harness | single env + planning | **native** | **native** |
| Decoder for "dreams" | **yes** | no (needs a probe) | no | no |
| Fits our JAX stack | **pinned upstream** | PyTorch (JAX ports exist) | **ours** | PyTorch |
| Hyperparameter robustness | good | **one config, 100+ tasks** | needs tuning | good |

<div class="verdict">

**Yes, as an addition, not a swap.** A compact JAX port (~600 lines) under the same jobs and evaluations; keep PPO, and decide whether Dreamer stays.

</div>

Note:
2 min. TD-MPC2 was the proposal's method and fits the task shape. It won't fix
the protocol problems on its own, and it has no decoder for the "dreams" idea.

---

<p class="kicker">Plan</p>

## Each step has a decision rule

| Step | Decision rule |
| --- | --- |
| 1 · Protocol: ≥ 5 seeds, success rate with CIs, GPU profile | No setting ships on fewer than 5 seeds |
| 2 · Scale PPO: 1–4k environments, 50M+ steps | All seeds walk → the environment is learnable; compare sample efficiency |
| 3 · Falls: terminate or truncate on a fall | Seed variance drops → keep it |
| 4 · TD-MPC2 (JAX) vs PPO (vs Dreamer), trained on levels 1–3 | Proposal metric: zero-shot completion on held-out courses, plus learning curves and wall-clock |

Note:
1.5 min.

---

<p class="kicker">Discussion</p>

## What we need from you

- **Compute:** the 5-seed rerun of today's jobs fits on a laptop (≈ 4.5 h). Scaled PPO and full-size Dreamer need a GPU: how many hours?
- **Scope:** the proposal says TD-MPC2 + SAC/PPO. Keep Dreamer as an extra arm (and the "dreams" visuals), or drop it?
- **Target:** what headline result do we want in four weeks?

Note:
1 min, then questions.
