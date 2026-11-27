<p class="kicker">Maton · status update · Nov 2026</p>

# Maton

### Teaching procedurally built walkers to cross obstacle courses with world models

<p class="small">TD-MPC2, SAC, DreamerV3 and PPO on MuJoCo MJX · 15 minutes</p>

Note:
30 s. One sentence on why: can a learned world model make locomotion over a
curriculum cheaper and more general than model-free RL?

---

<p class="kicker">The question</p>

## Does a world model generalise better zero-shot than model-free RL?

- **Proposal:** TD-MPC2 vs SAC/PPO, ≥ 3 seeds, evaluated on unseen conditions
- **Unseen conditions, in order:** obstacle courses → physics (gravity, friction) → walker configurations
- **Built:** our own MJX environment (instead of BipedalWalker-v3) with TD-MPC2, SAC, PPO and DreamerV3

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

## One TOML file per experiment, trained in sessions

<div class="pipeline">
<span class="card">job + walker + curriculum</span><span class="arrow">→</span>
<span class="card">train/</span><span class="arrow">→</span>
<span class="card">eval/{name}/</span><span class="arrow">→</span>
<span class="card">render/*.mp4</span>
</div>

- **Resumable sessions:** `just train job 2` trains for 2 hours; Ctrl-C pauses; resumes match uninterrupted runs bit for bit
- **Shared progress:** `just share job` commits weights via Git LFS; `just status` shows steps, hours, machines
- Published training is **protected and fingerprinted**; evaluations save exact physics trajectories
- Four trainers behind one registry; 139 tests

Note:
1 min. Keep this short. The point: experiments are cheap to define, reproducible,
and can be split across laptops and the GPU box.

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

## Back to the proposal: TD-MPC2 and SAC are built

| Trainer | Kind | Cost per 524k-step run (M1 Pro CPU) | Notes |
| --- | --- | --- | --- |
| PPO | model-free, on-policy | **≈ 8 min** | 16 envs |
| SAC | model-free, off-policy | **≈ 50 min** | proposal baseline; shares TD-MPC2's replay and update ratio |
| DreamerV3 | world model, decoder | **≈ 45 min** | shrunk model, 1 env; keeps the "dreams" idea alive |
| TD-MPC2 | world model, planning | **≈ 1.5 days** | official 5M model; compute-bound → RTX 3080 |

<div class="verdict">

Same environment, budget, evaluations and videos for all four. TD-MPC2 trades compute for samples, so judge it on **success vs environment steps**, and run it on the GPU.

</div>

Note:
2 min. The TD-MPC2 port is ~330 lines of Equinox on a shared off-policy driver.
Planning is batched across environments; evaluation plans without noise.

---

<p class="kicker">Plan</p>

## Each step has a decision rule

| Step | Decision rule |
| --- | --- |
| 1 · Protocol: ≥ 5 seeds, success rate with CIs, GPU profile | No setting ships on fewer than 5 seeds |
| 2 · Scale PPO: 1–4k environments, 50M+ steps | All seeds walk → the environment is learnable; compare sample efficiency |
| 3 · Falls: terminate or truncate on a fall | Seed variance drops → keep it |
| 4 · TD-MPC2 vs SAC vs PPO (vs Dreamer) on the 3080, levels 1–3 | Proposal metric: zero-shot completion on held-out courses, plus learning curves and wall-clock |

Note:
1.5 min.

---

<p class="kicker">Discussion</p>

## What we need from you

- **Compute:** PPO, SAC and Dreamer seeds fit on laptops; TD-MPC2 needs the 3080. Who runs which sessions?
- **Scope:** keep Dreamer as a fourth arm (and the "dreams" visuals), or drop it?
- **Target:** what headline result do we want in four weeks?

Note:
1 min, then questions.
