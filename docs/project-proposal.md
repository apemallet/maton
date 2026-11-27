## Project Proposal: Learning to Walk with Latent World Models

**Team Members:** Mason Bott, Simon Walker
**Team Name:** Team A

### Motivation

Teaching a multi-jointed robot to walk is a classic benchmark in continuous-control reinforcement learning because the state-action space is continuous, with often unstable dynamics, and small control errors that compound. Additionally, even with a successful walk, it's difficult to generalize to new terrain.

World-model-based reinforcement learning (World MBRL) offers a promising avenue to learn efficiently in sample-scarce scenarios while also providing an edge in generalization. By decoupling the world model from the policy, the model has potential to transfer learned locomotion dynamics across new terrain by predicting the consequences of candidate actions. But, model-free algorithms such as Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC) can also generalize, so whether a learned world model improves transfer is an empirical question.

### Research Question

Does learning with a world model improve zero-shot generalization to unseen conditions in a continuous bipedal walking task, compared with model-free reinforcement learning?

We will experiment with the following unseen conditions in order, as time allows:

1. Unseen obstacle courses
2. _(Stretch goal)_ Unseen physical dynamics parameters (such as gravity or friction)
3. _(Stretch goal)_ Novel walker configurations (differing amount of legs or other connections)

### Approach

To keep training feasible across multiple seeds on local hardware (an NVIDIA RTX 3080 and Apple Silicon MacBooks), we will build on a single standardized locomotion benchmark:

- **Environment (Tentative):** Gymnasium's `BipedalWalker-v3`, which features a 24-dimensional continuous state space.
- **World-Model Approach:** Start with **TD-MPC2** (Hansen et al., 2024). This is a decoder-free, World MBRL approach that predicts compact latent states, rewards, and values. We would train a lightweight implementation alongside model-free SAC and PPO baselines.
- **Training Courses:** First establish basic walking on `BipedalWalker-v3`, then use its obstacle-course variants or a controlled terrain generator to expose training to a curriculum of obstacle courses.
- **Experimental Protocol:** Each algorithm will be trained for the same number of environment steps across at least three random seeds, possibly more, as time permits. We will track learning curves, gait stability, and zero-shot evaluation. All algorithms will receive the same training course.
- **Zero-Shot Evaluation:** Cross-validation to unseen obstacle courses (first experiment). As time permits, vary physics parameter and walker configuration.

### Expected Deliverables

1. **Pipeline:** Code training TD-MPC2 and model-free baselines (SAC/PPO) on the BipedalWalker task, with separate training and held-out obstacle courses.
2. **Empirical Comparison:** Multi-seed plots of zero-shot course completion, episode return, forward distance, and falls on familiar and unseen courses, supported by learning curves and wall-clock cost measurements.
3. **Visuals:** Side-by-side rendered videos comparing the learned walking gaits of TD-MPC2 vs. SAC/PPO on obstacle courses.
4. **Stretch Goal (Broader Generalization):** If time permits, we will expand evaluation to more challenging scenarios including robustness to physics-simulation parameter changes.
