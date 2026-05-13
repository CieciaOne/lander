# Technology alternatives to the plan (Python vs Rust, simulation, CL)

A brief analysis: would **Rust** or another stack make it easier to deliver the plan (Phases 0–7) without harming the quality of the research?

---

## Rust — does it make sense for this plan?

**Conclusion: no.** For this master's thesis, Rust would not make the plan easier or better to deliver.

### What the plan requires

- **Simulation:** MuJoCo (or Isaac Gym), a Gymnasium env, rover + terrains.
- **RL:** PPO/SAC (Stable-Baselines3 or CleanRL), training loop, policy save/load.
- **CL:** EWC (Fisher, regularization in the loss), replay buffer, optionally LwF; integration with the RL loop and with the classifier's training loop.
- **Experiments:** many runs, λ / buffer sweeps, tables, plots, report.

### What exists in Rust

- **MuJoCo:** Bindings exist (e.g. `mujoco-rs`, `rusty_mujoco`) — Rust-side simulation is possible.
- **RL:** `gymnasium` in Rust is at an early stage (0.0.1); `sb3-burn` (Rust + Burn) is an SB3 equivalent but lacks a mature env ecosystem and ready CL integrations; `operant` provides fast vec envs with a Python interface.
- **CL (EWC, replay):** No standard libraries. EWC and replay in the literature and in published code are almost exclusively Python (PyTorch/JAX). In Rust one would have to implement Fisher computation, the replay buffer, and the loss integration from scratch (or wrap C / Python) — over the same time budget that is pure overhead with no payoff for the research questions.
- **Deep learning:** Rust does not have an ecosystem at the level of PyTorch / JAX (autodiff, optimizers, reproducibility, ready recipes for RL / CL). Burn / candle are progressing but do not deliver a ready "SB3 + EWC + MuJoCo" stack or a comparable basis for citations.

### Summary on Rust

- **Time:** Porting the env + an SB3-like stack + EWC + replay + experiments to Rust would extend the schedule (11–17 weeks in the plan) without a corresponding gain in the quality of the CL analysis.
- **Comparability:** Results in the thesis should be comparable with the literature (same metrics, standard frameworks). Python + Gymnasium + MuJoCo + SB3 / CleanRL is the standard; Rust in this context complicates citation and comparison.
- **Where Rust would make sense:** When the principal bottleneck is, for example, millions of CPU simulation steps and only the env is moved to Rust (e.g. through Operant / Python bindings) — that is a possible later optimisation, not a wholesale replacement of the stack for the purposes of this thesis.

---

## Alternatives that may make delivery easier / better (within Python)

### 1. CleanRL instead of (or alongside) Stable-Baselines3

- **Plus:** Single-file PPO / SAC implementations, simpler integration of EWC and replay directly in the loop (adding a term to the loss, buffer in one place).
- **Minus:** Fewer "out of the box" wrappers than SB3; eval, logging, and saving have to be wired up manually.
- **Recommendation:** Consider CleanRL for Phases 2–3 (baseline + CL in autonomy) to obtain a working EWC / replay faster; keep SB3 as an option or for cross-checks.

### 2. Isaac Gym (instead of MuJoCo) — only if simulation time becomes the bottleneck

- **Plus:** 2–3 orders of magnitude faster training (GPU, massive env parallelism); the plan already mentions it.
- **Minus:** Different physics (PhysX), different API; porting the env from MuJoCo is a separate piece of work; CL papers frequently use MuJoCo — easier citation and comparison.
- **Recommendation:** Stay with MuJoCo at the start. Isaac Gym becomes meaningful only when training time / step count becomes a real problem (e.g. multi-seed sweeps).

### 3. MJX (MuJoCo + JAX)

- **Plus:** MuJoCo on GPU via JAX, same MJCF models; speed-up without changing the engine.
- **Minus:** Training would have to be in JAX (e.g. a custom PPO loop or a library such as Brax), not SB3; fewer ready recipes for CL.
- **Recommendation:** Optional for "fast simulation" experiments without changing the main pipeline (the main baseline remains Python + MuJoCo CPU + SB3 / CleanRL).

### 4. Keep the current stack (Gymnasium + MuJoCo + SB3)

- Smallest schedule risk, fully aligned with the plan and the literature (CL in RL, EWC, replay).
- EWC and replay can be added as wrappers / callbacks around SB3 or as a separate loop (e.g. CleanRL) — without changing language.

---

## Final recommendation

- **Do not** port the plan to Rust — it would neither simplify the process nor improve the answers to RQ1–RQ3.
- **Do** stay with Python: Gymnasium, MuJoCo, Stable-Baselines3 (or CleanRL for a quicker EWC / replay integration), PyTorch for the classifier and CL in Phases 4–5.
- **Optional:** CleanRL for EWC / replay in autonomy; Isaac Gym or MJX only if a hard training-time problem emerges.

If a specific bottleneck appears during the work (e.g. only the env, only logging), a point-wise alternative such as a faster env in Rust with a Python API can still be considered — without rewriting the whole pipeline in Rust.
