# Experiment Plan: Planetary Rover — CL for Autonomy and Security

*Step by step: objectives, methods, success criteria.*

---

## Scope and relation to the thesis topic

Thesis topic: **"Analysis of continual learning techniques in models that learn new information on the fly."** The plan realises this objective in two layers: **(1) Phase 0** — review and taxonomy of CL techniques (regularization, replay, architectural), analysis criteria, and justification for the techniques selected for the experiments; **(2) Phases 3, 5, 7** — empirical analysis of the chosen techniques (**EWC**, **replay**, **EWC+replay**, optionally LwF) together with a comparative synthesis and recommendations. On-the-fly learning is realised through the **sequential introduction of new tasks** (terrains T1→T2→…, threat classes C1→C2→…) without full retraining. The Phase 0 deliverable (document `docs/analiza_technik_cl.md`) provides the basis for the theoretical chapter of the thesis.

---

## Overview of phases

| Phase | Name | Main objective | Deliverable |
|-------|------|----------------|-------------|
| 0 | Review of CL techniques | Taxonomy and criteria for analysing CL techniques; justification of EWC/replay selection | Document: taxonomy, comparative table, mapping onto "on-the-fly learning" |
| 1 | Environment and data | Rover simulation + terrains + threat signals | Env + dataset ready for training |
| 2 | Autonomy baseline | RL navigation without CL on a sequence of terrains | Baseline metrics, reference point |
| 3 | CL in autonomy | Analysis of EWC and replay on a sequence of terrains (on-the-fly learning) | Retention, technique comparison, configuration selection |
| 4 | Security baseline | Anomaly/threat detection without CL | Baseline detection metrics |
| 5 | CL in security | Analysis of EWC and replay on a sequence of threat classes (on-the-fly learning) | Retention, technique comparison, configuration selection |
| 6 | Fusion: multi-task CL | Shared representation, both task streams | Architecture comparison, results table |
| 7 | Technique analysis and report | Synthesis of CL technique analysis, answers to RQ1–RQ3, conclusions for models learning on the fly | Analytical report, tables, recommendations |

---

## Phase 0: Review and analysis of CL techniques (preparation for the experiments)

### 0.1 Objective
Formulate the framework for **analysing continual learning techniques** and justify the choice of experimental methods in the context of models that learn new information on the fly. The deliverable of this phase provides the basis for the theoretical chapter and ensures coherence with the thesis topic.

### 0.2 Step-by-step tasks

| Step | Task | Method | Success criterion |
|------|------|--------|-------------------|
| 0.1 | Taxonomy of CL techniques | Literature review: partition into (1) importance-based regularization (EWC, SI, MAS), (2) replay / memory (buffer, generative replay), (3) architectural (Progressive Neural Networks, PackNet). Brief description of each category. | Document with a table: category → representative methods → main characteristics. |
| 0.2 | Mapping onto "on-the-fly learning" | Define how sequential task introduction (new terrains, new classes) realises "learning new information on the fly"; reference to the classification of CL scenarios (task-incremental, domain-incremental). | One section: scenario in the thesis = X, why it requires techniques that limit forgetting. |
| 0.3 | Criteria for technique analysis | Establish comparison dimensions: retention vs forgetting, memory cost, implementation complexity, applicability in RL vs supervised settings. | List of criteria used in Phases 3, 5, 7. |
| 0.4 | Justification for choosing EWC and replay | Why EWC and replay (and EWC+replay): fit with rover constraints, no architectural change, comparability with the literature. Optionally: LwF as a third technique "if time permits". | Short subsection: "Selection of techniques for empirical analysis". |

### 0.3 Deliverable
- File or subsection: `docs/analiza_technik_cl.md` (or equivalent in the repository) — taxonomy, comparative table of techniques, analysis criteria, justification for choosing EWC/replay. Content to be reused in the theoretical chapter of the thesis.

---

## Phase 1: Environment and data

### 1.1 Objective
Have a working rover simulation and defined terrains and threat/anomaly types so that task sequences for CL can be reproduced.

### 1.2 Step-by-step tasks

| Step | Task | Method | Success criterion |
|------|------|--------|-------------------|
| 1.1 | Choice of simulation platform | MuJoCo (Isaac Gym / Gymnasium) or a simple 2D environment (e.g. PyGame + occupancy grid). Rover: differential drive, state = position + goal + scan (lidar-like rays). | Robot executes actions in the environment, receives observations and reward. |
| 1.2 | Definition of 3–4 terrains | Different layouts: map A (simple), B (narrow passages), C (dynamic obstacles?), D (different traction/slip). Each terrain = different map file / physics parameters. | Terrain can be loaded by ID, reset to (start, goal) on the chosen map. |
| 1.3 | Gym interface | `step`, `reset`, `observation_space`, `action_space`. Reward: +1 for the goal, penalties for collision/time. Optionally SPL. | Compatibility with SB3/CleanRL (e.g. `VecEnv`). |
| 1.4 | Definition of "threats" / anomalies | **Option A:** Simulated telemetry streams with labels (normal / anomaly_type_1, 2, 3). **Option B:** In-environment "events" (e.g. sensor damage, spoofed command) with labels. Dataset generation: N episodes per type. | Dataset (X, y) or generator with `next_batch()`; 3–4 threat classes. |
| 1.5 | Data pipeline | Script that loads terrain 1..K and returns (obs, reward, done, info). For security: script that loads a batch (obs_telemetry, label). | A single function `get_terrain(terrain_id)` and `get_security_batch(threat_type_id)`. |

### 1.3 Technical requirements (minimum)
- Python 3.8+
- Choice: Stable-Baselines3 or CleanRL (PPO/SAC)
- Repository with folders: `env/`, `data/`, `scripts/`, `configs/`

---

## Phase 2: Autonomy baseline (navigation without CL)

### 2.1 Objective
Determine how good an RL model trained **separately on each terrain** (and jointly on all of them) is — a reference point for forgetting and for CL.

### 2.2 Step-by-step tasks

| Step | Task | Method | Success criterion |
|------|------|--------|-------------------|
| 2.1 | Per-terrain training | For each terrain T: train PPO/SAC from scratch until convergence (e.g. success rate ≥ 90% on T). Save policy π_T. | K models (K = 3–4), each strong on its own terrain. |
| 2.2 | Joint training | A single policy on all terrains (random terrain selection at reset). Train until convergence. Policy π_joint. | Success rate on each terrain ≥ 80%. |
| 2.3 | Baseline metrics | For each π: success rate, SPL, mean time-to-goal on each terrain. Table: terrain × (π_1, π_2, …, π_joint). | Table + plot; π_T is clearly excellent on T and weak elsewhere (forgetting). |
| 2.4 | Fine-tuning baseline | Sequence T1→T2→T3→T4: train on T1, then fine-tune on T2 (without CL), then T3, T4. Evaluate on all terrains after every phase. | Forgetting curve: retention on T1 drops after learning T2, T3, T4. |

### 2.3 Deliverable
- Results file: `results/baseline_autonomy.json` (or CSV) + plotting script.
- Brief description: "Joint achieves X%, fine-tuning drops on T1 to Y% after 4 terrains".

---

## Phase 3: CL in autonomy — technique analysis (retention across terrains)

### 3.1 Objective
Carry out an **analysis of CL techniques** (EWC, replay, EWC+replay) in an **on-the-fly learning** scenario: the model receives terrains T1→T2→T3→T4 sequentially without full retraining. Compare retention against fine-tuning and joint training; select a configuration for the fusion phase.

### 3.2 Step-by-step tasks

| Step | Task | Method | Success criterion |
|------|------|--------|-------------------|
| 3.1 | EWC implementation | After training on T_k: compute Fisher Information (FI) on data from T_k; loss = L_new + λ Σ_i FI_i (θ_i − θ*_i)². Integration into the training loop (e.g. SB3 wrapper or a custom loop). | Retention on T1 after training on T2 is higher than with fine-tuning (λ sweep). |
| 3.2 | Replay implementation | Experience buffer from previous terrains (e.g. 10–20% capacity); when training on T_k, mix in samples from the buffer. Uniform or reservoir sampling. | Retention on T1 higher than fine-tuning; dependence on buffer size documented. |
| 3.3 | Sequential experiment | A single sequence: T1 → T2 → T3 → T4. For each method (fine-tune, EWC, replay, EWC+replay): sequential training, after each phase evaluate on all T1..T_k. | Success rate matrix: phase × terrain; retention curves. |
| 3.4 | Hyperparameter sweep | EWC: λ ∈ {1e2, 1e3, 1e4}. Replay: buffer size ∈ {5%, 10%, 20%}. Choose the configuration that yields retention ≥ 80% on T1 after all phases. | Table of λ / buffer vs retention; selected configuration saved to config. |
| 3.5 | Comparison with baseline | Plot: retention on T1 (and possibly T2) vs phase for fine-tune, EWC, replay, joint. | Answer to RQ1 for autonomy: "Yes, EWC/replay limit forgetting". |
| 3.6 | *(Optional)* LwF | Learning without Forgetting: distillation to the "old" model when training on a new terrain. Comparison with EWC/replay. | Extension of the technique analysis; only if time permits. |

### 3.3 Deliverable
- Code: `ewc.py`, `replay_buffer.py`, script `run_cl_autonomy.py`.
- Results: `results/cl_autonomy_retention.csv`, plots; **comparative table of techniques** (fine-tune vs EWC vs replay vs EWC+replay): retention, memory cost, complexity. Entry in plan.md: "Recommended configuration: EWC λ=X, replay Y%".

---

## Phase 4: Security baseline (threat detection without CL)

### 4.1 Objective
Determine detection quality when training **per class** and when training **jointly**; a reference point for forgetting in a stream of threat classes.

### 4.2 Step-by-step tasks

| Step | Task | Method | Success criterion |
|------|------|--------|-------------------|
| 4.1 | Detector architecture | Classifier (e.g. MLP or 1D CNN) on the input: telemetry / event-feature vector. Output: softmax over classes (normal + threat types). Loss: cross-entropy. | The model trains on (X, y) and returns class probabilities. |
| 4.2 | Data | For each threat class: train/val split. Class balancing or loss weighting. | Dataset ready for `DataLoader` / training loop. |
| 4.3 | Per-class training (and joint) | Model M_k on classes 1..k only (incrementally) or a separate model per class. Model M_joint on all classes. | Metrics: accuracy, macro F1 on each class. |
| 4.4 | Fine-tuning baseline | Class sequence: C1 → C2 → C3 → C4. Fine-tune without CL. After each phase: evaluate on C1..C_k. | Forgetting curve: accuracy on C1 drops after adding C2, C3, C4. |
| 4.5 | Metrics | Table: class × (M_joint, M_finetune_after_phase_4). Confusion matrix for fine-tune after the last phase. | Deliverable: `results/baseline_security.json` + short description. |

### 4.3 Deliverable
- Script `train_security_baseline.py`, `results/baseline_security.json`.

---

## Phase 5: CL in security — technique analysis (retention across threat classes)

### 5.1 Objective
Carry out an **analysis of CL techniques** (EWC, replay, EWC+replay) in an **on-the-fly learning** scenario: the model receives threat classes C1→C2→C3→C4 sequentially without full retraining. Compare retention against fine-tuning and joint training; select a configuration for the fusion phase.

### 5.2 Step-by-step tasks

| Step | Task | Method | Success criterion |
|------|------|--------|-------------------|
| 5.1 | EWC for the classifier | Analogous to Phase 3: Fisher after each phase; regularization in the loss. Same scheme, different network (classifier). | Retention on C1 higher than under fine-tuning. |
| 5.2 | Replay for security | Buffer of samples (obs, label) from previous classes; mix into training. | Higher retention; dependence on buffer size. |
| 5.3 | Sequential experiment | Sequence C1→C2→C3→C4 with EWC, replay, EWC+replay. Evaluation after each phase on all classes. | Accuracy/F1 matrix: phase × class; retention curves. |
| 5.4 | Sweep and configuration selection | λ (EWC), buffer size. Choose a configuration that satisfies e.g. "F1 on C1 after phase 4 ≥ 0.8". | Answer to RQ1 for security; recommendation saved to config. |
| 5.5 | Comparison with baseline | Plot: accuracy/F1 on C1 vs phase (fine-tune vs EWC vs replay vs joint). | Deliverable: `results/cl_security_retention.csv`, plots. |
| 5.6 | *(Optional)* LwF | As in Phase 3: LwF for the classifier; comparison with EWC/replay. | Extension of the technique analysis; only if time permits. |

### 5.3 Deliverable
- Reuse the same EWC/replay modules as in Phase 3 (task-agnostic abstraction). Results: `results/cl_security_*.csv`. **Comparative table of techniques** (as in Phase 3) for the security domain.

---

## Phase 6: Fusion — multi-task CL (autonomy + security)

### 6.1 Objective
Determine whether a shared representation (encoder) + two "heads" (navigation, detection) with a single CL mechanism is worthwhile compared to two separate models; answer to RQ2.

### 6.2 Step-by-step tasks

| Step | Task | Method | Success criterion |
|------|------|--------|-------------------|
| 6.1 | Shared architecture | Encoder E(obs) → embedding. Navigation head: E(obs) → action. Security head: E(obs_sec) → threat class. Input: obs from the simulation + obs_sec from the security dataset (if the format differs — a projection layer to a shared dimension). | A single network with two outputs; training can be alternating or joint. |
| 6.2 | Mixed-task scenario | Task sequence: [T1, C1, T2, C2, T3, C3, T4, C4] or [T1, T2, C1, C2, T3, T4, C3, C4]. At each step: train only on the current task using EWC/replay (shared between E and both heads). | Retention measured on terrains and threat classes after the full sequence. |
| 6.3 | Fusion metrics | Success rate on T1..T4; F1/accuracy on C1..C4. Memory (parameter count), inference time (encoder + one head). | Table: (shared architecture + CL) vs (separate autonomy model + separate security model). |
| 6.4 | Comparison: shared vs separate | **Separate:** two models (navigation with CL from Phase 3, detection with CL from Phase 5). **Shared:** one encoder + two heads with CL. The same memory budgets (e.g. parameter cap). | Answer to RQ2: whether the shared representation is worthwhile (retention vs memory vs complexity). |
| 6.5 | Memory limit (RQ3) | Simulated memory cap: e.g. max 50 MB for replay + model. Sweep buffer size and network size. Plot: retention (autonomy + security) vs memory. | Recommendation: "At budget X MB it pays off to Y". |

### 6.3 Deliverable
- Code: `models/shared_encoder.py`, `run_fusion_cl.py`. Results: `results/fusion_shared_vs_separate.csv`, retention–memory plot.

---

## Phase 7: CL technique analysis and final report

### 7.1 Objective
Synthesise the **analysis of continual learning techniques** in the context of models learning new information on the fly; collect the answers to RQ1–RQ3 and formulate conclusions and recommendations.

### 7.2 Step-by-step tasks

| Step | Task | Method | Success criterion |
|------|------|--------|-------------------|
| 7.1 | Technique-analysis synthesis | **Comparative table of CL techniques** (EWC vs replay vs EWC+replay): retention (autonomy + security), memory cost, implementation complexity, when to use. Reference to the criteria from Phase 0. | One section "Comparative analysis of techniques" with a table and conclusions: which technique fits which on-the-fly learning scenario. |
| 7.2 | RQ1 | Summary: retention in autonomy (Phase 3) and in security (Phase 5) with EWC/replay vs fine-tune. A sentence such as: "CL methods effectively limit forgetting in both domains (retention values X vs Y)." | One or two sentences in plan.md or the report. |
| 7.3 | RQ2 | Summary of the Phase 6 table: shared vs separate models. A sentence: "The shared representation is worthwhile when …" or "Separate models are better when …". | Conclusion from the table. |
| 7.4 | RQ3 | Summary of the retention–memory plot: at which budget which configuration (EWC, replay, hybrid) gives acceptable retention. | Recommendation in plan.md. |
| 7.5 | Final tables and plots | One "Results" section with tables and links to the result files. | Easy reproduction of results from the repository. |
| 7.6 | Limitations and future work | Briefly: simulation, no sim-to-real; possible extensions (more terrains, more classes, other CL techniques, real telemetry). | 3–5 bullet points. |

### 7.3 Deliverable
- **Analytical report**: synthesis of the CL technique analysis (comparative table, recommendations for "on-the-fly learning"). Updated `plan.md` with a "Results and conclusions" section, or a separate `report.md` / `wnioski.md`.

---

## Dependencies between phases

```
Phase 0 (review of CL techniques)
   │
   ▼
Phase 1 ──► Phase 2 ──► Phase 3
   │           │
   │           └──────────────┐
   ▼                          ▼
Phase 4 ──► Phase 5 ──► Phase 6 (fusion) ──► Phase 7 (technique analysis + report)
```

- Phase 0 can be run in parallel with the start of Phase 1 (e.g. literature review).
- Phases 2 and 4 can be run in parallel after Phase 1.
- Phase 6 requires Phases 3 and 5 to be complete (the selected CL configurations).

---

## Time estimate (indicative)

| Phase | Estimate |
|-------|----------|
| 0 | ~1 week (literature review, taxonomy) |
| 1 | 2–3 weeks |
| 2 | 1–2 weeks |
| 3 | 2–3 weeks |
| 4 | 1–2 weeks |
| 5 | 1–2 weeks |
| 6 | 2–3 weeks |
| 7 | ~1–2 weeks (technique-analysis synthesis + report) |

*Total: about 11–17 weeks part-time; shorter under full-time work and with a simplified environment.*

---

## Result files (checklist)

- [ ] `docs/analiza_technik_cl.md` (Phase 0: taxonomy, analysis criteria, justification for choosing EWC/replay)
- [ ] `results/baseline_autonomy.json`
- [ ] `results/cl_autonomy_retention.csv` (+ plots + comparative table of techniques)
- [ ] `results/baseline_security.json`
- [ ] `results/cl_security_retention.csv` (+ plots + comparative table of techniques)
- [ ] `results/fusion_shared_vs_separate.csv` (+ retention–memory plot)
- [ ] `configs/best_cl_autonomy.yaml`, `configs/best_cl_security.yaml`
- [ ] Final report with the technique-analysis synthesis (Phase 7) and a "Results and conclusions" section

After completing the experiments, fill in the "Results and conclusions" section at the end of this file.
