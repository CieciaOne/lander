<!-- Rewritten 2026-05-12 — the original Polish version focused on a Roomba-style
     robot and a comparison between RL and classical algorithms; this contradicted
     the thesis topic, which is purely an analysis of continual learning techniques
     applied to a Mars rover. The Roomba framing has therefore been replaced. -->

# Continual Learning for a Mars Rover — One-Pager

*A short companion to `docs/plan.md`. Authoritative content lives in the plan.*

---

## Thesis topic and motivation

Thesis topic: **"Analysis of continual learning techniques in models that learn new information on the fly."**

Planetary rovers operate in environments that cannot be enumerated in advance: terrain morphology, lighting, traction, and the catalogue of telemetry anomalies evolve over the lifetime of the mission. A model that is to remain useful on Mars therefore has to **acquire new information incrementally** — adding terrains to its navigation repertoire, adding anomaly classes to its threat detector — **without losing what it already knows**. The central technical obstacle is **catastrophic forgetting**: a neural network fine-tuned on a new task tends to overwrite the parameters that encoded the previous one.

Continual learning (CL) studies algorithms that mitigate this effect. This thesis carries out an empirical **analysis of selected CL techniques** in two coupled experimental tracks that share infrastructure (rover MJCF model, MuJoCo simulator, replay/EWC modules):

1. **Autonomy track (RL):** PPO/SAC policy on a differential-drive Mars rover with rocker-bogie suspension, trained sequentially on terrains T1→T2→T3→T4 in MuJoCo.
2. **Security track (supervised):** a classifier trained sequentially on threat / anomaly classes C1→C2→C3→C4 from rover telemetry.

A later phase (Phase 6) studies a shared encoder for both streams (multi-task CL).

---

## Research questions

| ID | Research question |
|----|-------------------|
| **RQ1** | Do CL techniques (EWC, replay buffer, EWC+replay hybrid) effectively limit catastrophic forgetting in (a) RL navigation on a sequence of Mars-rover terrains and (b) supervised threat classification on a sequence of threat classes? |
| **RQ2** | Is a shared representation (one encoder + two task-specific heads) for both autonomy and security worthwhile compared to two separate models with their own CL mechanism, in terms of retention, total memory, and inference cost? |
| **RQ3** | Under realistic on-board constraints (replay buffer size, task ordering, memory budget), which CL configuration (EWC-only, replay-only, EWC+replay) gives acceptable retention on a rover-grade compute envelope? |

RQ1–RQ3 are answered by Phases 3, 5, 6 and synthesised in Phase 7 of `docs/plan.md`.

---

## Continual learning — technique categories

CL methods are commonly grouped into three families:

| Category | Idea | Representative methods | Memory cost |
|----------|------|------------------------|-------------|
| **Regularization** | Penalise changes to parameters that were important for previous tasks. | EWC (Fisher Information), SI, MAS | Constant in number of tasks (one importance vector). |
| **Replay / rehearsal** | Keep (or generate) samples from previous tasks and mix them into the training of the current task. | Experience replay (uniform / reservoir), generative replay | Grows with the buffer size; linear in the number of tasks if not capped. |
| **Architectural** | Allocate new parameters / sub-networks per task; freeze or mask previous ones. | Progressive Neural Networks, PackNet, expert gates | Grows with the number of tasks; often requires task-ID at inference. |

---

## Techniques chosen for empirical analysis

The thesis focuses on **regularization** and **replay**, plus their hybrid. Rationale:

- **EWC** is a canonical, architecture-preserving regularizer; its memory cost is constant in the number of tasks, which matches the on-board memory budget of a rover.
- **Replay buffer** (uniform or reservoir) is the strongest single-technique baseline in the CL literature and is straightforward to share between the RL track (state/action transitions) and the supervised track ((obs, label) tuples).
- **EWC + replay (hybrid)** combines a cheap parameter-space anchor with a small set of representative samples; the literature reports that the hybrid often dominates either component alone at a moderate memory cost.
- Architectural methods are deliberately out of scope: they require task IDs at inference and a growing parameter count, which is awkward for an autonomous rover that should not know which terrain it is currently on.
- *(Optional)* **LwF** is kept as a third technique to add if time permits.

For each technique, the analysis dimensions used in Phases 3, 5, and 7 are:
**retention vs forgetting**, **memory cost**, **implementation complexity**, **applicability across RL and supervised settings**, and **sensitivity to task ordering**.

---

## Deliverables tied to this one-pager

- `docs/plan.md` — the authoritative experiment plan (Phases 0–7).
- `stage01/README.md` and `stage01/scenarios/*.md` — five scenarios that operationalise the research questions.
- `docs/analiza_technik_cl.md` *(to be written in Phase 0)* — full taxonomy, comparative table, and justification of the chosen techniques; this one-pager is its short companion.
