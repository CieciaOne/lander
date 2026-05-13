# Scenario 5: Fusion — multi-task CL (autonomy + security)

## Objective

Determine whether a **shared representation** (encoder) with two "heads" (navigation, threat detection) and a single CL mechanism is worthwhile compared to **two separate models** (one for autonomy with CL, one for security with CL). Answer to **RQ2** at the architectural level.

## Description

- **Shared architecture:** Encoder E(obs) → embedding; navigation head: E(obs) → action; security head: E(obs_sec) or E(obs) → threat class. Under a mixed task sequence (e.g. T1, C1, T2, C2, …) training is on the current task only, with EWC / replay **shared** across E and both heads.
- **Separate architecture:** Two models — navigation (with the best CL configuration from Phase 3), detection (with the best configuration from Phase 5); each with its own replay / EWC.

## Metrics

- **Retention:** Success rate on T1..T4 and F1 / accuracy on C1..C4 after the full sequence.
- **Memory:** Parameter count (shared vs sum of the two models).
- **Inference time:** Encoder + one head vs two separate forward passes.

## What it demonstrates (CL strengths and weaknesses)

- **Strength:** A shared representation may yield smaller total memory and better generalisation across tasks; one CL mechanism simplifies deployment.
- **Weakness:** Risk of interference between tasks (navigation vs detection); separate models may achieve better retention on their own task under the same budget. The tabulated comparison settles the question.

## Relation to the plan

- Phase 6: multi-task CL fusion; shared vs separate table; recommendation for RQ2.
