# Stage 01: CL experiment scenarios

This directory contains the **defined scenarios** used to verify the strengths and limitations of continual learning in the context of a planetary rover (navigation autonomy + threat detection). The scenarios form the basis of Phases 2–7 of `docs/plan.md`.

## Objective

- **Strengths of CL:** limitation of catastrophic forgetting, retention on previous tasks under sequential learning, the ability to learn "on the fly" without full retraining.
- **Weaknesses / limitations of CL:** sensitivity to task ordering, memory cost (replay), retention–buffer-capacity trade-off, difficulty in multi-task settings with a shared representation.

## Scenarios (files in `scenarios/`)

| ID | File | Description | What it demonstrates |
|----|------|-------------|----------------------|
| 1 | [01_autonomy_sequential_terrains.md](scenarios/01_autonomy_sequential_terrains.md) | Sequence of terrains T1→T2→T3→T4 for RL navigation | Forgetting (fine-tune) vs retention (EWC/replay) |
| 2 | [02_security_sequential_classes.md](scenarios/02_security_sequential_classes.md) | Sequence of threat classes C1→C2→C3→C4 for detection | Analogue in classification; comparison of CL techniques |
| 3 | [03_order_sensitivity.md](scenarios/03_order_sensitivity.md) | Different task orderings (easy→hard, reverse, random) | CL sensitivity to ordering; recommendations |
| 4 | [04_memory_retention_tradeoff.md](scenarios/04_memory_retention_tradeoff.md) | Replay buffer size vs retention; memory budget | Memory–retention trade-off; RQ3 |
| 5 | [05_fusion_multi_task.md](scenarios/05_fusion_multi_task.md) | Shared representation: autonomy + security in a single stream | RQ2: shared vs separate models |

## Mapping onto plan phases

- **Phases 2, 3** → scenarios 1, 3, 4 (autonomy).
- **Phases 4, 5** → scenarios 2, 3, 4 (security).
- **Phase 6** → scenario 5 (fusion).
- **Phase 7** → synthesis of results from all scenarios.

## Metrics shared across the analysis

- **Retention:** success rate (autonomy) / accuracy or F1 (security) on earlier tasks after training on subsequent ones.
- **Forgetting:** drop in retention relative to the baseline (model trained only on that task).
- **Memory:** model size + replay buffer size (where applicable).
- **Ordering:** influence of the order of tasks on final retention.
