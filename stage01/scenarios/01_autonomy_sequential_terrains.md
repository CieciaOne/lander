# Scenario 1: Autonomy — sequence of terrains (T1→T2→T3→T4)

## Objective

Demonstrate **catastrophic forgetting** under plain fine-tuning and the **advantage of CL techniques** (EWC, replay, EWC+replay) in maintaining retention on earlier terrains while sequentially learning new ones.

## Description

- **Task:** Rover navigation from start to goal with obstacle avoidance (Gymnasium + MuJoCo environment).
- **Terrains:** 3–4 different layouts (map A — simple, B — narrow passages, C — harder, D — optionally different traction/slip).
- **Sequence:** the model first trains on T1 until convergence, then **without full retraining** receives T2, then T3, T4. In each phase only training on the current terrain is allowed (plus, optionally, replay from earlier terrains).

## Variants to compare

| Variant | Description | Expectation |
|---------|-------------|-------------|
| **Fine-tune** | After each phase, fine-tune on the new terrain without any protection | Strong forgetting: retention on T1 drops after T2, T3, T4 |
| **EWC** | Fisher regularization after each phase (λ sweep) | Retention on T1 higher than fine-tune; dependence on λ |
| **Replay** | Experience buffer from previous terrains (5–20% capacity), mixed into training | Higher retention; dependence on buffer size |
| **EWC+replay** | Combination of both methods | Highest retention at a reasonable budget |
| **Joint (upper bound)** | Training from scratch on all terrains jointly | Reference: best possible retention without the "sequential" constraint |

## Metrics

- **Success rate** on every terrain after every phase (matrix: phase × terrain).
- **SPL** (Success weighted by Path Length), mean time-to-goal.
- **Retention curve:** e.g. success rate on T1 after phases 1, 2, 3, 4 — for each variant.

## What it demonstrates (CL strengths and weaknesses)

- **Strength:** EWC / replay visibly limit forgetting relative to fine-tune; on-the-fly learning is feasible without retraining on all terrains.
- **Weakness:** Joint may still be better when all data is available; CL requires choosing λ / buffer size and incurs a memory cost (replay) or a risk of underfitting (λ too large).
