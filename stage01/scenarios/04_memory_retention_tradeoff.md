# Scenario 4: Memory–retention trade-off (replay, budget)

## Objective

Demonstrate the **trade-off** between replay buffer size (or another memory budget) and achievable retention. Answer to **RQ3**: under what memory constraint is the CL solution worthwhile.

## Description

- **Replay:** Sweep over buffer size — e.g. 5%, 10%, 20% of the total number of samples from earlier tasks (or a fixed number of steps).
- **Memory cap:** Simulated limit (e.g. max 50 MB for model + replay). For different buffer sizes and, optionally, different network sizes, measure retention.
- **EWC:** The cost of EWC is essentially storing Fisher (a vector of importance weights) — constant in the number of tasks; it does not grow with time the way replay does.

## Metrics

- **Retention** (success rate / F1) on all tasks after the full sequence vs **buffer size** (and vs **total memory [MB]**).
- Plot: x-axis = memory, y-axis = retention; series: replay 5%, 10%, 20%, EWC, EWC+replay.

## What it demonstrates (CL strengths and weaknesses)

- **Strength:** At a small budget EWC may be the only option (no buffer); at a larger budget replay or the hybrid give better retention.
- **Weakness:** Replay requires memory proportional to the number of tasks; on an embedded device (the rover) the budget is limited — the recommendation "at budget X MB it pays off to use Y" therefore has practical value.

## Relation to the plan

- Phase 3.4, 5.4: hyperparameter sweep (buffer, λ).
- Phase 6.5: memory cap (RQ3); retention–memory plot.
