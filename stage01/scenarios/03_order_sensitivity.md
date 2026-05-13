# Scenario 3: Sensitivity to task ordering

## Objective

Determine whether the **order in which tasks are introduced** (terrains or classes) significantly affects the final retention and quality of the CL model. This is a typical **weakness** of CL: results depend on the order of tasks.

## Description

- **Domains:** Autonomy (terrains T1..T4) and/or security (classes C1..C4).
- **Ordering variants:**
  - **Easy → hard:** simple terrains / classes first, gradually harder ones afterwards.
  - **Hard → easy:** the reverse ordering.
  - **Random:** one or several random permutations.

## Metrics

- Final **retention on each task** after the full sequence finishes.
- **Mean retention** (e.g. mean success rate on T1..T4) as a function of ordering.
- Optionally: time to convergence on each task as a function of ordering.

## What it demonstrates (CL strengths and weaknesses)

- **Strength:** If CL is robust to ordering within some range, it can be applied flexibly.
- **Weakness:** If retention depends strongly on ordering, this is a practical limitation — e.g. it requires a fixed deployment order or a strategy for choosing the order (curriculum).

## Relation to the plan

- Phase 2/3 (autonomy): E4 in the plan — "Map ordering: easy→hard vs hard→easy vs random".
- Results feed into Phase 7 (recommendations: when ordering matters).
