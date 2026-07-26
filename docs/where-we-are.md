# Where we are — project status & restart guide

Concise "start here when we come back" summary. Full chronological detail is in
[`decision-log.md`](decision-log.md); this is the current state + next actions.

Last updated: 2026-07 (perception axis + full curriculum + wobble fix session).

## The two deliverables

1. **CL method comparison** (the thesis core) — 7 CL methods on a task sequence,
   measuring forgetting/retention vs a joint baseline. **DONE (N=1):**
   `scenario_14_skill_sequence`, EWC best. See `results/scenario_14_skill_sequence/
   comparison.png` + `summary.csv`. Ranking: **ewc 0.74 > hybrid 0.67 > distill
   0.64 > mas 0.63 > replay 0.61 > l2 0.52 > naive 0.29** (avg retention).
   Gap: **only N=1 seed** — needs multi-seed for error bars.

2. **Perception realism × CL** (new this session) — how much observation realism
   costs, and whether it interacts with CL. **DONE (N=1).**

## Perception modes (env axis, orthogonal to CL method)

Selectable via `run_scenario.py --perception {privileged,reactive,slam}`; any
scenario can run under each (obstacle phases exercise it). Env flags:
`obstacle_obs_mode`, `geo_heading_source`. All keep obs dim fixed (drop-in).

- **privileged** — ground-truth obstacle AABBs + geo_heading on the true map (upper bound).
- **reactive** — lidar + goal only (honest mapless).
- **slam** — obstacles discovered online into an `OccupancyMap`; geo_heading planned
  on the DISCOVERED map. Realistic "discover & plan" (occupancy mapping w/ sim odometry;
  not full SLAM). `bend only after obstacles discovered` (else grid-quantised bearing
  collapses obstacle-free-phase training).

### Key results
- **Single obstacle task** (`scenario_15`, 3M): **slam 0.73 ≈ privileged 0.70 ≫
  reactive 0.43**. Discovering the map matches the ground-truth cheat, both beat lidar-only.
- **Full curriculum** (`scenario_16`, loco→tracking→avoidance→combined, 1M/phase):
  EWC keeps **forgetting ≈ 0** in all perceptions; retention **slam 0.62 ≥ reactive
  0.58 > privileged 0.42** (privileged an N=1 outlier — see open items).

## scenario_16 full curriculum (the CL+perception task)

4 phases, ONE obstacle-capable env config (lidar+vw) so the policy carries weights:
`RC_c_locomotion → RC_c_tracking → RC_c_avoidance → RC_c_combined`. Per-task final
success (slam, 1M/phase, angvel penalty on): loco 1.00, tracking 1.00, **avoidance
0.20, combined 0.27** — obstacle skills WEAK.

### The plasticity finding (important, thesis-worthy)
**More steps DON'T fix weak avoidance**: 5M/phase gave avoidance 0.17 vs 1M's 0.20.
Contrast the standalone RC_field (no prior tasks, no EWC) → ~0.73 avoidance. So the
~0.2 ceiling is **EWC's stability-plasticity tradeoff**: the Fisher penalty protecting
loco/tracking resists the big weight change weaving needs. Budget is NOT the bottleneck.

## Control smoothness (wobble)
Rover fishtailed (mean|yaw_rate| 0.22 = 28% of max on straight drives). Fixed with
`angvel_penalty_scale` (ω² penalty): loco 0.22→0.046 (5× straighter), loco/field success
held. Adopted 0.15 in scenario_15/16. BUT 0.15 makes obstacle-phase weaving harder →
contributes to the low avoidance; consider lower (0.05) or zero on obstacle phases.

## → NEXT ACTIONS when we return (in priority order)

1. **Confirm the plasticity finding** — run scenario_16 with **naive (no EWC)** to a
   separate dir; if naive learns avoidance markedly better (~0.5+) while forgetting
   loco/tracking, that cleanly demonstrates the CL stability-plasticity tradeoff.
   `python scripts/run_scenario.py scenario_16_full_curriculum --cl-method naive
   --perception slam --train-steps 1000000 --n-envs 6 --seed 0 --results-dir results/_diag`
2. **Multi-seed** scenario_14 (7 methods) and the scenario_16 grid for error bars
   (N=1 everywhere now; gaps are modest).
3. **privileged outlier** — scenario_16 privileged regressed (0.42, forgetting 0.19)
   while reactive/slam were fine; re-seed to confirm fluke vs real.
4. **Tune for avoidance in-curriculum** — lower `ewc_lam`, and lower/zero
   `angvel_penalty_scale` on the obstacle phases (0.15 hurts weaving).

## Artifacts & how to reproduce
- Run a grid: `for p in privileged reactive slam; do python scripts/run_scenario.py
  scenario_16_full_curriculum --cl-method ewc --perception $p --train-steps 1000000
  --n-envs 6 --seed 0; done; python scripts/run_scenario.py scenario_16_full_curriculum --compare`
- View all phases in 3D (auto-detects perception from path): `mjpython
  scripts/visualize_all_phases.py results/scenario_16_full_curriculum/ewc__slam/seed_0`
- Trajectory/collision maps: `scripts/plot_obstacle_maps.py --perception slam --policy <ckpt>`
- Tier eval + bar chart: `scripts/eval_obstacle_policy.py --policy <ckpt>`
- Best standalone obstacle policy: `results/_obstacle_nav/slalom_field_hard_best.zip`
- Deferred/GPU-only: RecurrentPPO (LSTM) — `scripts/train_obstacle_recurrent.py`
  (doesn't learn on Mac CPU; needs GPU/MJX).

## Honest caveats
Everything is **N=1 seed**. Robust headline: EWC forgetting≈0 across the curriculum &
perceptions. Modest/uncertain: retention ordering (slam≥reactive≥privileged) and the
absolute obstacle-skill numbers. Obstacle avoidance IN the curriculum is weak (~0.2) —
plasticity-limited, not budget-limited.
