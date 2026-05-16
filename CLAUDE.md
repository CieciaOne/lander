# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Master's thesis on **continual learning techniques for Mars rover navigation**.
The work is empirical: implement multiple CL methods on top of PPO; train sequentially
across a curriculum of terrains; measure forgetting/retention against a joint-training
baseline. The rover is a rocker-bogie 6-wheel drive in MuJoCo; the policy outputs
2-D Ackermann (throttle, steer).

Source-of-truth research plan: [`docs/plan.md`](docs/plan.md). Design notes:
[`docs/design/`](docs/design/). Roadmap and gaps: [`docs/roadmap.md`](docs/roadmap.md).

## Architecture

```
src/rover_cl/
├── envs/           # MuJoCo rover + terrain catalog (Gymnasium + MJX)
├── cl/             # 7 CL methods (naive/replay/ewc/hybrid/l2/mas/distill)
├── missions/       # Task / Mission / Runner + scenario registry
├── eval/           # success_rate, retention matrix, forgetting
└── viz/            # thesis-style matplotlib plots
```

Four key abstractions; everything else is composition.

**`TerrainSpec`** (`envs/terrains.py`) — declarative arena: obstacles, friction,
start/goal/waypoints, optional heightmap. Adding a terrain = a factory function
that returns one, registered in `TERRAIN_CATALOG`. Randomised terrains
(`RT_*`, `RC_*`) set `randomize_on_reset` to a callable that re-rolls
obstacle positions / waypoints / heightmap per episode.

**`RoverNavEnv`** (`envs/nav.py`) — Gymnasium env that composes the rover MJCF
with a terrain.

- **Action (2-D, Ackermann)**: `action[0]=throttle` drives all 6 wheels; `action[1]=steer`
  routes through 4 corner steering knuckles (FR/FL `−steer`, RR/RL `+steer`).
- **Obs (40-D)**: 6 ego (`rel_fwd`, `rel_right`, `heading_to_target`, body-frame `linvel_x/y`,
  `angvel_z` — all against current target) + 2 lookahead (`rel_fwd_next`, `rel_right_next`
  to next target) + 8 obstacle slots × `(fwd_min, fwd_max, right_min, right_max)`. Each
  obstacle's AABB is **inflated by `ROVER_FOOTPRINT_RADIUS=0.9m`** (Minkowski sum)
  so the policy treats itself as a point against pre-inflated walls. Empty slots
  pad with a far-diagonal sentinel.
- **Reward**: `5.0×Δd_target` (progress) + `+20` per waypoint hit + `+50` on goal +
  speed bonus − `0.01` step cost − collision/hit/tipped penalties − `0.1×‖Δaction‖²`
  jerk − wheels-off penalty. Stuck-no-progress fires `-30` and ends the episode
  if no `≥0.5m` improvement for `200` steps; stuck-in-collision fires `-5` after
  `30` consecutive contact steps. Defaults in `RoverNavEnv.__init__`.
- **Per-reset jitter** (`±0.5m`, `±0.2rad`) so eval seeds produce different rollouts.
- **Info dict** has `pos_xy`, `yaw`, `collision`, `is_success`, `tipped`,
  `stuck_in_collision`, `stuck_no_progress`, `distance_to_goal`, `distance_to_target`,
  `waypoint_index`, `goal_hold`, `n_wheels_off_ground`, plus a `reward_terms`
  decomposition.

**`CLMethod`** (`cl/base.py`) — Protocol over a single shared SB3 PPO. `train_on(env,
steps, task_id)` is called once per task; weights carry over. `post_train(env, task_id)`
collects whatever the method needs after PPO learning (Fisher, replay buffer, teacher
snapshot).

**`Mission` / `Runner`** (`missions/base.py`) — Mission = list of `Task`s + CL method
choice. Runner trains each task in sequence, evaluates on all seen tasks after every
phase, writes `results.json` + checkpoints + per-(phase, eval-task) report PNGs.
Adaptive gate + interim eval are opt-in on `Task` (see `scenario_11`/`scenario_13`).

## CL methods (7)

All live in `src/rover_cl/cl/`. Mechanism summary:

| Method | Mechanism | When applied | Key knob |
|---|---|---|---|
| `naive` | No protection (control) | — | — |
| `replay` | BC rehearsal on stored (obs, action) | before PPO learn | `buffer_size_per_task` |
| `ewc` | Fisher-weighted L2 penalty toward `θ*` | after PPO learn | `lam` |
| `l2` | Uniform-weight L2 (Fisher=1, baseline) | after PPO learn | `lam` |
| `mas` | Memory Aware Synapses (alt importance) | after PPO learn | `lam` |
| `distill` | KL divergence to frozen teacher | before PPO learn | `distill_kl_weight` |
| `hybrid` | EWC + Replay together | both | `lam`, `buffer_size_per_task` |

The hybrid class reuses EwcCL's and ReplayCL's helper methods via Python's
unbound-method assignment (not multiple inheritance, not a mixin abstraction —
just function reuse).

## Scenarios

Active scenarios in [`scenarios.py`](src/rover_cl/missions/scenarios.py):

| Scenario | What it does | Use |
|---|---|---|
| `scenario_01_sequential_terrains` | T1 → T2 | Baseline forgetting test |
| `scenario_02_three_terrains` | T1 → T2 → T3 | Extended forgetting |
| `scenario_03_order_sensitivity` | T1→T2→T3 or reverse | Tests order matters |
| `scenario_04_replay_sweep` | Replay buffer size sweep | Hyperparameter study |
| `scenario_05_full_terrain_curriculum` | T1→T2→T3→T4_dunes | 4-task forgetting |
| `scenario_07/08/09_*` | Single-task / curriculum on blocked arc | Sanity baselines |
| `scenario_10_robust_curriculum` | 13-phase rich curriculum | Earlier broad curriculum (forgetting-prone, kept for comparison) |
| `scenario_11_robust_generalist` | 7-phase mixed-distribution curriculum | Improved curriculum with within-phase anchors |
| **`scenario_12_joint_training`** | Single phase on `RC_full_random` for 5M steps | **Joint-training baseline / deployable >90% candidate** |
| **`scenario_13_integrated_curriculum`** | 4-phase, no-single-skill-isolation | **Current best CL curriculum design** |

Scenarios 02_threat_classes and 06_fusion are stubs (`NotImplementedError`).

### Key curriculum-design lesson

Phase-specific single-skill phases (e.g. "obstacle-only" right after "path-only")
cause severe catastrophic forgetting because the new task's gradient pressure
destroys features the old task depended on. The integrated curriculum
(scenario_13) avoids this by ensuring **every phase after foundation has
obstacles AND waypoints in every episode** — no skill is trained in isolation.
The CL method's job becomes "preserve smoothly improving skills" instead of
"preserve skill A while task switches to skill B".

## Terrains

Three families:

- **`T1`–`T6`**: hand-designed static terrains used by scenarios 01–09.
- **`RT_*`**: randomized terrains (re-roll obstacles/waypoints/heightmap per
  episode) used by `scenario_11`. Single-skill phases: `RT_drive_random`,
  `RT_with_waypoint`, `RT_obstacle_field`, `RT_dunes`, etc.
- **`RC_*`**: mixed-distribution randomized terrains used by `scenario_13`.
  Each one has within-phase anchors so earlier skills survive. Notably
  `RC_navigation` has obstacles AND waypoints in every episode.

Adding a terrain = one factory function in `terrains.py` + one line in
`TERRAIN_CATALOG`. The catalog is the single source of truth for `--terrain`-style
lookups.

## Rover model (`assets/rover.xml`)

A passive Λ-shaped rocker-bogie 6-wheel drive, ~103 kg total, ~1/8 of real
Curiosity scale. Full geometry, masses, friction, and viewer details in
[`docs/design/environment.md`](docs/design/environment.md). Quick facts:

- **Track**: front/rear wheels at body-rel x=±0.85m, middles at ±0.95m
  (slightly outboard for steering clearance). Wheels 0.22m radius, 0.12m
  half-thickness.
- **Suspension**: rocker pivots at chassis upper side, bogies coupled by
  a joint-equality differential (`rocker_right = -rocker_left`). Joints
  are PASSIVE — no actuators on suspension.
- **Wheel collision is a SPHERE**, not the visual cylinder. MJX doesn't
  support cylinder-{box, hfield, mesh} collisions, so each wheel has a
  visible cylinder (`contype=conaffinity=0`) plus an invisible 0.22m
  sphere that does the actual contact. The CPU path uses the same
  sphere — slightly cheaper than cylinder, kinematically identical
  because the axle joint constrains motion to one axis.
- **14 actuators**: 6 wheel drives (vel, ctrl 0-5), 4 corner steers (pos, ctrl 6-9),
  4 arm joints (pos, ctrl 10-13, pinned to stow during navigation training).
- **Solver**: timestep 0.005s, Newton + elliptic friction cone, Mars gravity
  (3.71 m/s²). Don't change casually — see [`docs/design/environment.md`](docs/design/environment.md).

## MJX backend (optional, GPU)

`--backend mjx` runs N rovers in parallel under a single jitted JAX vmap.
Designed for Linux + CUDA (e.g. RTX 3060 Ti) — on Mac CPU JAX it works
but is slower than `--backend cpu` (SubprocVecEnv path). Files:

- `envs/nav_mjx.py` — `MjxNavEnv`, the JAX env.
- `envs/mjx_vec_env.py` — SB3 VecEnv adapter (numpy↔JAX at the boundary).

Two architectural limits of the MJX path:

1. **Static obstacle layout per Env instance.** MJX requires `geom_pos` to
   live in the compiled model; we bake in one `randomize_on_reset` roll at
   init and per-env variation comes from a pre-sampled spawn/target pool.
2. **Cold JIT compile is slow** (~5-15 min per unique model shape). Each
   curriculum phase with a different obstacle count triggers a new compile;
   persistent cache (`JAX_COMPILATION_CACHE_DIR`) helps on the second run.

For Mac iteration, stay on `--backend cpu`. For thesis-scale runs on a
3060 Ti, MJX is worth the setup cost.

## Commands

Activate venv first; all scripts assume it's active:

```bash
source .venv/bin/activate
```

Run a CL scenario:

```bash
# CPU path (default), Mac M3 Air: n_envs 4-6 is the sweet spot
python scripts/run_scenario.py scenario_13_integrated_curriculum \
    --cl-method hybrid --seeds 0,1,2 --train-steps 600000 --n-envs 6

# Joint-training baseline (no CL, single phase, big budget)
python scripts/run_scenario.py scenario_12_joint_training \
    --seeds 0 --train-steps 5000000 --n-envs 6

# MJX path (Linux + CUDA recommended; works on Mac but slower)
python scripts/run_scenario.py scenario_13_integrated_curriculum \
    --cl-method hybrid --seeds 0 --train-steps 600000 \
    --n-envs 256 --backend mjx

# Compare CL methods within a scenario (after multiple --cl-method runs)
python scripts/run_scenario.py scenario_13_integrated_curriculum --compare

# Single-seed quick smoke
python scripts/run_scenario.py scenario_11_robust_generalist \
    --cl-method ewc --seeds 0 --train-steps 100000 --n-envs 4
```

Visualize:

```bash
mjpython scripts/visualize_rover.py        # macOS — viewer needs main thread
python   scripts/visualize_rover.py        # Linux
mjpython scripts/visualize_all_phases.py --scenario scenario_13_integrated_curriculum --seed 0
```

Tests:

```bash
pytest tests/ -q                           # full suite (~45 s)
pytest tests/test_mjx.py -q                # MJX-only (longer due to JIT)
pytest tests/test_new_cl_methods.py -v     # the new CL methods (hybrid/l2/mas/distill)
```

## Training pipeline

- **PPO defaults** (`cl/base.py::DEFAULT_PPO_KWARGS`): `n_steps=2048`, `batch_size=128`,
  `lr=3e-4`, `gamma=0.995`, `ent_coef=0.01`, `policy_kwargs.net_arch=[128, 128]`,
  `device="cpu"`. Don't put a [128, 128] MLP on GPU — kernel-launch overhead
  exceeds the compute saving.
- **Parallel rollouts** (`--n-envs N`): `Runner` builds `SubprocVecEnv` of N
  `Monitor`-wrapped envs. EWC/Replay/Hybrid post-training (Fisher / buffer /
  teacher) uses a single fresh env via `cl.post_train(env, task_id)` —
  `train_on(env, ..., skip_post_train=True)` is the boundary.
- **Adaptive gate + interim eval** (opt-in per Task): chunked training loop
  in `Runner.run()`. Set `Task.min_success_to_advance` to trip-early when
  eval crosses threshold; `Task.interim_eval_every` to log success curves
  mid-phase. Both fields default to off.
- **Outputs**: `results/<scenario>/<method>/seed_<N>/` contains `results.json`,
  per-phase `ckpt_phase_*_after_*.zip` checkpoints, per-(phase, eval-task)
  report PNGs, and `retention_matrix.png` / `retention_curves.png` /
  `skill_survival.png` for the run-level summary.

## Test layout

| File | Coverage |
|---|---|
| `test_env.py` | RoverNavEnv reset/step/termination, every terrain compiles |
| `test_rover_features.py` | Actuator + sensor catalog, arm, steering |
| `test_scenarios.py` | scenario_01 end-to-end with naive and replay |
| `test_scenarios_registry.py` | All registered scenarios construct |
| `test_cl.py` | Original CL methods (naive, replay) |
| `test_ewc.py` | EWC mechanism + save/load |
| `test_new_cl_methods.py` | Hybrid, L2, MAS, Distill + scenarios 12/13 |
| `test_scenario_11.py` | Adaptive gate + interim eval + scenario_11 terrains |
| `test_mjx.py` | MJX env + VecEnv + PPO smoke train |
| `test_heightmap.py` | Heightmap generation + sampling |
| `test_metrics_and_plots.py` | Metrics math + matplotlib outputs |
| `test_multiseed.py` | Multi-seed aggregation |
| `test_config_loader.py` | YAML scenario config |
| `test_policy_viewer.py` | Policy-replay viewer |

Full suite runs in ~45 s on Mac (excluding MJX which adds ~2 min due to JIT compile).

## When writing code

- Prefer modifying existing files (env, mission, scenario registry) over creating
  new abstractions. Adding a terrain or scenario should be a single function added
  to the relevant registry.
- Keep `assets/` clean — `rover.xml` is the only authored MJCF; per-scenario
  world XML is generated at runtime in `compose_scene`.
- Results write to `results/<scenario>/<method>/seed_<N>/`. Gitignored.
- All docs/code in English. Code identifiers (file names, terrain IDs, etc.) too.

## Things that look broken but aren't

- The 5 MJCF `rangefinder` sensors return `−1` on miss. They're not in the policy
  obs — kept for ad-hoc inspection and tested by `tests/test_rover_features.py`.
- Wheel `ctrl > 0` rolls the rover backwards in raw MuJoCo — the wheel axle
  convention is `-Y forward`. `RoverNavEnv.step` negates internally so
  `action[0]=throttle>0` is "forward" at the policy interface.
- The arm at `ctrl=0` extends ~2 m forward — pin to a stow pose during nav
  training (the env does this on every step). Without that, the arm hits
  obstacles before the chassis does and the policy becomes overcautious.
- `_detect_collision` checks the contact GEOM name (`obs_*`), not the body name.
  Obstacles in `compose_scene` are top-level geoms in `<worldbody>` and live
  in body 0; filtering by body would silently miss every obstacle hit.
- Obstacles in the policy obs are inflated by `ROVER_FOOTPRINT_RADIUS=0.9m`
  (Minkowski sum). Two boxes that look 1.4m apart in the world appear with
  overlapping AABBs in the obs — correct, because the rover can't fit
  between them.
- Episodes can terminate before `max_steps` for stuck-in-collision (30 contact
  steps) or stuck-no-progress (200 steps with no improvement). Both fire small
  one-shot penalties. Without these guards PPO converges to a "wedge into
  obstacle and bleed reward" optimum.
- Wheel collision is a SPHERE not a cylinder; see "Rover model" above.
- The autoreset path in `MjxNavEnv` previously ran a 150-step physics settle
  inside the vmap, hitting all envs every step. Fixed by precomputing the
  rest pose at init and overriding only the freejoint XY/yaw on reset.
  Watch for similar vmap+jp.where pitfalls if you add new branches to the
  MJX env.
