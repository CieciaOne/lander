# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Master's thesis ("praca magisterska") on **continual learning techniques for Mars rover navigation**. The work is empirical: implement EWC/replay/hybrid CL methods on top of PPO; train sequentially across terrains (T1→T2→T3→…); measure forgetting/retention against fine-tune and joint-training baselines. Source-of-truth roadmap: `docs/plan.md` (Phases 0–7) and the five scenario specs in `stage01/scenarios/`.

Docs were originally Polish and are now in English; if you add new docs, write them in English. Code identifiers (file names, function names, terrain IDs, etc.) are also English.

Three living docs describe the system in detail — read whichever matches your question:

- **`docs/research_overview.md`** — every stat collected, all 12 sensors and the env's 27-D obs (which casts its own 21-ray HazCam-style fan, separate from the MJCF rangefinders), and a step-by-step walk-through of the train→evaluate→plot loop.
- **`docs/roadmap.md`** — planned organic terrain via MuJoCo HField (XML snippet + Perlin/PNG generators), new mission scenarios mapping onto `stage01/scenarios/`, and the honest gap list (EWC, supervised threat track, multi-seed sweeps).
- **`docs/ergonomics_review.md`** — concrete friction-point review of the current CLI + scripts, ranked improvement list (config files, `--seeds N,M,…`, policy-replay viewer), and what's thesis-blocker vs nice-to-have.

## Architecture

Single Python package `src/rover_cl/` organized by concern:

```
src/rover_cl/
├── envs/          RoverNavEnv (Gymnasium) + terrain framework
├── cl/            CL methods: NaiveCL (no protection), ReplayCL (BC rehearsal)
├── missions/      Task / Mission / Runner + scenario registry
├── eval/          metrics: success_rate, retention matrix, forgetting
└── viz/           matplotlib plots (retention heatmap, curves, method comparison)
```

The four key abstractions:

- **`TerrainSpec`** (`envs/terrains.py`) — declarative arena description (obstacles, friction, start/goal/intermediate waypoints). Adding a terrain = writing a function returning a `TerrainSpec` and registering it in `TERRAIN_CATALOG`. `waypoints=((x, y), ...)` adds intermediate stops the rover hits in order before the final goal — used to bias arc directions around blocking obstacles (e.g. `T1_blocked_arc`).
- **`RoverNavEnv`** (`envs/nav.py`) — Gymnasium env that composes the rover MJCF with a terrain.
  - **Action (2-D, Ackermann)**: `action[0]=throttle` drives all 6 wheels at the same velocity; `action[1]=steer` routes through the 4 corner steering knuckles (FR/FL get `-steer`, RR/RL get `+steer`, same convention as `drive_ackermann` in the viewer). Skid-steer was replaced because the heavy chassis + low-gain drives produced more lateral scrub than yaw.
  - **Obs (38-D)**: 6 pose/velocity fields (`rel_fwd`, `rel_right`, `heading_to_target` against the current waypoint or final goal, plus `linvel_x/y` and `angvel_z`) + 8 obstacle slots × `(fwd_min, fwd_max, right_min, right_max)`. Each obstacle's AABB is projected into rover body frame and **inflated by `ROVER_FOOTPRINT_RADIUS = 0.9 m`** (Minkowski-sum trick) so the policy can treat itself as a point against pre-inflated walls. Slots beyond the obstacle count pad with a far-diagonal sentinel.
  - **Reward** = `progress_reward_scale × Δd_target` + `waypoint_bonus` on each intermediate hit + `goal_bonus` on final hit + speed-bonus (linear in time-remaining, doubles checkpoint payoff when reached immediately) − `step_cost` − `collision_penalty/step` while in contact − one-shot `hit_penalty` on contact entry − `tipped_penalty` − `early_terminate_penalty`. Proximity penalty is disabled by default (was creating freeze optima; collision deterrent does the work now). Defaults in `envs/nav.py::RoverNavEnv.__init__`.
  - **Per-reset jitter**: spawn position ±0.5 m and yaw ±0.2 rad, drawn from `self.np_random` so passing different seeds to `env.reset(seed=...)` produces different trajectories. Used by `evaluate_with_trajectories(seed_base=...)` to diversify the 10 eval rollouts.
  - **Early-termination guards**: episode ends after 30 consecutive collision steps OR after 200 steps without `d_target` improving by ≥ 0.5 m. Each fires a small one-shot penalty so PPO learns "this state is bad, explore something else" instead of bleeding `-3/step` indefinitely.
  - **Info dict**: `pos_xy`, `yaw`, `collision`, `is_success`, `tipped`, `stuck_in_collision`, `stuck_no_progress`, `distance_to_goal` (always the *final* goal), `distance_to_target` (current waypoint), `waypoint_index`. Used by `eval/metrics.py::rollout_with_trajectory` to build the top-down reports.
- **`CLMethod`** (`cl/base.py`) — Protocol over a single shared SB3 PPO. `train_on(env, steps, task_id)` is called once per task; weights carry over.
- **`Mission` / `Runner`** (`missions/base.py`) — a Mission is a list of `Task`s + a CL method choice. The Runner trains each task in order, evaluates on all seen tasks after every phase, and writes `results.json` + checkpoints. Adding a scenario = one function in `missions/scenarios.py` returning a `Mission`.

The rover lives in `assets/rover.xml` (authored MJCF — see below). Terrains compose it into a full scene via string templating in `envs/terrains.py::compose_scene`. Registered scenarios are:

- **CL-forgetting / retention**: `scenario_01_sequential_terrains` (T1→T2), `scenario_02_three_terrains` (T1→T2→T3), `scenario_05_full_terrain_curriculum` (T1→T2→T3→T4_dunes).
- **CL studies**: `scenario_03_order_sensitivity` (T1→T2→T3 *or* reverse), `scenario_04_replay_sweep` (sweep `buffer_size`).
- **Single-task navigation**: `scenario_07_blocked_arc` (T1_blocked_arc with a left-biasing waypoint), `scenario_08_blocked_arc_hills` (07 + gentle hfield).
- **Curriculum**: `scenario_09_curriculum_arc` (T1_flat → T1_blocked_arc → T1_blocked_arc_hills, each warm-starting from the previous phase's weights).
- **Stubs**: `scenario_02_threat_classes`, `scenario_06_fusion` raise `NotImplementedError`.

See `docs/design/scenarios.md` for what each one is meant to measure, and `docs/design/README.md` for the full design-notes index.

## Rover model

`assets/rover.xml` is the authoritative rover MJCF: a Mars rover with **passive Λ-shaped rocker-bogie suspension** matching Curiosity / Perseverance topology, plus an actuated 4-DOF arm and a sensor mast.

### Suspension geometry

Chassis sits at world z = 0.75, half-extents (0.55, 0.95, 0.18). Wheels at world z = 0.25 (axle), 17 cm of wheel below chassis-bottom level. Per side:

- **Rocker** pivots at the chassis upper side (body-relative `pos="±0.65 0.10 0.10"`, world pivot at z ≈ 0.85). Two cylindrical arms — the front arm extends outboard and forward dropping 0.35 m over 0.55 m forward (~32° from horizontal), the rear arm extends outboard and back dropping 0.15 m over 0.50 m back (~17°). Joint range **±0.44 rad (±25°)**, damping=2, no spring — passive bearing only.
- **Steering knuckle** at each corner — a small body with a vertical cylindrical strut going from the rocker arm tip down 0.25 m to the wheel hub. Rotates about Z (steering axis); the axis passes through the wheel center so steering has **zero scrub radius**. Joint range **±1.0 rad (≈±57°)**, position-controlled.
- **Bogie** at the rear arm tip (Λ-shaped): pivot at world z ≈ 0.70; arms drop 0.20 m over 0.35 m forward and back (~30°). Joint range **±0.35 rad (±20°)**, damping=8, no spring. Carries the middle wheel (rigid vertical strut on the BOGIE body, NOT the wheel — so the strut doesn't rotate with the wheel) and the rear steering knuckle.
- Middle wheels are fixed-direction (no steering, real-rover correct); only the 4 corner wheels steer.

All suspension geoms are `<cylinder>` (no `<capsule>` end-caps that would show as "balls"), with `contype=0` so rigid skeleton links don't self-collide.

### Differential

The two rockers are coupled by a MuJoCo joint equality constraint (`rocker_right = -rocker_left`, with explicit `solref="0.005 1"` / `solimp="0.99 0.999 0.001"` for tight resolution). Same kinematic relationship as the physical differential bar (Sojourner) / differential gearbox (Curiosity, Perseverance) — when one side rotates up, the other rotates down by the same angle, so the chassis stays at the average pitch. Bogies are independent per side. Tested by `tests/test_env.py::test_rocker_bogie_differential`.

### Masses, inertias, friction

Total rover mass ~103 kg (≈ 1/8 of real Perseverance for sim-tractable timestep).

- **Chassis** (`<inertial>` on `base_link`): mass=70 kg, `diaginertia="22 8 28"` (uniform-box approximation for 1.1×1.9×0.36), `pos="0 0 -0.18"` — CoM offset to chassis bottom level (RTG/battery height on the real rover; lowers the tipping lever arm).
- **Wheels**: mass=1.5 kg each, `diaginertia="0.031 0.024 0.024"` (cylinder along axle X).
- **Arm bodies**: density=300 (≈ 0.4–0.8 kg each, ~2 kg total).
- **Wheel friction**: `(1.8, 0.1, 0.01)` — sliding/spin/rolling. Tuned for grip without excessive lateral scrub during cornering.

### Drive actuator gain (intentional)

Wheel velocity actuators are **low-gain on purpose**: `gainprm=30`, `forcerange=±60 N·m` (down from the typical 60/150). High-gain actuators dump huge torques into slipping wheels, which pumps the bogie up during in-place spin. With kv=30 the actuator can still hold commanded speed on the ground but won't go ballistic when the wheel is sliding. Don't bump these back without re-testing the spin demo.

### Solver settings

`<option timestep="0.005" gravity="0 0 -3.71" integrator="implicitfast" solver="Newton" iterations="150" tolerance="1e-10" cone="elliptic" impratio="3"/>` — Newton + elliptic friction cone gives clean rolling-with-scrub behavior; the small timestep is needed for accurate friction resolution. **Don't change these casually.** Increasing the timestep destabilizes corner turns.

### 14 actuators (`nu=14`)

- `drive_{right|left}_{front|middle|rear}` × 6 — velocity-controlled wheels, **ctrl 0–5**, ctrlrange ±3.0 rad/s, forcerange ±60.
- `steer_{right|left}_{front|rear}` × 4 — position-controlled corner steering, **ctrl 6–9**, ctrlrange ±1.0 rad. Order in ctrl: right_front, right_rear, left_front, left_rear.
- `arm_yaw`, `arm_shoulder`, `arm_elbow`, `arm_wrist` × 4 — position-controlled arm joints, **ctrl 10–13**, ctrlranges `±1.57 / ±1.5 / ±2.5 / ±2.5` rad. Default `ctrl=0` = arm extended straight forward (+Y) from the chassis front face.

### 12 sensors (`nsensor=12`)

- IMU + body pose × 6: `imu_gyro`, `imu_accel`, `base_pos`, `base_quat`, `base_linvel`, `base_angvel` (all on the chassis `imu` site at body origin).
- 5-ray forward lidar fan: `lidar_m60`, `lidar_m30`, `lidar_0`, `lidar_p30`, `lidar_p60` — sweeps ±60° around forward (+Y), sites on the chassis front-top edge. Each returns ray distance in metres, or **−1** when no hit within range. Treat −1 as the "max range" sentinel; do not feed it raw to a policy.
- Tool-tip end-effector position: `tool_pos` (framepos on the arm `tool_tip` site).

### Cameras

- `chase` — third-person follow camera at body-relative `pos="0 -4.5 2.0"` in `targetbody` mode; always looks at `base_link`.
- `navcam` — forward-facing camera on the mast head, looks down the rover's +Y forward axis, fovy=70°.

In the viewer, press **TAB** to cycle: tracking → chase → navcam → free.

### Arm

Mounted at the **chassis front face**, centered (`body pos="0 0.95 -0.05"`), oriented forward. At default `ctrl=0` the arm extends straight forward along +Y. Full range of motion (verified by `tests/test_rover_features.py` and the visualizer):

- yaw ±85°: tool sweeps ±0.8 m laterally
- shoulder UP to +80°: tool reaches z ≈ 1.6 m (overhead)
- shoulder DOWN to -80°: tool reaches z ≈ 0 m (ground-level interaction)
- elbow ±2.5 rad: full fold
- wrist ±2.5 rad: roll

### Forward direction = +Y

The mast and the front wheels both point in +Y. **Negative** `ctrl` on the wheel drive actuators rolls the rover in +Y (forward) — the wheel axle convention rolls -Y under positive ctrl. `RoverNavEnv.step` negates the action internally so user-facing `action[i] > 0` means forward. Both `_build_obs` and `_cast_lidar` now compute `body_forward = (-sin(yaw), cos(yaw))` (the body's +Y direction in world) consistently.

### Viewer (`scripts/visualize_rover.py`)

A 14-phase **feature demo** (not teleop), ~50 s cycle:

1. settle
2. forward (straight, smooth ramp)
3. Ackermann LEFT (4-wheel counter-steer arc)
4. spin in place CCW
5. Ackermann RIGHT
6. stop
7–12. **Arm: each joint individually through its full range** — yaw right, yaw left, shoulder up, shoulder down, elbow fold, wrist roll
13. deploy combined pose (reach down-forward)
14. stow

All `ctrl` channels go through a **slew-rate limiter** so phase transitions are smooth (no step changes). The viewer defaults to **`mjCAMERA_TRACKING`** mounted on `base_link` — camera follows the rover but you can orbit/zoom/elevate with the mouse. `--free-camera` starts with the standard free camera instead. TAB cycles cameras. Lidar readings print on every phase transition.

Drive helpers:
- `drive_straight(forward_vel)` — all wheels straight, all driven equally. `forward_vel > 0` = move +Y.
- `drive_ackermann(forward_vel, steer_angle)` — 4-wheel counter-steer Ackermann. `steer_angle > 0` curves right.
- `drive_spin_in_place(omega)` — corner wheels tangent to circle around rover center; middle wheels commanded at their Y-velocity for circular motion. `omega > 0` = CCW.

### Original asset files

**Do not edit the originals** in `data/my_rover/` or `data/my_rover_bkp/` — those are the user's source assets. `assets/rover_chassis.urdf` is a corrected copy (mislabeled left-side wheel names fixed) but unused at runtime; `assets/rover.xml` is the active MJCF, authored from scratch because the URDF's collision meshes are mis-positioned in `base_link` frame instead of their own link frames. `assets/meshes/` holds renamed copies of the originals; not currently referenced by `rover.xml`. `data/blender_files/perseverance_rover.urdf` (M2020) is **not used**.

## Commands

**Always activate the venv first.** Python actions assume the venv is active:

```bash
source .venv/bin/activate
```

Then:

```bash
# Run all tests (≈6 s)
pytest tests/ -q

# Run one scenario end to end (training + plots; results land in results/<scenario>/<method>/seed_<N>/)
python scripts/run_scenario.py scenario_01_sequential_terrains --cl-method naive   --train-steps 30000 --seed 0
python scripts/run_scenario.py scenario_01_sequential_terrains --cl-method replay --train-steps 30000 --seed 0

# Parallel rollout collection (Mac M3 / M-series): --n-envs 4 spins up 4 MuJoCo
# instances via SubprocVecEnv. Cuts wall-clock for long training runs roughly
# 2–3× once worker spinup amortizes. EWC / replay still do their post-training
# Fisher/buffer collection on a single fresh env after PPO learning finishes.
python scripts/run_scenario.py scenario_07_blocked_arc --cl-method naive --train-steps 500000 --n-envs 4 --seed 0

# MJX (GPU): batch N envs in a single JAX process. On a 3060Ti N=512+ is
# comfortable. --mjx-impl warp enables Nvidia's MuJoCo-Warp pipeline for
# extra collision pairs (requires pip install nvidia-warp on Linux+CUDA).
python scripts/run_scenario.py scenario_07_blocked_arc --cl-method naive \
    --train-steps 500000 --n-envs 512 --backend mjx --seed 0

# After running multiple methods, write a comparison bar chart:
python scripts/run_scenario.py scenario_01_sequential_terrains --compare

# Interactive viewer (macOS requires mjpython; Linux uses plain python):
mjpython scripts/visualize_rover.py        # macOS
python   scripts/visualize_rover.py        # Linux
```

The viewer is **macOS-sensitive**: MuJoCo's passive viewer must own the main event loop, which only `mjpython` provides on macOS. The script guards against this and prints a clear error if launched with `python` on darwin.

## Test layout

Tests cover the actual research scenarios with tiny budgets so the suite stays fast:

- `tests/test_env.py` — `RoverNavEnv` reset/step/termination, every terrain in the catalog compiles, rover drives forward when actuated, rocker-bogie differential constraint.
- `tests/test_rover_features.py` — full actuator + sensor catalog (14 actuators / 12 sensors), lidar with and without an obstacle, arm responds to commanded pose, corner steering tracks commanded angle.
- `tests/test_scenarios.py` — `scenario_01_sequential_terrains` runs end-to-end with both `naive` and `replay`, produces a valid retention matrix and plots.
- `tests/test_cl.py` — CL methods individually (factory, save/load, replay buffer state).
- `tests/test_metrics_and_plots.py` — metrics math + matplotlib outputs.

Whole suite runs in ~10 s. Real research runs go via `scripts/run_scenario.py`, not pytest.

## Training pipeline

- **PPO defaults** (`cl/base.py::DEFAULT_PPO_KWARGS`): `n_steps=2048`, `batch_size=128`, `learning_rate=3e-4`, `gamma=0.995` (effective horizon ≈ 200 steps, so a +50 goal bonus at step 500 backprops as ≈ +4 at step 0), `ent_coef=0.01` (keeps the policy stochastic during training so it doesn't collapse to a "drive forward and freeze" optimum), `policy_kwargs.net_arch=[128, 128]`.
- **Parallel rollouts** (`--n-envs N`): `Runner` builds a `SubprocVecEnv` of N `Monitor`-wrapped envs for PPO learning. EWC/replay's post-training data collection still uses a single fresh env via the new `cl.post_train(env, task_id)` hook — `train_on(env, ..., skip_post_train=True)` is the boundary. M3 Air (8 logical CPUs) sweet spot is `--n-envs 4`–`6`; bigger gives diminishing returns. NN compute on MPS / Neural Engine is **not** a win for the [128, 128] MLP — kernel-launch overhead exceeds savings; CPU + Accelerate BLAS is the right backend.
- **MJX backend** (`--backend mjx`): GPU-accelerated rollouts via MuJoCo XLA. One process, `--n-envs N` rovers run in parallel under a single `jit(vmap(...))` compile. Designed for Linux + CUDA (e.g. 3060 Ti); on Mac CPU JAX it works but is slower than the SubprocVecEnv path. `--mjx-impl warp` flips to the Nvidia MuJoCo-Warp backend on supported hardware (requires `pip install nvidia-warp`). Module: `rover_cl/envs/nav_mjx.py` (jitted env) + `rover_cl/envs/mjx_vec_env.py` (SB3 VecEnv adapter). The MJX path bakes ONE `randomize_on_reset` roll into the model at instance creation (per-env variation comes from a pre-sampled spawn/target pool, not per-reset rerolls), and uses a sphere collision shape inside each wheel body (cylinder is visual-only) because MJX doesn't support cylinder-{box, hfield, mesh}. The CPU path is unaffected.
- **Evaluation** writes a per-(phase, eval-task) **top-down report PNG** alongside `results.json` and checkpoints. Each report shows: terrain (with hfield rendered as `gist_earth` underlay when present), obstacles with drop-shadows, start / waypoints / goal with halos, all 10 eval trajectories speed-coloured via `LineCollection` + `magma` colormap, contact positions as red Xs, plus an info sidebar with success rate %, outcome tiles, and per-row averaged metrics. See `docs/design/evaluation.md`.
- **Thesis plot style** is applied at import time in `viz/plots.py::_apply_thesis_style` — serif body, dropped top/right spines, soft grid, 200 DPI saves, 8-colour curated palette. All plots (`plot_retention_matrix`, `plot_retention_curves`, `plot_method_comparison_with_variance`, `plot_run_report`) share it.

See `docs/design/training.md` for the design rationale and `docs/design/rewards.md` for the reward-term history.

## When writing code

- Prefer modifying existing files (env, mission, scenario registry) over creating new abstractions. Adding a terrain or scenario should be a single function added to the relevant registry.
- Keep `assets/` clean — `rover.xml` is the only authored MJCF; per-scenario world XML is generated at runtime in `compose_scene`.
- Configurations (PPO hyperparams, EWC λ, replay buffer size/strategy) belong in `configs/` as YAML when you start sweeping; for now defaults live in the relevant constructors.
- Results (JSON, PNGs, checkpoints) write to `results/<scenario>/<method>/seed_<N>/`. `results/` is gitignored.
- Don't reintroduce Polish strings in new docs/code. Translation work happened in 2026-05; preserve English.

## Things that look broken but aren't

- `data/my_rover/rover.urdf` has incorrectly named left-side wheels (`*_001` labeled "right") and collision meshes positioned in `base_link` frame instead of their own link frames. We do not load it — `assets/rover.xml` was authored from scratch instead.
- The 5 MJCF `rangefinder` sensors on `base_link` (`lidar_0`, `lidar_m30`, `lidar_p30`, `lidar_m60`, `lidar_p60`) still exist in `assets/rover.xml` and return **−1** when no hit is found. They are independent of the env obs — `RoverNavEnv` no longer feeds any ray-cast lidar into the policy. The sensors are kept for ad-hoc inspection and are exercised by `tests/test_rover_features.py`. If you build a new sensor-fusion track, treat −1 as the max-range sentinel.
- Raw MuJoCo callers (outside `RoverNavEnv`) that want to drive forward must use **negative** wheel `ctrl` because the wheel-axle convention rolls the rover in -Y under positive ctrl. The env negates internally so `action[0]=throttle>0` already drives forward; `scripts/visualize_rover.py` exposes the same user-facing convention.
- The arm at `ctrl=0` extends straight forward ~2 m, so during navigation training the env pins it to a stow pose (`shoulder=+1.5, elbow=+2.5`) on every step — without that, the arm hits obstacles long before the chassis does, and the policy ends up overcautious. The viewer can still drive the arm freely; only the training env holds it.
- `_detect_collision` checks the contact GEOM name (`obs_*`), not the body name. Obstacles in `compose_scene` are top-level geoms in `<worldbody>` and so live in body 0 ("world") along with the ground; filtering by body would silently drop every obstacle hit. (This was a real, latent bug before the Ackermann/obs swap.)
- **Obstacles in the policy obs are inflated by ~0.9 m** (rover footprint radius) via Minkowski sum, so two boxes that look "1.4 m apart" in the world appear with overlapping AABBs in the obs — that's correct, because the rover physically can't fit. This is intentional, not a bug.
- **Episodes can terminate before `max_steps`** for reasons other than success/tipped: 30 consecutive collision steps or 200 steps without ≥ 0.5 m progress toward the current target. Both fire a small one-shot penalty and end the episode. Without these guards, PPO would converge to a "wedge into obstacle and bleed reward" optimum.
- **Spawn pose has per-reset jitter** (±0.5 m, ±0.2 rad) drawn from `self.np_random`. Tests that assert exact spawn coords would break, but they don't — tests assert *progress* (e.g. `d1 < d0`), which still holds.
- **Middle wheels lift slightly (40–60 mm over 3–5 s) during in-place spin.** This is an intrinsic property of the rocker-bogie + steered-corner-wheel design — the bogie articulates in response to the lateral scrub forces from the middle wheels. Real Curiosity / Perseverance face the same constraint and **avoid tight in-place spins** in favor of arc turns. The Ackermann turn phases have effectively zero lift. Don't try to "fix" the spin lift by adding stiffness/springs to the bogie — that would violate the passive-suspension contract.
- Wheel drive actuator gain is intentionally **low** (`gainprm=30`, `forcerange=±60`). High gain makes the spin lift catastrophic. See "Drive actuator gain" subsection above.
