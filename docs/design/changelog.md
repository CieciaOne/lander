# Design changelog

Chronological summary of major design changes. New entries go at the top.
For details on any change, the per-topic file in this directory has the
"why" and the current state.

## 2026-05 — Mocap obstacles (collision-on-randomized-terrain fix)

- **Critical correctness fix.** Randomized terrains were silently broken:
  `_apply_terrain_roll` mutated `model.geom_pos` at runtime, but MuJoCo's
  *static-geom* BVH (built at compile time) doesn't refresh from that
  field. The broad-phase collision filter looked up obstacles at their
  compile-time `(0, 0, HIDE_Z)` positions and rejected every chassis-vs-obs
  pair. Result: rover phased through every randomized obstacle, never paid
  the `hit_penalty`/`collision_penalty`, and "60% success on
  RT_dense_obstacles" reflected zero real avoidance learning.
- Fix: when `terrain.randomize_on_reset is not None`, `compose_scene` now
  emits each obstacle wrapped in a `<body mocap="true">` parent.
  `_apply_terrain_roll` writes `data.mocap_pos[mocap_id]` instead of
  `model.geom_pos[gid]` — mocap bodies use the dynamic-AABB path which
  *does* track runtime position changes.
- `rollout_with_trajectory`'s obstacle layout snapshot switched from
  `model.geom_pos` to `data.geom_xpos` so it works for both static
  (T1_blocked_arc) and mocap (RT_*) obstacles.
- Verified: teleport-into-obstacle on `RT_dense_obstacles` now generates
  42 chassis-vs-`obs_0` contacts and `_detect_collision = True`. Hidden
  slots (parked at `HIDE_Z = -50`) produce 0 contacts. All 87 tests pass.
- Affects all randomized terrains (`RT_*`) and `scenario_10`. Existing
  randomized-run checkpoints / `results.json` are *meaningless* — they
  trained without any collision signal. Retrain to get real numbers.

## 2026-05 — Curriculum-v2: per-episode color-coding + reward retuning

- `EpisodeTrajectory` extended with `obstacle_layout`, `start_pos`,
  `goal_pos`, `waypoints`. `rollout_with_trajectory` captures all four
  from the live model right after `env.reset(...)` so randomized terrains
  produce reports whose obstacles, paths, starts, goals, and contacts
  actually match what the rover faced.
- `plot_run_report` got per-episode colour-coding: in *randomized mode*
  (any trajectory carries per-episode pose / layout), each episode's
  path + obstacles + start + goal + waypoints + contact crosses all share
  a distinct colour from the new 12-colour `_EPISODE_COLORS` palette,
  with a tiny pill-numbered start marker. Outcome (success/timeout/
  tipped) lives in the endpoint ring colour. Speed colorbar hidden in
  this mode.
- New `max_drawn_trajectories=5` parameter: ranks all trajectories
  (success first, then closest to goal) and draws only the top N. The
  sidebar stats still compute over the full 10 so the success rate is
  honest. Title shows "showing 5 best of 10".
- New script `scripts/regenerate_reports.py` re-evaluates saved
  checkpoints with the current plotting pipeline so existing runs can
  get the new visuals without retraining.
- `RoverNavEnv.waypoint_bonus` default 5 → **20**: previous value was too
  small for the policy to overcome the disorientation of the
  target-snap moment when a waypoint is hit. Phase 1 had 10% success;
  20 should crack it.
- Waypoint lateral jitter: `RT_with_waypoint` 2.0 → 0.8, `RT_with_two_waypoints`
  2.5 → 1.2. Tighter jitter keeps waypoints near the obvious
  start→goal line.
- EWC `lam` default for `scenario_10` bumped 100 → **400** so basic
  drive skills survive the obstacle-heavy mid-curriculum.
- `scenario_10` reordered to 9 phases ending in a new `RT_mixed`
  capstone — combines obstacles + waypoints + heightmap so the final
  checkpoint retains every skill (previously phase 7 = dunes overwrote
  obstacle skills).

## 2026-05 — Domain-randomization curriculum (`scenario_10`)

- New module `envs/randomization.py` with `TerrainRoll` (per-episode
  randomized values) and sampling helpers
  (`sample_start_goal_pair`, `sample_waypoints_between`,
  `sample_obstacles_along_path`).
- `TerrainSpec` gained `randomize_on_reset: Callable | None` — when set,
  the env mutates `model.geom_pos`, `model.geom_size`, optionally
  `model.hfield_data`, plus the spec's `start_pos / goal_pos /
  waypoints`, then calls `mj_forward`. This lets a single compiled model
  serve every episode of a phase with fresh randomized content.
- Nine new randomized terrains (`RT_drive_random`, `RT_with_waypoint`,
  `RT_with_two_waypoints`, `RT_one_obstacle`, `RT_obstacle_field`,
  `RT_dense_obstacles`, `RT_gentle_hills`, `RT_dunes`, `RT_mixed`).
- New mission `scenario_10_robust_curriculum`: 8-phase domain-randomized
  curriculum that walks "drive anywhere" → "obstacle field" → "dunes". The
  final-phase checkpoint is the robust deployable model.
- Bounded randomization design: each phase fixes structure (count, size,
  region ranges) and varies the specifics. Later phases widen the bounds.
- See `scenarios.md` and `environment.md`.

## 2026-05 — Thesis-grade visualization

- Module-level matplotlib style applied at import: serif body, dropped
  top/right spines, soft grid, 200 DPI saves, custom 8-colour palette
  (slate / sienna / sage / wine / ochre / charcoal / lavender / teal).
- Per-plot polish: mako colormap on retention heatmaps, annotated
  end-of-line values on curves, white inter-bar gaps + inside-bar `n=N`
  labels on comparison charts.
- `plot_run_report` rebuilt: `gist_earth` heightmap underlay, obstacle
  drop-shadows, halo'd markers, speed-colored `LineCollection` paths
  with `magma` colormap, info sidebar with big headline % + outcome
  tiles + per-row averaged metrics inside a `FancyBboxPatch`.
- See `evaluation.md`.

## 2026-05 — Smarter failure handling

- Early-termination guards: episode ends after 30 consecutive collision
  steps (`stuck_in_collision`) OR 200 steps without ≥ 0.5 m progress
  toward the current target (`stuck_no_progress`). Each fires a small
  one-shot penalty. Removes the "wedge into obstacle and bleed reward"
  local optimum.
- **Obstacle inflation**: obstacles in the policy obs are now
  AABB-padded by `ROVER_FOOTPRINT_RADIUS = 0.9 m` (Minkowski sum) so
  the policy can treat itself as a point against pre-inflated walls.
- `ent_coef = 0.01` added to PPO defaults — keeps the policy stochastic
  during training so it doesn't collapse to "drive forward" early.
- New scenario `scenario_09_curriculum_arc`: `T1_flat →
  T1_blocked_arc → T1_blocked_arc_hills`, each phase warm-starting from
  the previous phase's weights.
- See `rewards.md`, `observations.md`, `scenarios.md`.

## 2026-05 — Parallel rollouts + eval seed variation

- `Runner(n_envs=N)` builds a `SubprocVecEnv` of N `Monitor`-wrapped
  envs for PPO learning. EWC / Replay's post-training collection runs
  on a fresh single env via the new `cl.post_train(env, task_id)` hook;
  `train_on(env, ..., skip_post_train=True)` is the boundary.
- CLI flag `--n-envs N`. On Mac M3 (8 cores) the sweet spot is 4–6
  workers. CPU + Accelerate BLAS is the correct backend for the
  [128, 128] MLP — PyTorch MPS overhead exceeds savings at this size.
- Eval seed variation: `evaluate_with_trajectories(seed_base=X)` calls
  `env.reset(seed=X+i)` per episode. Combined with the per-reset
  spawn-pose jitter (`±0.5 m, ±0.2 rad`), this makes the 10 eval
  rollouts produce 10 distinct trajectories.
- See `training.md`, `evaluation.md`, `environment.md`.

## 2026-05 — Reward rebalance for "freeze" optima

- `progress_reward_scale`: 1.0 → 3.0 → **5.0**. Per-step gradient toward
  the target now dominates any plausible cost in tight gaps.
- `proximity_penalty_scale`: 0.15 → 0.03 → **0.0**. The penalty was
  creating freeze local minima — once collision detection actually
  worked, hit_penalty + collision_penalty were sufficient deterrents.
- `collision_penalty`: 1.0 → **3.0/step**. `hit_penalty` added: **10.0
  one-shot** on collision entry. A 3-step graze now costs ≥ 19 instead
  of 3.
- Speed bonus on every checkpoint (waypoint + final goal): linear in
  time-remaining, doubles the bonus when reached immediately. Pushes
  the policy to rush instead of dawdle.
- See `rewards.md`.

## 2026-05 — Collision-detection bug fix + arm stow

- `_detect_collision` was filtering contacts by **other body name**,
  rejecting everything in body 0 ("world"). But obstacles in
  `compose_scene` are top-level `<geom>`s in `<worldbody>` and so all
  live in body 0 — every obstacle contact was being silently dropped.
  Fix: filter by geom **name** (`obs_*` prefix) instead.
- Arm at `ctrl=0` extends ~2 m forward → would hit obstacles before
  the chassis. Env now pins the arm to a stow pose `(yaw=0,
  shoulder=+1.5, elbow=+2.5, wrist=0)` on every step. Tool tip sits
  ~24 cm behind chassis front face — entirely inside the rover
  footprint.
- These two fixes together are what made the collision penalty start
  influencing PPO behavior.
- See `rewards.md` (historical-bug section), `environment.md`.

## 2026-05 — Action space swap (skid → Ackermann)

- Action interpretation changed from `(right_wheel_vel, left_wheel_vel)`
  skid-steer to **`(throttle, steer)` Ackermann**:
  - All 6 drive wheels run at the same velocity (`throttle * 3 rad/s`).
  - 4 corner steering knuckles get Ackermann counter-steer (front
    `-steer`, rear `+steer`), matching `drive_ackermann` in the viewer.
- Skid-steer produced more lateral wheel scrub than yaw because of
  the heavy chassis + intentionally low-gain drive actuators. Ackermann
  gives clean arcs that PPO can actually exploit.
- See `environment.md`.

## 2026-05 — Observation rep swap (lidar → AABB)

- 21-ray `mj_ray` lidar fan replaced with **8 obstacle AABBs** in rover
  body frame: `(fwd_min, fwd_max, right_min, right_max)`. Sorted by
  nearest-point distance, padded with a far-diagonal sentinel.
- obs_dim 27 → 38.
- Why: lidar gave forward-only ray distances; obstacle AABBs give
  ground-truth geometry in **all directions** at the natural granularity
  for gap reasoning. The MJCF rangefinder sensors are still defined on
  the rover for ad-hoc inspection but aren't fed into the policy.
- Later got inflated by rover footprint radius (see "Smarter failure
  handling" entry).
- See `observations.md`.

## 2026-05 — Waypoint mechanic

- `TerrainSpec` gains `waypoints: tuple[(x, y), ...]` and
  `waypoint_radius`. The env tracks a current target, advances when
  the rover enters its radius, and only the *final* target awards
  `goal_bonus`.
- Intermediate hits award `waypoint_bonus` (default 5.0).
- Markers render as translucent blue disks; `wpN` labels in the report.
- Used by `T1_blocked_arc` to bias arc-left navigation around a
  centerline obstacle.
- See `environment.md` (TerrainSpec table), `scenarios.md`.

## Earlier (pre-session) — Established baseline

- Rover MJCF authored from scratch with rocker-bogie Λ suspension,
  differential equality constraint, 14 actuators, 12 sensors. See
  CLAUDE.md "Rover model" for details — that block has been stable.
- CL framework with three methods (Naive, Replay, EWC). EWC penalty
  pass uses a fresh SGD optimizer with gradient clipping (Adam's
  per-parameter scaling pushed params the wrong way under large
  penalty gradients). See `training.md`.
- Scenarios 01–05 registered.
- Initial top-down report wired in.
