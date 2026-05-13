# Scenarios

> Code: `src/rover_cl/missions/scenarios.py`. Registered under
> `SCENARIO_REGISTRY` and resolved via `get_scenario(name, **kwargs)`.

## Picking a scenario

| What you want to know | Use |
|---|---|
| Did I break the project? | `scenario_01_sequential_terrains` with `--cl-method naive` |
| Headline CL retention number for the thesis | `scenario_05_full_terrain_curriculum` |
| Why is my rover stuck / hitting things? | `scenario_07_blocked_arc` (single task, full report) |
| Does suspension matter? | `scenario_08_blocked_arc_hills` |
| Does curriculum order matter? | `scenario_03_order_sensitivity` |
| Memory-vs-retention tradeoff in replay | `scenario_04_replay_sweep` |
| Can the policy learn the arc from scratch? | `scenario_09_curriculum_arc` |
| **Single robust ready-to-deploy model** | `scenario_10_robust_curriculum` |

## CL forgetting / retention

### `scenario_01_sequential_terrains`
- **Tasks**: `T1_flat → T2_corridor`.
- **Goal**: Smallest 2-task probe of forgetting. The fastest A/B between
  `naive` (expected to forget T1 after training T2) and `replay` / `ewc`
  (expected to retain).
- **Default budget**: `train_timesteps=100_000` per task.
- **Output**: 2 × 2 retention matrix, retention curves, comparison bar
  when run with `--compare`.

### `scenario_02_three_terrains`
- **Tasks**: `T1_flat → T2_corridor → T3_obstacle_field`.
- **Goal**: Stretch the sequence by one task. Tests whether the CL
  method's retention holds with three skills to preserve, not just two.

### `scenario_05_full_terrain_curriculum`
- **Tasks**: `T1_flat → T2_corridor → T3_obstacle_field → T4_dunes`.
- **Goal**: Headline curriculum across the catalog including the
  **organic hfield** (`T4_dunes`). Adds suspension dynamics to the
  forgetting test. This is the "main result" scenario for the thesis's
  continual-learning track.

## CL studies

### `scenario_03_order_sensitivity`
- **Tasks**: `T1 → T2 → T3` or `T3 → T2 → T1` via the `direction` kwarg.
- **Goal**: Does curriculum *order* matter? Run both directions with
  the same CL method and compare final retention. If the rocker-bogie
  benefits from easy-then-hard ordering, EWC's `avg_retention` should
  differ noticeably between directions.

### `scenario_04_replay_sweep`
- **Tasks**: `T1 → T2 → T3` with `buffer_size` configurable.
- **Goal**: Memory–retention tradeoff for replay. Sweep
  `buffer_size ∈ {100, 1000, 5000}` across invocations and plot retention
  vs buffer size. Meaningful only for `cl_method='replay'`; with anything
  else `cl_kwargs` is cleared so the comparison stays honest.

## Single-task navigation (diagnostic)

These exist to debug the *navigation policy itself*, not the CL machinery.
CL method choice doesn't affect outcomes for single-task scenarios.

### `scenario_07_blocked_arc`
- **Tasks**: `T1_blocked_arc` only.
- **Goal**: Force the rover to do an **arc-left around a centerline
  blocker**, biased by an intermediate waypoint at `(-2.2, 6)`. Tests
  whether the policy can learn obstacle avoidance + waypoint chasing
  without any CL forgetting in play.
- **Geometry**: 3 obstacles — two off-axis (left + right), one
  dead-center between start and goal. Center blocker is 1.1 × 1.1 m;
  rover can't fit straight through (footprint 1.8 m), so it must arc.
- **Default budget**: `train_timesteps=200_000`.

### `scenario_08_blocked_arc_hills`
- **Tasks**: `T1_blocked_arc_hills`.
- **Goal**: Same arc-left maneuver but the ground is a gentle perlin
  heightmap (≤ 0.15 m bumps). Tests whether the suspension /
  wheel-slip dynamics break a navigation policy that worked on a
  perfectly flat plane. Flat-trained policies should *mostly* transfer.
- **Default budget**: `train_timesteps=300_000`.

## Curriculum

### `scenario_10_robust_curriculum` (the "robust ready-to-deploy" recipe)
- **Tasks**: 8 randomized-terrain phases, monotonically increasing difficulty.
  - 0 `RT_drive_random` — flat plane, no obstacles, random start + goal.
  - 1 `RT_with_waypoint` — + 1 random waypoint.
  - 2 `RT_with_two_waypoints` — + 2 random waypoints.
  - 3 `RT_one_obstacle` — 1 random obstacle on the path.
  - 4 `RT_obstacle_field` — 3-6 random obstacles.
  - 5 `RT_dense_obstacles` — 8-12 random obstacles.
  - 6 `RT_gentle_hills` — random heightmap ≤ 0.2 m, no obstacles.
  - 7 `RT_dunes` — random heightmap ≤ 0.6 m, no obstacles.
  - (8 `RT_mixed` — capstone with everything at once; commented out by default.)
- **Goal**: Build a single robust policy by **domain randomization**: every
  reset re-rolls obstacle positions, start/goal poses, and (where applicable)
  the heightmap shape. The policy never sees the same configuration twice
  within a phase. Curriculum order means earlier phases teach foundational
  skills (drive, steer, follow waypoint) that the harder phases reuse.
- **Recommended `cl_method='ewc'`** so the curriculum *adds* skills without
  overwriting earlier phases.
- **Default budget**: `train_timesteps=200_000` × 8 phases = 1.6 M total.
  With `--n-envs 6` on M3 Air, ~2-3 hours wall-clock.
- **Output**: 8 checkpoints (`ckpt_phase_k_after_RT_*.zip`); the last one is
  the deployable model.

### `scenario_09_curriculum_arc`
- **Tasks**: `T1_flat → T1_blocked_arc → T1_blocked_arc_hills`.
- **Goal**: Bring the policy up to the hardest task incrementally,
  rather than asking PPO to discover driving + arcing + suspension
  dynamics simultaneously. Each phase warm-starts from the previous
  phase's weights. The retention matrix tells you whether learning the
  later phases destroyed the earlier skills.
- Combine with `--cl-method ewc` or `replay` to also measure forgetting
  along the curriculum; with `naive` it's a pure transfer-learning
  baseline (expected to forget the earlier phases).

## Stubs (registered but not implemented)

### `scenario_02_threat_classes`
- Placeholder for the supervised threat-classification track (parallel
  to nav). Raises `NotImplementedError`.
- Blocked on the classifier env + dataset. See `docs/roadmap.md` §4.

### `scenario_06_fusion`
- Multi-task fusion of nav + threat tracks under a shared encoder.
- Blocked on `scenario_02_threat_classes`.

## Adding a new scenario

1. Write a factory function in `missions/scenarios.py` that returns a
   `Mission(name=..., tasks=[...], cl_method=..., seed=...)`. Standard
   kwargs: `cl_method`, `train_timesteps`, `eval_episodes`, `max_steps`,
   `seed`. Use `_make_task(terrain, ...)` for terrains.
2. Register in `SCENARIO_REGISTRY` at the bottom of the file.
3. Add the key to `EXPECTED_REGISTRY_KEYS` in
   `tests/test_scenarios_registry.py` (the registry-shape test).
4. (Optional) Add a YAML in `configs/` for the most-common invocation,
   e.g. `configs/scenario_XX_<method>.yaml`.
5. (Optional) Document it here.
