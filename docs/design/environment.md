# Environment

> Code: `src/rover_cl/envs/nav.py`, `src/rover_cl/envs/terrains.py`,
> `assets/rover.xml`.

## Simulator

MuJoCo 3.5.0, ARM64-native on M-series. Solver:

```
<option timestep="0.005" gravity="0 0 -3.71" integrator="implicitfast"
        solver="Newton" iterations="150" tolerance="1e-10"
        cone="elliptic" impratio="3"/>
```

- Mars gravity (`-3.71 m/s²`).
- Newton solver + elliptic friction cone → clean rolling-with-scrub.
- `timestep=0.005` × `control_decimation=5` → 0.025 s per env step.
- **Do not raise the timestep**; it destabilizes corner turns.

## Terrain framework

`envs/terrains.py::TerrainSpec` is a dataclass declarative description.
Adding a terrain = one factory function returning a `TerrainSpec` +
registration in `TERRAIN_CATALOG`. The Env composes the rover MJCF with the
terrain at runtime via `compose_scene` (string templating).

`TerrainSpec` fields:

| Field | Meaning |
|---|---|
| `name` | Used in MJCF `<mujoco model="…"/>` and in result paths. |
| `arena_half_extent` | Half-size of the ground plane (m). |
| `obstacles: list[Obstacle]` | Axis-aligned boxes; each is rendered as a single `<geom>` under `<worldbody>`. |
| `start_pos, start_yaw` | Rover spawn — *before* per-reset jitter. |
| `goal_pos, goal_radius` | Final goal position + acceptance radius (m). |
| `waypoints: tuple[(x, y), ...]` | Intermediate stops the rover must hit (in order) before `goal_pos`. Empty by default. |
| `waypoint_radius` | Acceptance radius for waypoints (default 1.5 m). |
| `heightmap, heightmap_extent` | Optional 2-D heightmap → ground becomes a MuJoCo `hfield`. |

### Heightmaps

When `heightmap is not None`, `compose_scene` swaps the ground `plane` for
an `hfield` and `compile_scene` writes the normalized [0, 1] heightmap into
`model.hfield_data`. The world-frame elevation at a point is
`heightmap[r, c] * heightmap_extent[2]` metres.

Two generators are provided:

- `generate_heightmap_perlin(nrow, ncol, scale, octaves, seed)` — additive
  OpenSimplex noise rescaled to [0, 1]. Used by `T4_dunes` and
  `T1_blocked_arc_hills`.
- `generate_heightmap_slope(nrow, ncol, grade, axis)` — linear ramp.
  Used by `T6_slope`.

The rover's `reset()` accounts for hfield elevation by spawning at
`qpos[2] = 0.95 + heightmap_extent[2]` (so the rover doesn't start
inside the surface).

## Rover model

`assets/rover.xml` is the authoritative MJCF — see CLAUDE.md "Rover model"
section for the full geometry / mass / sensor breakdown. Highlights that
matter for env design:

- **Effective footprint radius ≈ 0.9 m** at the rocker arms / wheels —
  *larger* than the 0.55 m chassis half-width. The policy obs reflects this
  via Minkowski-sum inflation (see `observations.md`).
- **Arm at `ctrl=0` extends ~2 m forward**. The env pins the arm to the
  stow pose `(yaw=0, shoulder=+1.5, elbow=+2.5, wrist=0)` on every step so
  it sits ~24 cm behind the chassis front face — otherwise the arm tip
  collides with obstacles long before the chassis does.
- **14 actuators**: 6 drive wheels (velocity-controlled, ctrl 0–5), 4
  corner steering knuckles (position-controlled, ctrl 6–9), 4 arm joints
  (position-controlled, ctrl 10–13).

## Action space

2-D continuous in `[-1, 1]`, Ackermann-style:

| Index | Meaning | Mapping |
|---|---|---|
| `action[0]` | `throttle` | All 6 wheel actuators set to `-throttle * 3.0 rad/s`. The leading minus is the wheel-axle sign convention (positive ctrl rolls the rover in −Y; user-facing throttle > 0 should mean +Y forward). |
| `action[1]` | `steer` | Front knuckles get `-steer * 1.0 rad`, rear knuckles get `+steer * 1.0 rad`. Same convention as `scripts/visualize_rover.py::drive_ackermann`. `steer > 0` → CW yaw (turn right). |

The arm is always held at the stow pose by the env regardless of action.

### Why not skid-steer?

Old action was `(right_vel, left_vel)`. With the heavy chassis and the
intentionally low-gain drive actuators (`gainprm=30`, `forcerange=±60`), a
differential wheel command produced *more lateral wheel scrub than yaw*.
PPO struggled to find clean turning. Ackermann via the 4 corner knuckles
gives crisp arcs that match `drive_ackermann` in the viewer.

## Per-reset start jitter

`reset()` samples `start_pos ± 0.5 m` and `start_yaw ± 0.2 rad` from
`self.np_random`, which is reseeded by `super().reset(seed=...)`. Two
purposes:

1. Eval episodes get diverse trajectories instead of 10 copies of the same
   deterministic rollout (the policy is deterministic and the env was
   otherwise fully deterministic).
2. Training sees varied initial states → policies are more robust to
   spawn-position drift.

`evaluate_with_trajectories(seed_base=…)` (`eval/metrics.py`) passes
`seed_base + i` per episode.

## Episode termination

`step()` can terminate the episode for any of these reasons:

| Reason | Trigger | Truncated? | One-shot reward |
|---|---|---|---|
| Success | Within `goal_radius` of the final goal for ≥ `GOAL_HOLD_STEPS = 5` consecutive steps | `terminated=True` | `+goal_bonus` + speed bonus |
| Tipped | `upright_cos < 0.5` (≈ 60° from vertical) | `terminated=True` | `-tipped_penalty` |
| Stuck in collision | `_collision_streak ≥ collision_terminate_steps (30)` | `terminated=True` | `-early_terminate_penalty` |
| Stuck no progress | `_steps_since_progress ≥ stuck_window_steps (200)` (target distance hasn't dropped by ≥ `stuck_min_progress = 0.5 m`) | `terminated=True` | `-early_terminate_penalty` |
| Timeout | `step_count ≥ max_steps` | `truncated=True` | — |

See `rewards.md` for *why* the stuck guards exist.

## Info dict (per-step)

`step()` returns an `info` dict that downstream tooling reads:

| Key | Type | Notes |
|---|---|---|
| `pos_xy` | `tuple[float, float]` | World-frame rover position. |
| `yaw` | `float` | World-frame yaw (rad). |
| `distance_to_goal` | `float` | Distance to the **final** goal (not the current waypoint), so metrics stay comparable across single- and multi-waypoint terrains. |
| `distance_to_target` | `float` | Distance to the current target (waypoint or goal). |
| `waypoint_index` | `int` | Current target index. Advances when a waypoint is reached. |
| `is_success` | `bool` | True only when the *final* goal is held for `GOAL_HOLD_STEPS`. |
| `collision` | `bool` | Any contact with `obs_*` geoms this step. |
| `tipped` | `bool` | `upright_cos < TIP_OVER_COS`. |
| `stuck_in_collision` | `bool` | The persistent-collision guard fired this step. |
| `stuck_no_progress` | `bool` | The no-progress guard fired this step. |
| `episode` | `dict` | Only on terminal/truncated steps: `EpisodeOutcome` summary. |
