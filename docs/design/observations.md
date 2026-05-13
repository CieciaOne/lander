# Observations

> Code: `src/rover_cl/envs/nav.py::RoverNavEnv._build_obs`.

## Current obs (38-D)

| Index | Field | Source |
|---|---|---|
| 0 | `rel_fwd` to current target | Rover-frame projection of `target − pos`. Positive = target ahead (along body +Y). |
| 1 | `rel_right` to current target | Same projection. Positive = target to rover's right (body +X). |
| 2 | `heading_to_target` | `atan2(rel_right, rel_fwd)`. Wrapping angle in radians. |
| 3 | `linvel_x` | World-frame linear velocity, x-component. |
| 4 | `linvel_y` | World-frame linear velocity, y-component. |
| 5 | `angvel_z` | World-frame yaw rate. |
| 6 + 4 i | `fwd_min`  of obstacle slot *i* | Rover-frame AABB **inflated by ROVER_FOOTPRINT_RADIUS = 0.9 m**. |
| 6 + 4 i + 1 | `fwd_max` | Same AABB. |
| 6 + 4 i + 2 | `right_min` | Same. |
| 6 + 4 i + 3 | `right_max` | Same. |

`K_OBSTACLES = 8` slots. Slots beyond the obstacle count pad with a
"far-diagonal" sentinel `(SENSE, SENSE+0.1, SENSE, SENSE+0.1)` so the
policy can learn to ignore them. Slots are **sorted by ascending
nearest-point distance** (closest obstacle in slot 0), and obstacles
beyond `OBSTACLE_SENSE_RANGE = 8 m` are dropped entirely.

## Why this representation

### History

1. **First pass: 5-D pose + 5-ray MJCF rangefinder sensors.** Cheap but
   tiny: 5 rays in a ±60° fan, ground-truth only "is something here?", no
   directional info. PPO struggled.
2. **Second pass: 27-D pose + 21-ray lidar fan.** The env cast its own
   `mj_ray` fan over 120° at 6 m max range, mimicking
   Curiosity/Perseverance HazCam stereo. Better, but still forward-only —
   the rover was blind to anything behind it, and at 5 m range each ray
   covered ~52 cm of arc, so a 1.1 m obstacle was 2–3 rays wide.
3. **Current: 38-D pose + 8 inflated AABBs.** Direct geometry: for each
   nearby obstacle the policy gets a rectangle in its own frame, and the
   rectangle has been pre-inflated by rover radius. The policy can
   directly read "obstacle A ends at right_max = −0.4, obstacle B starts
   at right_min = +0.5, gap = 0.9 m — too narrow, plan around."

### Why bounding boxes, not nearest points

Earlier iteration used `(rel_fwd_near, rel_right_near, max_size)` — one
point on the nearest obstacle surface plus a scalar "how big". That
obscures gap reasoning: the policy has to decode "is this a 1.1 m wall or
a 5 m wall?" from `max_size`. The AABB representation makes gaps a simple
subtraction the network can learn quickly.

### Why inflate by rover radius (Minkowski sum)

Standard motion-planning trick: treat the rover as a point against
obstacles padded by rover radius (~0.9 m at the rocker arms / wheels).
Two boxes with a real-world 1.4 m gap appear in the obs with overlapping
AABBs, which is **correct** — the rover physically cannot fit through.
Before inflation, the policy had to *learn* its own footprint from
collision experience, which it could do in principle but slowly. With
inflation it gets the geometry for free.

### What's missing

The obs covers box obstacles but says nothing about heightmap topology.
For `T4_dunes` / `T1_blocked_arc_hills` the policy has zero obstacle
slots filled and must rely on IMU (`linvel`, `angvel`) + the progress
signal. Adding a small heightmap patch around the rover would be the
next obvious extension if hfield terrain becomes a focal task.

## MJCF rangefinder sensors (`lidar_*`)

`assets/rover.xml` still defines 5 `rangefinder` sensors (`lidar_m60`,
`lidar_m30`, `lidar_0`, `lidar_p30`, `lidar_p60`) on the chassis front.
They are **not used** by `RoverNavEnv`. They're kept because:

- `tests/test_rover_features.py` exercises them as a smoke test of the
  MJCF.
- They're available for ad-hoc inspection / a future sensor-fusion track.

When reading them, **−1 means "no hit within range"** — that's MuJoCo's
documented sentinel for `rangefinder`. Convert to your max-range
constant before feeding to anything.
