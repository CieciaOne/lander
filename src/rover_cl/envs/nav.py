"""RoverNavEnv: drive the rover from start to goal on a given terrain.

Action space: 2-D continuous in [-1, 1] — Ackermann-style throttle + steer.
  action[0] = throttle  (forward velocity command, all 6 wheels same speed)
  action[1] = steer     (4-wheel counter-steer angle; >0 = turn right / CW yaw)

Old skid-steer action (right_vel, left_vel) was switched to Ackermann because
the rocker-bogie's heavy chassis + low-gain drives meant a differential wheel
command produced more lateral scrub than yaw. Routing steering through the
4 corner knuckles (matches `drive_ackermann` in the viewer) gives clean arcs
that PPO can actually exploit.

Observation: pose+goal (6) + K_OBSTACLES * (fwd_min, fwd_max, right_min, right_max)
in the rover's body frame. For each of the K nearest box obstacles, the env
reports the obstacle's AABB **as projected into rover-frame coordinates**, so
the policy can directly read "obstacle A spans rel_fwd ∈ [2.4, 3.6] and
rel_right ∈ [-1.0, -0.4]" and compute gaps between objects. Empty slots are
padded with a far-diagonal sentinel that's outside the action-relevant zone.

Reward: progress toward current target + waypoint/goal bonus - step cost
         - proximity penalty - collision/tipped penalties.

Termination: success (within goal_radius for GOAL_HOLD_STEPS), timeout, tipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .terrains import TerrainSpec, compile_scene, compose_scene, get_terrain


# Obstacle observation — for each of the K nearest box obstacles, the policy
# sees the obstacle's axis-aligned bounding box AFTER projecting it into the
# rover body frame: (fwd_min, fwd_max, right_min, right_max). This lets the
# policy compute gaps directly (right_min of one obstacle vs. right_max of the
# adjacent one), which the previous (nearest_point, max_size) representation
# obscured. Obstacles farther than OBSTACLE_SENSE_RANGE from the rover are
# dropped; empty slots are padded with a far-diagonal sentinel that the policy
# learns to ignore.
#
# K_OBSTACLES = 8 covers T1 (3), T2 (2 walls), and the 8 nearest of T3's 14.
# Hfield terrains (T4_dunes, T6_slope) have zero box obstacles, so all 8 slots
# pad → policy reads "open terrain" and relies on IMU / progress signal.
K_OBSTACLES = 8
OBSTACLE_SENSE_RANGE = 8.0
OBSTACLE_FEATURES_PER_SLOT = 4  # (fwd_min, fwd_max, right_min, right_max)

# Rover footprint radius (chassis is 0.55 m half-width, but rocker arms +
# wheels extend further). Used to inflate obstacle AABBs in the policy's
# observation: Minkowski sum of obstacle with rover-radius disk, so the
# policy "sees" the geometry it actually needs to clear when treating the
# rover as a point.
ROVER_FOOTPRINT_RADIUS = 0.9

# Arm navigation-stow pose. At ctrl=0 the arm extends ~2 m straight forward
# along +Y and is the FIRST thing to hit any obstacle. shoulder=+1.5 rad
# (fully up) plus elbow=+2.5 rad (fully folded) tucks the tool tip 24 cm
# BEHIND the chassis front face — entirely inside the rover footprint, so
# collision detection only fires on the actual chassis / wheels.
ARM_STOW_CTRL = (0.0, 1.5, 2.5, 0.0)  # (yaw, shoulder, elbow, wrist)
MAX_WHEEL_VEL = 3.0     # rad/s — wheel angular velocity at throttle=1
MAX_STEER_RAD = 1.0     # rad — corner steer angle at |steer|=1 (matches MJCF ctrlrange)
TIP_OVER_COS = 0.5      # cos(angle from upright) below which we count as tipped
GOAL_HOLD_STEPS = 5


@dataclass
class EpisodeOutcome:
    success: bool
    steps: int
    distance_to_goal: float
    cumulative_reward: float


class RoverNavEnv(gym.Env):
    """Gymnasium env that composes the rover with a terrain."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        terrain: TerrainSpec | str,
        max_steps: int = 500,
        control_decimation: int = 5,
        progress_reward_scale: float = 5.0,
        goal_bonus: float = 50.0,
        # Collision penalties softened from (3.0/step + 10 hit) → (1.5/step
        # + 5 hit). Previous values made "freeze near start" a strictly
        # better local optimum than "move and risk a graze" — see the
        # phase-5/8 reports where every episode terminated at exactly
        # `stuck_window_steps` with the rover wiggling in place. Halved
        # collision deterrents combined with a much larger stuck-no-
        # progress penalty (below) reverse the gradient: any progress at
        # all dominates freezing.
        collision_penalty: float = 1.5,      # per-step while in contact
        hit_penalty: float = 5.0,            # one-shot on collision transition
        step_cost: float = 0.01,
        tipped_penalty: float = 20.0,
        # Proximity penalty disabled by default. The hit_penalty (+collision_penalty
        # sustained) is a strong post-contact deterrent and doesn't create
        # pre-contact local minima the way proximity did. If you want a pre-contact
        # gradient back, set proximity_penalty_scale > 0 and proximity_safety_dist
        # to a value smaller than 1.0 m so it can't compete with progress reward.
        proximity_penalty_scale: float = 0.0,
        proximity_safety_dist: float = 1.0,
        # waypoint_bonus bumped 5 → 20 after analyzing scenario_10 results:
        # waypoint phases had 0% success because the transition reward (rover
        # crosses waypoint → target snaps to next position → confusion) was
        # too small to overcome the disorientation that follows. At 20 the
        # checkpoint moment is unambiguously a big positive event.
        waypoint_bonus: float = 20.0,
        # Speed bonus on every checkpoint (waypoint + final goal): adds an extra
        # `speed_bonus_scale * base_bonus * (1 - steps_used / max_steps)`. At
        # step 0 the rover effectively gets DOUBLE the bonus; the bonus decays
        # linearly to 0 if it takes the full episode. This rewards reaching
        # checkpoints quickly without changing the per-step economics.
        speed_bonus_scale: float = 1.0,
        # Per-reset starting-pose jitter. With deterministic policy + fixed
        # terrain, every eval episode produces the same rollout — adding small
        # jitter both diversifies eval trajectories AND makes the policy more
        # robust during training (sees a range of initial states). Set to 0.0
        # to disable.
        start_jitter_pos: float = 0.5,
        start_jitter_yaw: float = 0.2,
        # Early-termination guards. Kill the "stay stuck in collision indefinitely"
        # and "freeze near start" failure modes by ending episodes when those
        # states are detected, so PPO can't converge to them.
        collision_terminate_steps: int = 30,   # ≈ 0.75 s in continuous contact
        stuck_window_steps: int = 200,         # ≈ 5 s with no measurable progress
        stuck_min_progress: float = 0.5,       # m of d_target reduction required
        # Split the early-terminate penalty by failure mode. Freezing
        # (stuck_no_progress) used to fire the same 5.0 penalty as "wedged
        # in collision", which made freezing a strict win: 200 steps × 0.01
        # step cost + 5 penalty = -7, versus 30 × 1.5 + 5 + 5 = -55 for a
        # genuine collision attempt. New defaults: 30 penalty on freezing
        # (≈ matches the collision-streak cost), 5 on collision-streak
        # termination (already paid via collision_penalty).
        stuck_no_progress_penalty: float = 30.0,
        stuck_in_collision_penalty: float = 5.0,
        # Backwards-compat alias. Older callers (and tests) pass
        # `early_terminate_penalty=X` — split it into the two new
        # parameters when provided.
        early_terminate_penalty: float | None = None,
        # Action smoothness penalty. Per step, scales the squared L2 norm
        # of (action_t - action_{t-1}). Max possible per step is 4 × 2 = 8
        # (both action dims swing -1 → +1). The 0.05 scale caps the per-
        # step penalty at -0.4 in the pathological case; typical run cost
        # is ~10 over an episode. Damps the jagged "boogie woogie" paths
        # seen in scenario_10 reports without overpowering the progress
        # gradient.
        action_jerk_scale: float = 0.05,
        # Wheel-grounded penalty: -wheels_off_scale × n_wheels_off per step.
        # 6 wheels off (rover airborne) costs 0.18/step. Discourages
        # aggressive turns that lift the rocker-bogie clear of the ground.
        # Set to 0.0 to disable (e.g. on dunes where some lift is normal).
        wheels_off_scale: float = 0.03,
        # Adaptive time budget: when a waypoint is reached, extend `max_steps`
        # by `waypoint_time_bonus_per_metre * dist_to_next_target + waypoint_time_bonus_base`.
        # Lets episodes with widely-spaced targets actually finish without
        # tuning a single huge max_steps for every terrain.
        # At top speed (~0.6 m/s, 0.025 s/step) the rover needs ~67 steps/m.
        # `40 steps/m + 200 base` ≈ 1.5× wall-clock budget per segment — slack
        # but not endless.
        waypoint_time_bonus_per_metre: float = 40.0,
        waypoint_time_bonus_base: int = 200,
        seed: int = 0,
        render_mode: str | None = None,
    ):
        if isinstance(terrain, str):
            terrain = get_terrain(terrain, seed=seed)
        self.terrain = terrain
        self.max_steps = max_steps
        self.control_decimation = control_decimation
        self.progress_reward_scale = progress_reward_scale
        self.goal_bonus = goal_bonus
        self.collision_penalty = collision_penalty
        self.hit_penalty = hit_penalty
        self.step_cost = step_cost
        self.tipped_penalty = tipped_penalty
        self.proximity_penalty_scale = proximity_penalty_scale
        self.proximity_safety_dist = proximity_safety_dist
        self.waypoint_bonus = waypoint_bonus
        self.speed_bonus_scale = speed_bonus_scale
        self.start_jitter_pos = start_jitter_pos
        self.start_jitter_yaw = start_jitter_yaw
        self.collision_terminate_steps = collision_terminate_steps
        self.stuck_window_steps = stuck_window_steps
        self.stuck_min_progress = stuck_min_progress
        if early_terminate_penalty is not None:
            stuck_no_progress_penalty = early_terminate_penalty
            stuck_in_collision_penalty = early_terminate_penalty
        self.stuck_no_progress_penalty = stuck_no_progress_penalty
        self.stuck_in_collision_penalty = stuck_in_collision_penalty
        self.action_jerk_scale = action_jerk_scale
        self.wheels_off_scale = wheels_off_scale
        self.waypoint_time_bonus_per_metre = waypoint_time_bonus_per_metre
        self.waypoint_time_bonus_base = waypoint_time_bonus_base
        self.render_mode = render_mode
        self._np_random_seed = seed

        self._model, self._data = compile_scene(terrain)

        self._wheel_actuator_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in [
                "drive_right_front", "drive_right_middle", "drive_right_rear",
                "drive_left_front",  "drive_left_middle",  "drive_left_rear",
            ]
        ]
        self._base_pos_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "base_pos")
        self._base_quat_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "base_quat")
        self._base_linvel_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "base_linvel")
        self._base_angvel_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "base_angvel")
        self._base_link_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

        # Cache the 4 corner-steering actuator IDs (env applies a single Ackermann
        # steer command across them: FR & FL get -steer, RR & RL get +steer —
        # same convention as scripts/visualize_rover.py::drive_ackermann).
        self._steer_actuator_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in ["steer_right_front", "steer_right_rear",
                      "steer_left_front",  "steer_left_rear"]
        ]

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # obs layout:
        #   [0..6)  pose+velocity vs current target (6 dims)
        #   [6..8)  NEXT target rel_fwd, rel_right (lookahead — sentinel (0, 0)
        #           if there's no next target, i.e. current target IS the goal).
        #           Lets the policy plan its trajectory across waypoint
        #           boundaries instead of swinging wide at every transition.
        #   [8..)   K obstacles × (fwd_min, fwd_max, right_min, right_max)
        obs_dim = 6 + 2 + K_OBSTACLES * OBSTACLE_FEATURES_PER_SLOT
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._renderer: mujoco.Renderer | None = None
        self._step_count = 0
        self._prev_dist = 0.0
        self._goal_hold = 0
        self._cumulative_reward = 0.0

    # ------------------------------------------------------------------ helpers

    def _sensor(self, sensor_id: int, dim: int) -> np.ndarray:
        adr = self._model.sensor_adr[sensor_id]
        return self._data.sensordata[adr: adr + dim].copy()

    def _base_pose_xy(self) -> tuple[np.ndarray, float]:
        pos = self._sensor(self._base_pos_id, 3)
        quat = self._sensor(self._base_quat_id, 4)  # (w, x, y, z)
        # yaw from quaternion (Z-axis component)
        w, x, y, z = quat
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = float(np.arctan2(siny_cosp, cosy_cosp))
        return pos[:2].astype(np.float32), yaw

    def _upright_cos(self) -> float:
        quat = self._sensor(self._base_quat_id, 4)
        w, x, y, z = quat
        # body +Z in world frame: third column of rot matrix
        z_world_z = 1 - 2 * (x * x + y * y)
        return float(z_world_z)

    def _build_obstacle_features(self, pos_xy: np.ndarray, yaw: float
                                 ) -> tuple[np.ndarray, float]:
        """Return (K×4 features flat array, min_distance_to_any_obstacle_surface).

        Each box obstacle is projected into the rover body frame by rotating
        its 4 world-axis-aligned corners by -yaw, then taking the AABB of those
        rotated corners. The result `(fwd_min, fwd_max, right_min, right_max)`
        lets the policy see the obstacle as a rectangle in its own view — gaps
        between adjacent obstacles are then a simple subtraction the network
        can learn quickly.

        Sorted by ascending nearest-point distance (closest first). Slots past
        the obstacle count pad with a far-diagonal zero-thickness sentinel.
        """
        c, s = np.cos(yaw), np.sin(yaw)
        rover_x, rover_y = float(pos_xy[0]), float(pos_xy[1])
        items: list[tuple[float, float, float, float, float]] = []
        for ob in self.terrain.obstacles:
            cx, cy = ob.pos[0], ob.pos[1]
            sx, sy = ob.size[0], ob.size[1]
            # Distance from rover to nearest point on the world-AABB.
            nx = float(np.clip(rover_x, cx - sx, cx + sx))
            ny = float(np.clip(rover_y, cy - sy, cy + sy))
            d = float(np.hypot(nx - rover_x, ny - rover_y))
            if d > OBSTACLE_SENSE_RANGE:
                continue
            # Project the 4 world-corners into rover frame, then take AABB.
            # Finally inflate by ROVER_FOOTPRINT_RADIUS so the policy treats
            # the rover as a point against an obstacle padded by rover radius
            # — Minkowski sum trick from standard motion planning. A gap that
            # registers as positive between two inflated obstacles is one the
            # rover can actually pass through; a non-positive gap means "no
            # path here, plan around."
            fwds: list[float] = []
            rights: list[float] = []
            for wx in (cx - sx, cx + sx):
                for wy in (cy - sy, cy + sy):
                    dx, dy = wx - rover_x, wy - rover_y
                    fwds.append(-s * dx + c * dy)
                    rights.append(c * dx + s * dy)
            fwd_min = min(fwds) - ROVER_FOOTPRINT_RADIUS
            fwd_max = max(fwds) + ROVER_FOOTPRINT_RADIUS
            right_min = min(rights) - ROVER_FOOTPRINT_RADIUS
            right_max = max(rights) + ROVER_FOOTPRINT_RADIUS
            items.append((d, fwd_min, fwd_max, right_min, right_max))

        items.sort(key=lambda f: f[0])
        out = np.empty((K_OBSTACLES, OBSTACLE_FEATURES_PER_SLOT), dtype=np.float32)
        # Sentinel: a thin-line, far-diagonal "obstacle" the policy can ignore.
        out[:] = (
            OBSTACLE_SENSE_RANGE, OBSTACLE_SENSE_RANGE + 0.1,
            OBSTACLE_SENSE_RANGE, OBSTACLE_SENSE_RANGE + 0.1,
        )
        for i in range(min(len(items), K_OBSTACLES)):
            _, fmin, fmax, rmin, rmax = items[i]
            out[i] = (fmin, fmax, rmin, rmax)
        min_dist = items[0][0] if items else float("inf")
        return out.ravel(), min_dist

    def _current_target(self) -> np.ndarray:
        return np.array(self._targets[self._wp_idx], dtype=np.float32)

    def _build_obs(self) -> np.ndarray:
        pos_xy, yaw = self._base_pose_xy()
        target = self._current_target()
        delta = target - pos_xy
        # Rotate world delta into the rover's body frame. Rover forward = body's
        # +Y direction (front wheels + mast); rover right = body's +X. In world
        # coords, these axes are:
        #     body_forward (body +Y) = (-sin(yaw),  cos(yaw))
        #     body_right   (body +X) = ( cos(yaw),  sin(yaw))
        # Project delta onto each → signed coords in body frame.
        c, s = np.cos(yaw), np.sin(yaw)
        rel_fwd   = -s * delta[0] + c * delta[1]   # positive = target AHEAD
        rel_right =  c * delta[0] + s * delta[1]   # positive = target to rover's RIGHT
        heading_to_goal = float(np.arctan2(rel_right, rel_fwd))

        linvel = self._sensor(self._base_linvel_id, 3)[:2]
        angvel_z = float(self._sensor(self._base_angvel_id, 3)[2])

        # Lookahead: where's the NEXT target in body frame? When the rover is
        # heading toward waypoint W_k, knowing where W_{k+1} sits lets it
        # round corners instead of swerving to W_k then yanking the wheel
        # toward W_{k+1}. If there is no next target (current target IS the
        # final goal), use a sentinel (0, 0) so the policy reads "no further
        # plan beyond this".
        if self._wp_idx + 1 < len(self._targets):
            nxt = np.array(self._targets[self._wp_idx + 1], dtype=np.float32)
            d_nxt = nxt - pos_xy
            rel_fwd_next   = -s * d_nxt[0] + c * d_nxt[1]
            rel_right_next =  c * d_nxt[0] + s * d_nxt[1]
        else:
            rel_fwd_next = 0.0
            rel_right_next = 0.0

        obstacle_feats, min_obstacle_dist = self._build_obstacle_features(pos_xy, yaw)
        # Stash min distance so step() can reuse it for the proximity penalty
        # without rebuilding the feature list.
        self._min_obstacle_dist = min_obstacle_dist

        obs = np.empty(6 + 2 + K_OBSTACLES * OBSTACLE_FEATURES_PER_SLOT, dtype=np.float32)
        obs[0] = rel_fwd
        obs[1] = rel_right
        obs[2] = heading_to_goal
        obs[3] = float(linvel[0])
        obs[4] = float(linvel[1])
        obs[5] = angvel_z
        obs[6] = float(rel_fwd_next)
        obs[7] = float(rel_right_next)
        obs[8:] = obstacle_feats
        return obs

    _WHEEL_GEOM_NAMES = (
        "wheel_rf", "wheel_rm", "wheel_rr",
        "wheel_lf", "wheel_lm", "wheel_lr",
    )

    def _count_wheels_off_ground(self) -> int:
        """Return how many of the 6 wheels are NOT in contact with any non-rover geom.

        Iterates `data.contact[:ncon]`, building the set of wheel geom ids
        that touch *anything* other than another rover-tree geom this step.
        A wheel touching the ground or an obstacle counts as grounded; only
        wheels with zero ground/world contacts this step are "off".
        """
        if not hasattr(self, "_wheel_geom_ids"):
            self._wheel_geom_ids = []
            for n in self._WHEEL_GEOM_NAMES:
                gid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, n)
                if gid >= 0:
                    self._wheel_geom_ids.append(gid)
        if not self._wheel_geom_ids:
            return 0
        grounded = set()
        wheel_set = set(self._wheel_geom_ids)
        for i in range(self._data.ncon):
            c = self._data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in wheel_set and g2 not in wheel_set:
                grounded.add(g1)
            elif g2 in wheel_set and g1 not in wheel_set:
                grounded.add(g2)
        return len(self._wheel_geom_ids) - len(grounded)

    def _detect_collision(self) -> bool:
        """True iff any rover-tree body is in contact with a geom named `obs_*`.

        Obstacles in `compose_scene` are top-level geoms in `<worldbody>`, so
        they all live in body 0 ("world"). Filtering on body name would drop
        every obstacle contact along with the ground — the discriminator is the
        GEOM name, which is unique per obstacle (`obs_0`, `obs_1`, ...).
        """
        rover_body_id_min = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if rover_body_id_min < 0:
            return False
        rover_bodies = set()
        for b in range(self._model.nbody):
            cur = b
            while cur != 0:
                if cur == rover_body_id_min:
                    rover_bodies.add(b)
                    break
                cur = self._model.body_parentid[cur]
        for i in range(self._data.ncon):
            c = self._data.contact[i]
            b1 = self._model.geom_bodyid[c.geom1]
            b2 = self._model.geom_bodyid[c.geom2]
            in1, in2 = b1 in rover_bodies, b2 in rover_bodies
            if in1 != in2:
                other_geom = c.geom2 if in1 else c.geom1
                gname = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, other_geom) or ""
                if gname.startswith("obs_"):
                    return True
        return False

    # -------------------------------------------------------------------- API

    def _apply_terrain_roll(self, roll) -> None:
        """Apply a per-episode randomization roll to the live MuJoCo model.

        Obstacle geom positions/sizes are written directly to the compiled
        model; hidden slots use a far-below-floor z so they have no contact.
        If the roll includes a fresh heightmap, it's written to
        model.hfield_data. Also mutates the TerrainSpec's start/goal/waypoints
        so the rest of the env (obs, reward, info) reads the rolled values.
        """
        # Update spec fields read by obs/reward/info.
        self.terrain.start_pos = tuple(roll.start_pos)         # type: ignore[assignment]
        self.terrain.start_yaw = float(roll.start_yaw)
        self.terrain.goal_pos = tuple(roll.goal_pos)           # type: ignore[assignment]
        self.terrain.waypoints = tuple(roll.waypoints)         # type: ignore[assignment]

        # Update obstacle geom positions + sizes. Geoms are named `obs_<i>`
        # in compose_scene; resolve their ids once and write into the
        # model's mutable numpy arrays.
        n_slots = len(self.terrain.obstacles)
        if len(roll.obstacle_positions) != n_slots:
            raise ValueError(
                f"TerrainRoll has {len(roll.obstacle_positions)} obstacles but "
                f"terrain was compiled with {n_slots} slots"
            )
        for i in range(n_slots):
            gid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, f"obs_{i}")
            if gid < 0:
                continue
            # Size is fine to mutate via model.geom_size at runtime — only the
            # static-geom BVH cache is broken by that path, and mocap bodies
            # (used for randomized obstacles) avoid the static-BVH path
            # entirely. For non-randomized terrains the geom is static and
            # this writes through normally.
            self._model.geom_size[gid] = np.asarray(roll.obstacle_sizes[i], dtype=np.float64)

            # Find the parent body. If it's a mocap body (set up by
            # compose_scene when the terrain is randomized), update its
            # data.mocap_pos so the dynamic-AABB broad phase tracks the new
            # position correctly. Otherwise (static geom in worldbody) fall
            # back to mutating model.geom_pos and pray the BVH is OK — for
            # non-randomized terrains the position never moves so it's fine.
            bid = self._model.geom_bodyid[gid]
            mocap_id = self._model.body_mocapid[bid]
            if mocap_id >= 0:
                self._data.mocap_pos[mocap_id] = np.asarray(
                    roll.obstacle_positions[i], dtype=np.float64,
                )
            else:
                self._model.geom_pos[gid] = np.asarray(
                    roll.obstacle_positions[i], dtype=np.float64,
                )

        # Fresh heightmap (optional).
        if roll.heightmap is not None and self.terrain.heightmap is not None:
            flat = np.asarray(roll.heightmap, dtype=np.float32).ravel(order="C")
            if flat.shape != self._model.hfield_data.shape:
                raise ValueError(
                    f"roll heightmap shape mismatch: model expects "
                    f"{self._model.hfield_data.shape}, got {flat.shape}"
                )
            self._model.hfield_data[:] = flat
            self.terrain.heightmap = np.asarray(roll.heightmap, dtype=np.float32)

        # Re-propagate the model so the next mj_step uses the new layout.
        mujoco.mj_forward(self._model, self._data)

    def reset(self, *, seed: int | None = None, options: dict | None = None
              ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)

        # Domain randomization: if the terrain provides a `randomize_on_reset`
        # callable, sample a fresh TerrainRoll and apply it to the live MuJoCo
        # model (obstacle positions/sizes, heightmap data) and to the spec's
        # start_pos / goal_pos / waypoints so obs + reward + visualization all
        # see the new layout. Hidden obstacle slots get z=HIDE_Z (below floor).
        if self.terrain.randomize_on_reset is not None:
            roll = self.terrain.randomize_on_reset(self.np_random)
            self._apply_terrain_roll(roll)

        sx, sy = self.terrain.start_pos
        # Per-reset jitter so deterministic policy + deterministic env produce
        # varied eval rollouts. Sampled from self.np_random (Gym superclass),
        # which super().reset(seed=...) reseeds at the top of this method.
        if self.start_jitter_pos > 0:
            sx = sx + float(self.np_random.uniform(-self.start_jitter_pos, self.start_jitter_pos))
            sy = sy + float(self.np_random.uniform(-self.start_jitter_pos, self.start_jitter_pos))
        self._data.qpos[0] = sx
        self._data.qpos[1] = sy
        # Spawn slightly above the LOCAL ground height at the rover's spawn XY,
        # not above the global max-elevation. The old "0.95 + hfield_max" rule
        # was conservative — it ensured the wheels never started buried — but
        # on dunes it dropped the rover from up to 1.55 m above ground, which
        # exceeded the 150-step (0.75 s) settle budget and left the suspension
        # still bouncing when the episode began.
        #
        # New approach: query the heightmap at (sx, sy), spawn `SPAWN_CLEARANCE`
        # metres above that point. Same fall distance everywhere, regardless of
        # dune geometry, so the settle phase has a consistent budget.
        from rover_cl.envs.randomization import heightmap_height_at_xy
        terrain_z_here = 0.0
        if self.terrain.heightmap is not None:
            terrain_z_here = heightmap_height_at_xy(
                self.terrain.heightmap,
                self.terrain.heightmap_extent,
                sx, sy,
            )
        # SPAWN_CLEARANCE: base_link sits ≈ 0.75 m above ground at rest (see
        # `settled chassis z` note in CLAUDE.md). Add 0.20 m so the wheels
        # start ~20 cm above the ground, which is just enough drop to engage
        # the rocker-bogie without violent impact.
        SPAWN_CLEARANCE = 0.20
        self._data.qpos[2] = terrain_z_here + 0.75 + SPAWN_CLEARANCE
        # yaw quaternion (w, x, y, z) around Z, with optional jitter.
        yaw = self.terrain.start_yaw
        if self.start_jitter_yaw > 0:
            yaw = yaw + float(self.np_random.uniform(-self.start_jitter_yaw, self.start_jitter_yaw))
        self._data.qpos[3] = float(np.cos(yaw / 2))
        self._data.qpos[4] = 0.0
        self._data.qpos[5] = 0.0
        self._data.qpos[6] = float(np.sin(yaw / 2))
        # zero velocities so the freejoint doesn't carry over momentum from a
        # previous episode's truncation
        self._data.qvel[:] = 0.0
        # Pin the arm to its navigation-stow pose during settle so it folds in
        # rather than swinging through the air pointing forward.
        for i, v in enumerate(ARM_STOW_CTRL):
            self._data.ctrl[10 + i] = v
        # 150 settle steps × 0.005 s timestep = 0.75 s of sim time — enough
        # for the rover to fall the ~0.20 m to equilibrium AND for the rocker /
        # bogie joints to damp out under the new 0.95 m spawn. Earlier 20 / 80
        # was not enough; the rover entered the episode mid-bounce with random
        # vertical velocity, contaminating the early progress reward signal.
        for _ in range(150):
            mujoco.mj_step(self._model, self._data)

        # Build target sequence: intermediate waypoints (in order), then final goal.
        self._targets: list[tuple[float, float]] = [
            *self.terrain.waypoints,
            self.terrain.goal_pos,
        ]
        self._target_radii: list[float] = [
            self.terrain.waypoint_radius
        ] * len(self.terrain.waypoints) + [self.terrain.goal_radius]
        self._wp_idx = 0
        # Episode time budget — starts at the base `max_steps` and grows each
        # time a waypoint is reached (`_extend_step_budget`).
        self._effective_max_steps = self.max_steps

        pos_xy, _ = self._base_pose_xy()
        target = self._current_target()
        self._prev_dist = float(np.linalg.norm(target - pos_xy))
        self._step_count = 0
        self._goal_hold = 0
        self._cumulative_reward = 0.0
        self._was_colliding = False
        self._collision_streak = 0
        # `best_d_target` tracks the smallest distance to current target ever
        # seen; the `stuck_window_steps` countdown resets whenever it drops by
        # at least `stuck_min_progress` metres. Otherwise the rover is "stuck"
        # and we end the episode early.
        self._best_d_target = self._prev_dist
        self._steps_since_progress = 0
        # Action smoothness: previous step's action, used by the jerk penalty.
        # First step has no prior action; init to zero (no penalty on step 1).
        self._prev_action = np.zeros(2, dtype=np.float64)
        obs = self._build_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        # Action-jerk penalty: squared L2 of the action change since last
        # step. Computed up-front so we can stash `_prev_action` cleanly
        # at the end.
        action_jerk_sq = float(np.sum((action - self._prev_action) ** 2))
        action_jerk_penalty = self.action_jerk_scale * action_jerk_sq
        throttle = float(action[0])
        steer    = float(action[1])
        # All 6 drive wheels at the same speed. positive ctrl rolls -Y so we
        # negate to make "throttle > 0 => +Y forward" the user-facing convention.
        wheel_cmd = -throttle * MAX_WHEEL_VEL
        self._data.ctrl[0:6] = wheel_cmd
        # 4-wheel Ackermann counter-steer (matches drive_ackermann in the viewer):
        # steer > 0 turns CW (right). Fronts go -steer, rears go +steer.
        steer_cmd = steer * MAX_STEER_RAD
        # Actuator order: 6=right_front, 7=right_rear, 8=left_front, 9=left_rear.
        self._data.ctrl[6] = -steer_cmd  # right front
        self._data.ctrl[7] = +steer_cmd  # right rear
        self._data.ctrl[8] = -steer_cmd  # left front
        self._data.ctrl[9] = +steer_cmd  # left rear
        # Hold the arm stowed every step so position actuators don't drift.
        for i, v in enumerate(ARM_STOW_CTRL):
            self._data.ctrl[10 + i] = v

        for _ in range(self.control_decimation):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        pos_xy, _ = self._base_pose_xy()
        target = self._current_target()
        dist = float(np.linalg.norm(target - pos_xy))

        progress = self._prev_dist - dist
        self._prev_dist = dist

        current_radius = self._target_radii[self._wp_idx]
        is_final_target = self._wp_idx == len(self._targets) - 1

        waypoint_reached_bonus = 0.0
        in_goal = False
        if dist < current_radius:
            if is_final_target:
                self._goal_hold += 1
                in_goal = True
            else:
                # Advance to next waypoint; recompute _prev_dist against new
                # target so progress shaping doesn't snap to a large step.
                self._wp_idx += 1
                next_target = self._current_target()
                dist_to_next = float(np.linalg.norm(next_target - pos_xy))
                self._prev_dist = dist_to_next
                waypoint_reached_bonus = self.waypoint_bonus
                self._goal_hold = 0
                # Adaptive time budget: extend max_steps by enough to give the
                # rover ~1.5× a top-speed traversal to the next target.
                bonus_steps = int(
                    self.waypoint_time_bonus_per_metre * dist_to_next
                    + self.waypoint_time_bonus_base
                )
                self._effective_max_steps += bonus_steps
                # Critical: reset stuck-detection state. _best_d_target was
                # tracking the OLD target's minimum distance (a small number
                # because the rover just reached it). Without this reset, the
                # NEW target's distance is suddenly much larger, the stuck
                # guard never sees "progress" against the stale best, and the
                # episode dies ~200 steps after every waypoint hit. Reset both
                # the reference distance and the no-progress counter.
                self._best_d_target = dist_to_next
                self._steps_since_progress = 0
        else:
            self._goal_hold = 0

        # Wheels-off-ground penalty: counts wheels with zero contact this
        # step. Stash the count for info-dict reporting too.
        n_wheels_off = self._count_wheels_off_ground()
        wheels_off_penalty = self.wheels_off_scale * float(n_wheels_off)

        collision = self._detect_collision()
        # One-shot hit penalty fires on the step the rover *enters* contact, so
        # even brief touches are costly (a 3-step graze without this would only
        # have lost ~3*collision_penalty, far less than the +50 goal bonus).
        new_hit = collision and not self._was_colliding
        self._was_colliding = collision
        if collision:
            self._collision_streak += 1
        else:
            self._collision_streak = 0

        # Stuck-detection bookkeeping: reset the no-progress counter every time
        # we beat the best-ever distance by at least stuck_min_progress metres.
        # Skip on steps where a waypoint was just hit — the waypoint-advance
        # branch above already reset `_best_d_target` / `_steps_since_progress`
        # to track the NEW target. Without this skip, the stale `dist` (≈ 0,
        # measured against the JUST-REACHED old target) clobbers the reset and
        # makes the new target's "best" 0 — guaranteeing a stuck-trigger 200
        # steps later.
        if waypoint_reached_bonus == 0.0:
            if dist < self._best_d_target - self.stuck_min_progress:
                self._best_d_target = dist
                self._steps_since_progress = 0
            else:
                self._steps_since_progress += 1

        upright = self._upright_cos()
        tipped = upright < TIP_OVER_COS

        # Early-termination guards. Both end the episode AND apply a small
        # one-shot penalty, so the policy gets a clear "this state is bad,
        # explore something else" signal instead of bleeding -3/step forever.
        stuck_in_collision = (self.collision_terminate_steps > 0
                              and self._collision_streak >= self.collision_terminate_steps)
        stuck_no_progress = (self.stuck_window_steps > 0
                             and self._steps_since_progress >= self.stuck_window_steps)
        early_terminate = stuck_in_collision or stuck_no_progress

        success = self._goal_hold >= GOAL_HOLD_STEPS
        terminated = success or tipped or early_terminate
        truncated = self._step_count >= self._effective_max_steps

        obs = self._build_obs()

        # Proximity penalty: encourages steering away from obstacles before
        # contact. Uses _min_obstacle_dist (in metres, stashed by _build_obs)
        # so the penalty has a physically meaningful safety threshold and
        # fades linearly to 0 at proximity_safety_dist.
        proximity_penalty = 0.0
        if (self.proximity_penalty_scale > 0
                and self.proximity_safety_dist > 0
                and self._min_obstacle_dist < self.proximity_safety_dist):
            proximity_penalty = self.proximity_penalty_scale * (
                1.0 - self._min_obstacle_dist / self.proximity_safety_dist
            )

        # Speed bonus on checkpoint hits: linearly discounted by elapsed step
        # count. time_factor=1.0 means "got there at step 0", 0.0 means "took
        # the whole budget". Scales the base bonus by `1 + speed_bonus_scale *
        # time_factor`, so at speed_bonus_scale=1 reaching the goal in zero
        # time would pay double (the practical effect: rover learns to rush).
        time_factor = max(0.0, 1.0 - self._step_count / float(self.max_steps))
        speed_bonus = 0.0
        if waypoint_reached_bonus > 0.0:
            speed_bonus += self.speed_bonus_scale * self.waypoint_bonus * time_factor
        if success:
            speed_bonus += self.speed_bonus_scale * self.goal_bonus * time_factor

        reward = (
            self.progress_reward_scale * progress
            - self.step_cost
            - proximity_penalty
            - (self.collision_penalty if collision else 0.0)
            - (self.hit_penalty if new_hit else 0.0)
            - (self.tipped_penalty if tipped else 0.0)
            - (self.stuck_in_collision_penalty if stuck_in_collision and not success else 0.0)
            - (self.stuck_no_progress_penalty if stuck_no_progress and not success else 0.0)
            - action_jerk_penalty
            - wheels_off_penalty
            + waypoint_reached_bonus
            + speed_bonus
            + (self.goal_bonus if success else 0.0)
        )
        self._cumulative_reward += reward

        # distance_to_goal is reported against the FINAL goal (not the current
        # waypoint) so downstream metrics / tests stay consistent across
        # single- and multi-waypoint terrains.
        final_goal = np.array(self.terrain.goal_pos, dtype=np.float32)
        dist_to_goal = float(np.linalg.norm(final_goal - pos_xy))
        # Stash pose so external rollout / plotting code doesn't need private
        # state access. Cheap (3 floats / step) and lets `rollout_with_trajectory`
        # build top-down path plots without coupling to env internals.
        _, yaw_now = self._base_pose_xy()
        # Remember this action for next-step jerk.
        self._prev_action = action.copy()

        info: dict[str, Any] = {
            "distance_to_goal": dist_to_goal,
            "distance_to_target": dist,
            "waypoint_index": self._wp_idx,
            "is_success": bool(success),
            "collision": bool(collision),
            "tipped": bool(tipped),
            "stuck_in_collision": bool(stuck_in_collision),
            "stuck_no_progress": bool(stuck_no_progress),
            "pos_xy": (float(pos_xy[0]), float(pos_xy[1])),
            "yaw": float(yaw_now),
            "n_wheels_off_ground": int(n_wheels_off),
        }
        if terminated or truncated:
            info["episode"] = EpisodeOutcome(
                success=success,
                steps=self._step_count,
                distance_to_goal=dist_to_goal,
                cumulative_reward=self._cumulative_reward,
            ).__dict__
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=240, width=320)
        self._renderer.update_scene(self._data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_env(terrain_name: str, seed: int = 0, **kwargs) -> RoverNavEnv:
    return RoverNavEnv(terrain=terrain_name, seed=seed, **kwargs)
