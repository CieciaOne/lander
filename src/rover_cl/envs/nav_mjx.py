"""MJX-backed parallel rover navigation env.

Same physics as `RoverNavEnv` (rocker-bogie rover on a `TerrainSpec`) but
stepped with MuJoCo XLA (`mujoco.mjx`) so a batch of `n_envs` rovers runs in
parallel under a single `jit(vmap(...))` compile. The intended deployment
target is a CUDA GPU; on Mac CPU JAX it still works but is slower than the
native-MuJoCo `SubprocVecEnv` path.

Two design constraints vs. the CPU env:

  1) **Static obstacle layout per Env instance.** MJX requires `geom_pos` to
     live in the compiled model, so we can't re-roll obstacle positions on
     every reset. Instead, we take ONE `randomize_on_reset` roll at init
     time and compile the model with that. Per-env variation comes from
     spawning + target sampling, which we vectorise across the batch.

  2) **No wheels-off-ground penalty.** Per-step contact analysis in JAX is
     awkward, and the wheels-off heuristic only meaningfully fires when the
     rover is mid-tip — at which point the tipped_penalty already dominates.

The action / obs / reward shapes match `RoverNavEnv` so a policy trained on
one backend can be evaluated on the other.
"""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import Any

import flax.struct
import jax
import jax.numpy as jp
import mujoco
import mujoco.mjx as mjx
import numpy as np

from .nav import (
    ARM_STOW_CTRL,
    GOAL_HOLD_STEPS,
    K_OBSTACLES,
    MAX_STEER_RAD,
    MAX_WHEEL_VEL,
    OBSTACLE_FEATURES_PER_SLOT,
    OBSTACLE_SENSE_RANGE,
    REAR_AXLE_Y,
    ROVER_FOOTPRINT_RADIUS,
    TIP_OVER_COS,
    WHEEL_POS_BODY,
    WHEELBASE,
)
from .terrains import TERRAIN_CATALOG, compile_scene
from .randomization import heightmap_height_at_xy


# Maximum waypoints (including final goal) any RT_ terrain produces. Some
# samplers (e.g. RT_with_two_waypoints with bounce/turn-chains) emit up to 4
# intermediate waypoints + 1 goal = 5 entries; we keep a slot of slack at 6.
# Bumping this is cheap (just more zero padding in the layout pool).
MAX_TARGETS = 6

# Per-instance pre-sampled layout pool size. On reset we pick a random index
# into this pool, so each of `n_envs` envs gets a different start/target set
# without calling CPU samplers per-reset (which would break the JAX boundary).
DEFAULT_POOL_SIZE = 4096

# Sentinel for "no obstacle in this slot" in the obs (matches `RoverNavEnv`'s
# `_build_obstacle_features` behaviour: a thin-line obstacle way outside the
# sense range that the policy learns to ignore).
_OBS_SENTINEL = jp.array(
    [OBSTACLE_SENSE_RANGE, OBSTACLE_SENSE_RANGE + 0.1,
     OBSTACLE_SENSE_RANGE, OBSTACLE_SENSE_RANGE + 0.1],
    dtype=jp.float32,
)


@flax.struct.dataclass
class MjxState:
    """Per-env episode state. Vectorised — all leaves have a leading N axis."""

    # MJX physics state (mjx.Data) — its leaves carry the batch axis.
    data: Any

    # Episode bookkeeping.
    step_count: jp.ndarray             # (N,) int32
    wp_idx: jp.ndarray                 # (N,) int32
    effective_max_steps: jp.ndarray    # (N,) int32
    prev_dist: jp.ndarray              # (N,) float32 — to current target
    best_dist: jp.ndarray              # (N,) float32 — running min vs current target
    steps_since_progress: jp.ndarray   # (N,) int32
    goal_hold: jp.ndarray              # (N,) int32 — consecutive steps inside goal radius
    collision_streak: jp.ndarray       # (N,) int32 — consecutive collision steps
    was_colliding: jp.ndarray          # (N,) bool — was-in-contact flag for new-hit detection
    prev_action: jp.ndarray            # (N, 2) float32 — for jerk penalty
    cumulative_reward: jp.ndarray      # (N,) float32

    # Per-env mission (sampled at reset; held until next reset).
    targets: jp.ndarray                # (N, MAX_TARGETS, 2) float32 — waypoint chain (last entry is final goal)
    target_radii: jp.ndarray           # (N, MAX_TARGETS) float32
    num_targets: jp.ndarray            # (N,) int32 — how many entries are real

    # PRNG key carried so autoreset can sample fresh starts.
    key: jp.ndarray                    # (N, 2) uint32


@dataclasses.dataclass
class MjxReward:
    """Reward / termination knobs. Mirrors `RoverNavEnv` defaults."""

    progress_scale: float = 5.0
    goal_bonus: float = 50.0
    waypoint_bonus: float = 20.0
    speed_bonus_scale: float = 1.0
    step_cost: float = 0.01
    collision_penalty: float = 1.5
    hit_penalty: float = 5.0
    tipped_penalty: float = 20.0
    stuck_in_collision_penalty: float = 5.0
    stuck_no_progress_penalty: float = 30.0
    action_jerk_scale: float = 0.1
    collision_terminate_steps: int = 30
    stuck_window_steps: int = 200
    stuck_min_progress: float = 0.5
    waypoint_time_bonus_per_metre: float = 40.0
    waypoint_time_bonus_base: int = 200


class MjxNavEnv:
    """JAX-vectorised rover-navigation env.

    Not a Gym env — `step()` and `reset()` return JAX arrays. Wrap in
    `MjxVecEnv` for SB3 compatibility (numpy at the boundary).
    """

    def __init__(
        self,
        terrain: str,
        n_envs: int = 64,
        seed: int = 0,
        max_steps: int = 500,
        control_decimation: int = 5,
        reward_cfg: MjxReward | None = None,
        start_jitter_pos: float = 0.5,
        start_jitter_yaw: float = 0.2,
        impl: str = "jax",     # 'jax' (default) or 'warp' (Linux+CUDA + nvidia-warp)
        pool_size: int = DEFAULT_POOL_SIZE,
    ):
        self.terrain_name = terrain
        self.n_envs = n_envs
        self.max_steps = max_steps
        self.control_decimation = control_decimation
        self.reward = reward_cfg or MjxReward()
        self.start_jitter_pos = float(start_jitter_pos)
        self.start_jitter_yaw = float(start_jitter_yaw)

        # ---- terrain spec + concrete roll ----------------------------------
        # The spec defines the invariants (obstacle slots, hfield size, ...).
        # We take ONE roll from `randomize_on_reset` (if any) to bake into
        # the compiled model; that's our static obstacle layout for the life
        # of this Env instance. Per-env variation later comes from the pool.
        spec = TERRAIN_CATALOG[terrain](seed=seed)
        rng_np = np.random.default_rng(seed)
        if spec.randomize_on_reset is not None:
            roll = spec.randomize_on_reset(rng_np)
            spec.start_pos = tuple(roll.start_pos)
            spec.start_yaw = float(roll.start_yaw)
            spec.goal_pos = tuple(roll.goal_pos)
            spec.waypoints = tuple(roll.waypoints)
            # Apply rolled obstacle positions/sizes to the spec so the
            # compiled model has them baked in.
            new_obs = []
            for i, ob in enumerate(spec.obstacles):
                if i < len(roll.obstacle_positions):
                    new_obs.append(dataclasses.replace(
                        ob,
                        pos=tuple(roll.obstacle_positions[i]),
                        size=tuple(roll.obstacle_sizes[i]),
                    ))
                else:
                    new_obs.append(ob)
            spec.obstacles = tuple(new_obs)
            if roll.heightmap is not None:
                spec.heightmap = np.asarray(roll.heightmap, dtype=np.float32)
        self.spec = spec

        # ---- compile MuJoCo model + put on MJX ------------------------------
        model, _ = compile_scene(spec)
        self._mj_model = model
        self.mx_model = mjx.put_model(model, impl=impl)
        self._impl = impl

        # Cache static obstacle data for the obs encoder (JAX arrays, shape
        # (M, 2) for centers, (M, 2) for half-extents in xy). Obstacles with
        # pos.z < ground (HIDE_Z) are masked out via geom_mask.
        if spec.obstacles:
            obs_xy = np.array([[ob.pos[0], ob.pos[1]] for ob in spec.obstacles], dtype=np.float32)
            obs_hwhh = np.array([[ob.size[0], ob.size[1]] for ob in spec.obstacles], dtype=np.float32)
            obs_z = np.array([ob.pos[2] for ob in spec.obstacles], dtype=np.float32)
            obs_mask = (obs_z > -10.0).astype(np.float32)  # 1 = visible, 0 = parked below floor
        else:
            obs_xy = np.zeros((0, 2), dtype=np.float32)
            obs_hwhh = np.zeros((0, 2), dtype=np.float32)
            obs_mask = np.zeros((0,), dtype=np.float32)
        self._obs_xy = jp.asarray(obs_xy)
        self._obs_hwhh = jp.asarray(obs_hwhh)
        self._obs_visible_mask = jp.asarray(obs_mask)
        self._n_obstacles = obs_xy.shape[0]

        # Cache obstacle GEOM IDs for collision detection. In the compiled
        # model these are named "obs_<i>". We mark them in a boolean array
        # of length ngeom so `_compute_collision` can scan all contacts and
        # match geoms by id (jit-friendly: no Python loops at runtime).
        obs_geom_ids = []
        for i in range(self._n_obstacles):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"obs_{i}")
            if gid >= 0:
                obs_geom_ids.append(gid)
        is_obstacle_geom = np.zeros(model.ngeom, dtype=bool)
        for gid in obs_geom_ids:
            is_obstacle_geom[gid] = True
        # Rover-tree geoms (anything under base_link) for "rover vs obstacle"
        # check. Geoms directly named ground / obs_* / arena_* are world; the
        # rest belong to rover bodies.
        rover_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        rover_body_set = set()
        for b in range(model.nbody):
            cur = b
            while cur != 0:
                if cur == rover_body_id:
                    rover_body_set.add(b)
                    break
                cur = model.body_parentid[cur]
        is_rover_geom = np.array(
            [model.geom_bodyid[g] in rover_body_set for g in range(model.ngeom)],
            dtype=bool,
        )
        self._is_rover_geom = jp.asarray(is_rover_geom)
        self._is_obstacle_geom = jp.asarray(is_obstacle_geom)

        # ---- actuator id lookup --------------------------------------------
        # Drive actuators 0..5 are wheels in WHEEL_POS_BODY order; 6..9 are
        # the 4 corner steers; 10..13 are arm joints. Match `RoverNavEnv`.
        self._wheel_act_ids = jp.asarray([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in [
                "drive_right_front", "drive_right_middle", "drive_right_rear",
                "drive_left_front",  "drive_left_middle",  "drive_left_rear",
            ]
        ])
        self._steer_act_ids = jp.asarray([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in ["steer_right_front", "steer_right_rear",
                      "steer_left_front",  "steer_left_rear"]
        ])
        self._arm_act_ids = jp.asarray([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in ["arm_yaw", "arm_shoulder", "arm_elbow", "arm_wrist"]
        ])
        self._wheel_pos_body = jp.asarray(WHEEL_POS_BODY, dtype=jp.float32)

        # Sensor address lookup for IMU + body pose readout.
        def _sensor_adr(name: str) -> int:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            return int(model.sensor_adr[sid])
        self._adr_pos = _sensor_adr("base_pos")
        self._adr_quat = _sensor_adr("base_quat")
        self._adr_linvel = _sensor_adr("base_linvel")
        self._adr_angvel = _sensor_adr("base_angvel")

        # ---- per-env mission pool ------------------------------------------
        self._layout_pool = self._build_layout_pool(spec, seed=seed + 1, size=pool_size)

        # ---- compile step/reset (lazy on first call) -----------------------
        # We jit + vmap them. Static args go to partial.
        self._jit_step = jax.jit(jax.vmap(self._step_one, in_axes=(0, 0)))
        # Reset is a vmap over per-env subkey + layout index.
        self._jit_reset_envs = jax.jit(jax.vmap(self._reset_one, in_axes=(None, 0, 0)))

        self.obs_dim = 6 + 2 + K_OBSTACLES * OBSTACLE_FEATURES_PER_SLOT
        self.action_dim = 2

        self.state: MjxState | None = None

    # ====================================================================== layout pool

    def _build_layout_pool(self, spec, seed: int, size: int) -> dict[str, jp.ndarray]:
        """Pre-sample `size` (start_xy, yaw, target_chain) tuples in numpy.

        On reset, each env picks a random index into this pool and gets a
        unique mission. This sidesteps trying to call the numpy samplers
        from inside a jitted function.
        """
        rng = np.random.default_rng(seed)
        start_xys = np.zeros((size, 2), dtype=np.float32)
        start_yaws = np.zeros((size,), dtype=np.float32)
        target_xys = np.zeros((size, MAX_TARGETS, 2), dtype=np.float32)
        target_radii = np.zeros((size, MAX_TARGETS), dtype=np.float32)
        num_targets = np.zeros((size,), dtype=np.int32)

        for i in range(size):
            if spec.randomize_on_reset is not None:
                roll = spec.randomize_on_reset(rng)
                sx, sy = roll.start_pos
                yaw0 = roll.start_yaw
                wps = list(roll.waypoints)
                goal = roll.goal_pos
            else:
                sx, sy = spec.start_pos
                yaw0 = spec.start_yaw
                wps = list(spec.waypoints)
                goal = spec.goal_pos

            # Apply per-reset jitter on start xy / yaw (mirrors RoverNavEnv).
            sx = sx + rng.uniform(-self.start_jitter_pos, self.start_jitter_pos)
            sy = sy + rng.uniform(-self.start_jitter_pos, self.start_jitter_pos)
            yaw0 = yaw0 + rng.uniform(-self.start_jitter_yaw, self.start_jitter_yaw)

            start_xys[i] = (sx, sy)
            start_yaws[i] = yaw0

            chain = list(wps) + [goal]
            chain_radii = (
                [spec.waypoint_radius] * len(wps) + [spec.goal_radius]
            )
            n = min(len(chain), MAX_TARGETS)
            for k in range(n):
                target_xys[i, k] = chain[k]
                target_radii[i, k] = chain_radii[k]
            num_targets[i] = n

        return {
            "start_xys": jp.asarray(start_xys),
            "start_yaws": jp.asarray(start_yaws),
            "target_xys": jp.asarray(target_xys),
            "target_radii": jp.asarray(target_radii),
            "num_targets": jp.asarray(num_targets),
        }

    # ====================================================================== Ackermann routing

    def _action_to_ctrl(self, action: jp.ndarray) -> jp.ndarray:
        """Map 2-D action to 14-D MuJoCo ctrl vector (per-wheel Ackermann)."""
        action = jp.clip(action, -1.0, 1.0)
        throttle = action[0]
        steer = action[1]

        v_base = -throttle * MAX_WHEEL_VEL    # negate: positive ctrl rolls -Y
        steer_rad = steer * MAX_STEER_RAD

        # Ackermann differential. With front-wheel steer δ, the ICR sits at
        # body-frame (R_x, REAR_AXLE_Y) where R_x = WHEELBASE / tan(δ). Each
        # wheel's commanded angular velocity scales with distance from ICR.
        # We normalise so the fastest wheel hits v_base exactly.
        eps = 1e-3
        use_arc = jp.abs(steer_rad) >= eps
        R = WHEELBASE / jp.tan(jp.where(use_arc, steer_rad, eps))
        R_abs = jp.abs(R)
        diffs = self._wheel_pos_body - jp.array([R, REAR_AXLE_Y])
        mults_arc = jp.linalg.norm(diffs, axis=1) / R_abs
        # Normalise so max=1.
        mults_arc = mults_arc / jp.maximum(jp.max(mults_arc), 1.0)
        # When going straight, all wheels equal.
        mults = jp.where(use_arc, mults_arc, jp.ones(6))
        drive_ctrl = v_base * mults

        steer_cmd = steer_rad
        # Right_front, right_rear, left_front, left_rear receive (-, +, -, +).
        steer_ctrl = jp.array([-steer_cmd, +steer_cmd, -steer_cmd, +steer_cmd])

        arm_ctrl = jp.array(ARM_STOW_CTRL, dtype=jp.float32)

        ctrl = jp.zeros(self._mj_model.nu, dtype=jp.float32)
        ctrl = ctrl.at[self._wheel_act_ids].set(drive_ctrl)
        ctrl = ctrl.at[self._steer_act_ids].set(steer_ctrl)
        ctrl = ctrl.at[self._arm_act_ids].set(arm_ctrl)
        return ctrl

    # ====================================================================== sensors / obs

    def _pose_xy_yaw(self, data) -> tuple[jp.ndarray, jp.ndarray]:
        """Read base XY position + yaw from sensordata."""
        sd = data.sensordata
        pos_xy = sd[self._adr_pos: self._adr_pos + 2]
        w = sd[self._adr_quat + 0]
        x = sd[self._adr_quat + 1]
        y = sd[self._adr_quat + 2]
        z = sd[self._adr_quat + 3]
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        yaw = jp.arctan2(siny, cosy)
        return pos_xy, yaw

    def _upright_cos(self, data) -> jp.ndarray:
        sd = data.sensordata
        x = sd[self._adr_quat + 1]
        y = sd[self._adr_quat + 2]
        return 1.0 - 2.0 * (x * x + y * y)

    def _obstacle_features(self, pos_xy, yaw) -> jp.ndarray:
        """Return (K_OBSTACLES * 4,) feature vector. Same semantics as the
        CPU env's `_build_obstacle_features` but vectorised over all M
        terrain obstacles, then top-K-by-distance with sentinel pad."""
        if self._n_obstacles == 0:
            return jp.tile(_OBS_SENTINEL, K_OBSTACLES)

        # For each obstacle, compute nearest world-AABB point to rover and
        # the inflated body-frame AABB of its 4 corners.
        c = jp.cos(yaw)
        s = jp.sin(yaw)
        rx, ry = pos_xy[0], pos_xy[1]

        ox = self._obs_xy[:, 0]
        oy = self._obs_xy[:, 1]
        sx = self._obs_hwhh[:, 0]
        sy = self._obs_hwhh[:, 1]

        # Nearest-point distance to AABB.
        nx = jp.clip(rx, ox - sx, ox + sx)
        ny = jp.clip(ry, oy - sy, oy + sy)
        d_near = jp.hypot(nx - rx, ny - ry)

        # 4 world corners per obstacle: (ox ± sx, oy ± sy). Stack into
        # (M, 4, 2). Then translate to rover-relative and rotate by -yaw to
        # get body-frame coords. Body forward = world (-sin yaw, cos yaw).
        signs = jp.array([[-1, -1], [-1, +1], [+1, -1], [+1, +1]], dtype=jp.float32)
        corners = (
            jp.stack([ox, oy], axis=-1)[:, None, :]      # (M, 1, 2)
            + signs[None, :, :] * jp.stack([sx, sy], axis=-1)[:, None, :]
        )                                                 # (M, 4, 2)
        dx = corners[..., 0] - rx                        # (M, 4)
        dy = corners[..., 1] - ry                        # (M, 4)
        fwd = -s * dx + c * dy
        right = c * dx + s * dy
        fwd_min = jp.min(fwd, axis=1) - ROVER_FOOTPRINT_RADIUS
        fwd_max = jp.max(fwd, axis=1) + ROVER_FOOTPRINT_RADIUS
        right_min = jp.min(right, axis=1) - ROVER_FOOTPRINT_RADIUS
        right_max = jp.max(right, axis=1) + ROVER_FOOTPRINT_RADIUS
        feats = jp.stack([fwd_min, fwd_max, right_min, right_max], axis=-1)  # (M, 4)

        # Mask out hidden / too-far obstacles by pushing their distance to inf.
        invalid = (self._obs_visible_mask < 0.5) | (d_near > OBSTACLE_SENSE_RANGE)
        d_sorted_key = jp.where(invalid, jp.inf, d_near)

        # Top-K closest. `top_k` returns largest, so negate.
        k = min(K_OBSTACLES, self._n_obstacles)
        _, idx = jax.lax.top_k(-d_sorted_key, k=k)
        feats_k = feats[idx]                                        # (k, 4)
        valid_k = ~jp.isinf(d_sorted_key[idx])                      # (k,)
        sentinel = jp.broadcast_to(_OBS_SENTINEL, (k, 4))
        feats_k = jp.where(valid_k[:, None], feats_k, sentinel)

        # Pad to K_OBSTACLES with sentinel.
        if k < K_OBSTACLES:
            pad = jp.broadcast_to(_OBS_SENTINEL, (K_OBSTACLES - k, 4))
            feats_k = jp.concatenate([feats_k, pad], axis=0)

        return feats_k.reshape(-1)

    def _build_obs(self, state: MjxState, data) -> jp.ndarray:
        pos_xy, yaw = self._pose_xy_yaw(data)
        targets = state.targets        # (MAX_TARGETS, 2)
        target = targets[state.wp_idx]
        delta = target - pos_xy
        c, s = jp.cos(yaw), jp.sin(yaw)
        rel_fwd = -s * delta[0] + c * delta[1]
        rel_right = c * delta[0] + s * delta[1]
        heading = jp.arctan2(rel_right, rel_fwd)

        sd = data.sensordata
        linvel = sd[self._adr_linvel: self._adr_linvel + 2]
        angvel_z = sd[self._adr_angvel + 2]

        # Lookahead to next target if there is one; sentinel (0, 0) otherwise.
        next_idx = jp.minimum(state.wp_idx + 1, state.num_targets - 1)
        has_next = state.wp_idx + 1 < state.num_targets
        next_target = targets[next_idx]
        d_nxt = next_target - pos_xy
        rel_fwd_next = jp.where(has_next, -s * d_nxt[0] + c * d_nxt[1], 0.0)
        rel_right_next = jp.where(has_next, c * d_nxt[0] + s * d_nxt[1], 0.0)

        obstacle_feats = self._obstacle_features(pos_xy, yaw)

        obs = jp.zeros(self.obs_dim, dtype=jp.float32)
        obs = obs.at[0].set(rel_fwd)
        obs = obs.at[1].set(rel_right)
        obs = obs.at[2].set(heading)
        obs = obs.at[3].set(linvel[0])
        obs = obs.at[4].set(linvel[1])
        obs = obs.at[5].set(angvel_z)
        obs = obs.at[6].set(rel_fwd_next)
        obs = obs.at[7].set(rel_right_next)
        obs = obs.at[8:].set(obstacle_feats)
        return obs

    # ====================================================================== collision

    def _detect_collision(self, data) -> jp.ndarray:
        """True iff any contact this step is between a rover geom and an
        obstacle geom (`obs_*`)."""
        if self._n_obstacles == 0:
            return jp.array(False)
        contact = data._impl.contact
        g1 = contact.geom[:, 0]
        g2 = contact.geom[:, 1]
        dist = contact.dist
        # mjx contact arrays are fixed-size; valid contacts have dist <= 0
        # (penetrating) and an "active" mask we approximate via dist threshold.
        valid = dist <= 0.0
        # Look up category membership for each side.
        rover_1 = self._is_rover_geom[g1]
        rover_2 = self._is_rover_geom[g2]
        obs_1 = self._is_obstacle_geom[g1]
        obs_2 = self._is_obstacle_geom[g2]
        hit = (rover_1 & obs_2) | (rover_2 & obs_1)
        return jp.any(valid & hit)

    # ====================================================================== reset (single env)

    def _set_qpos_for_spawn(self, data, start_xy, yaw0):
        """Place the rover at (start_xy, yaw0) at the correct spawn altitude,
        with zeroed velocities."""
        if self.spec.heightmap is not None:
            terrain_z = heightmap_height_at_xy(
                self.spec.heightmap, self.spec.heightmap_extent,
                float(np.asarray(start_xy[0])), float(np.asarray(start_xy[1])),
            )
        else:
            terrain_z = 0.0
        # SPAWN_CLEARANCE = 0.20, same as CPU env.
        z = terrain_z + 0.75 + 0.20

        qpos = data.qpos
        qpos = qpos.at[0].set(start_xy[0])
        qpos = qpos.at[1].set(start_xy[1])
        qpos = qpos.at[2].set(z)
        qpos = qpos.at[3].set(jp.cos(yaw0 / 2))
        qpos = qpos.at[4].set(0.0)
        qpos = qpos.at[5].set(0.0)
        qpos = qpos.at[6].set(jp.sin(yaw0 / 2))
        qvel = jp.zeros_like(data.qvel)
        ctrl = jp.zeros_like(data.ctrl)
        # Pin arm to stow pose during settle.
        ctrl = ctrl.at[self._arm_act_ids].set(jp.array(ARM_STOW_CTRL))
        return data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)

    # Settle steps after spawning. The CPU env uses 150 for safety; in MJX
    # the autoreset branch runs the settle loop on every step (vmap turns
    # `cond` into select-both-branches), so we use 50 — still enough for a
    # 0.20 m fall + bogie damping under Mars gravity (~66 steps), tightened
    # by accepting a small initial qvel residual the policy can absorb.
    _SETTLE_STEPS = 50

    def _settle(self, data, n_steps: int | None = None):
        n = n_steps if n_steps is not None else self._SETTLE_STEPS

        def body(i, dat):
            return mjx.step(self.mx_model, dat)

        return jax.lax.fori_loop(0, n, body, data)

    def _reset_one(self, base_data, key, layout_idx) -> MjxState:
        """Reset a single env to a freshly sampled mission."""
        pool = self._layout_pool
        start_xy = pool["start_xys"][layout_idx]
        yaw0 = pool["start_yaws"][layout_idx]
        targets = pool["target_xys"][layout_idx]
        radii = pool["target_radii"][layout_idx]
        num_t = pool["num_targets"][layout_idx]

        data = self._set_qpos_for_spawn(base_data, start_xy, yaw0)
        data = self._settle(data, n_steps=150)

        pos_xy, _ = self._pose_xy_yaw(data)
        first_target = targets[0]
        dist0 = jp.linalg.norm(first_target - pos_xy)

        return MjxState(
            data=data,
            step_count=jp.zeros((), dtype=jp.int32),
            wp_idx=jp.zeros((), dtype=jp.int32),
            effective_max_steps=jp.array(self.max_steps, dtype=jp.int32),
            prev_dist=dist0.astype(jp.float32),
            best_dist=dist0.astype(jp.float32),
            steps_since_progress=jp.zeros((), dtype=jp.int32),
            goal_hold=jp.zeros((), dtype=jp.int32),
            collision_streak=jp.zeros((), dtype=jp.int32),
            was_colliding=jp.array(False),
            prev_action=jp.zeros(2, dtype=jp.float32),
            cumulative_reward=jp.zeros((), dtype=jp.float32),
            targets=targets,
            target_radii=radii,
            num_targets=num_t,
            key=key,
        )

    # ====================================================================== step (single env)

    def _step_one(self, state: MjxState, action: jp.ndarray):
        """Step one env. Returns (new_state, obs, reward, done, info_dict).

        On done, auto-resets to a fresh mission from the pool (sampled from
        `state.key`). The pre-done obs/reward/etc are still returned (SB3
        VecEnv convention)."""
        rc = self.reward

        action = jp.clip(action, -1.0, 1.0)

        # Build ctrl and step physics `control_decimation` times.
        ctrl = self._action_to_ctrl(action)
        def step_body(i, dat):
            return mjx.step(self.mx_model, dat.replace(ctrl=ctrl))
        data = jax.lax.fori_loop(0, self.control_decimation, step_body, state.data)

        pos_xy, yaw_now = self._pose_xy_yaw(data)
        target = state.targets[state.wp_idx]
        dist = jp.linalg.norm(target - pos_xy)
        radius = state.target_radii[state.wp_idx]
        is_final = state.wp_idx == state.num_targets - 1
        reached = dist < radius

        progress = state.prev_dist - dist

        # Waypoint advance: if reached and NOT final, bump wp_idx, reset
        # prev_dist / best_dist / steps_since_progress / goal_hold against
        # the new target. The final-target case keeps wp_idx pinned and lets
        # goal_hold climb.
        new_wp_idx = jp.where(reached & ~is_final, state.wp_idx + 1, state.wp_idx)
        new_target = state.targets[new_wp_idx]
        dist_to_new = jp.linalg.norm(new_target - pos_xy)
        is_waypoint_hit = reached & ~is_final

        new_goal_hold = jp.where(
            reached & is_final, state.goal_hold + 1,
            jp.where(is_waypoint_hit, jp.array(0, dtype=jp.int32), jp.array(0, dtype=jp.int32)),
        )

        # Adaptive time budget: extend on waypoint hits.
        bonus_steps = jp.where(
            is_waypoint_hit,
            (rc.waypoint_time_bonus_per_metre * dist_to_new
             + rc.waypoint_time_bonus_base).astype(jp.int32),
            jp.array(0, dtype=jp.int32),
        )
        new_eff_max = state.effective_max_steps + bonus_steps

        new_prev_dist = jp.where(is_waypoint_hit, dist_to_new, dist)

        # Stuck-detection on the CURRENT target (or, after a waypoint hit,
        # against the NEW target — reset best/no-progress counters then).
        improved = (dist + rc.stuck_min_progress) < state.best_dist
        new_best = jp.where(
            is_waypoint_hit,
            dist_to_new,
            jp.where(improved, dist, state.best_dist),
        )
        new_no_progress = jp.where(
            is_waypoint_hit,
            jp.array(0, dtype=jp.int32),
            jp.where(improved,
                     jp.array(0, dtype=jp.int32),
                     state.steps_since_progress + 1),
        )

        collision = self._detect_collision(data)
        new_hit = collision & ~state.was_colliding
        new_collision_streak = jp.where(
            collision, state.collision_streak + 1, jp.array(0, dtype=jp.int32),
        )

        upright = self._upright_cos(data)
        tipped = upright < TIP_OVER_COS

        stuck_in_coll = (rc.collision_terminate_steps > 0) & (
            new_collision_streak >= rc.collision_terminate_steps
        )
        stuck_no_prog = (rc.stuck_window_steps > 0) & (
            new_no_progress >= rc.stuck_window_steps
        )

        success = new_goal_hold >= GOAL_HOLD_STEPS
        truncated = (state.step_count + 1) >= new_eff_max
        done = success | tipped | stuck_in_coll | stuck_no_prog | truncated

        # Reward.
        action_jerk_sq = jp.sum((action - state.prev_action) ** 2)
        jerk_pen = rc.action_jerk_scale * action_jerk_sq

        time_factor = jp.maximum(
            0.0, 1.0 - (state.step_count + 1) / jp.float32(self.max_steps),
        )
        wp_reached_bonus = jp.where(is_waypoint_hit, rc.waypoint_bonus, 0.0)
        speed_bonus = (
            jp.where(is_waypoint_hit, rc.speed_bonus_scale * rc.waypoint_bonus * time_factor, 0.0)
            + jp.where(success, rc.speed_bonus_scale * rc.goal_bonus * time_factor, 0.0)
        )

        reward = (
            rc.progress_scale * progress
            - rc.step_cost
            - jp.where(collision, rc.collision_penalty, 0.0)
            - jp.where(new_hit, rc.hit_penalty, 0.0)
            - jp.where(tipped, rc.tipped_penalty, 0.0)
            - jp.where(stuck_in_coll & ~success, rc.stuck_in_collision_penalty, 0.0)
            - jp.where(stuck_no_prog & ~success, rc.stuck_no_progress_penalty, 0.0)
            - jerk_pen
            + wp_reached_bonus
            + speed_bonus
            + jp.where(success, rc.goal_bonus, 0.0)
        )

        # Provisional next state (pre-autoreset).
        next_state = MjxState(
            data=data,
            step_count=state.step_count + 1,
            wp_idx=new_wp_idx,
            effective_max_steps=new_eff_max,
            prev_dist=new_prev_dist,
            best_dist=new_best,
            steps_since_progress=new_no_progress,
            goal_hold=new_goal_hold,
            collision_streak=new_collision_streak,
            was_colliding=collision,
            prev_action=action,
            cumulative_reward=state.cumulative_reward + reward,
            targets=state.targets,
            target_radii=state.target_radii,
            num_targets=state.num_targets,
            key=state.key,
        )

        # Obs computed from the pre-autoreset state (SB3 convention).
        obs = self._build_obs(next_state, data)

        # Autoreset on done.
        new_key, sub_key = jax.random.split(state.key)
        layout_idx = jax.random.randint(
            sub_key, (), 0, self._layout_pool["start_xys"].shape[0],
        )
        reset_state = self._reset_one(data, new_key, layout_idx)
        # When done, replace next_state with the fresh reset (Brax/MJX convention).
        next_state = jax.tree_util.tree_map(
            lambda new, keep: jp.where(done, new, keep),
            reset_state, next_state,
        )

        info = {
            "is_success": success,
            "collision": collision,
            "tipped": tipped,
            "stuck_in_collision": stuck_in_coll,
            "stuck_no_progress": stuck_no_prog,
            "distance_to_goal": dist,           # distance to FINAL goal computed below in vec wrapper
            "distance_to_target": dist,
            "waypoint_index": state.wp_idx,
            "goal_hold": new_goal_hold,
            "pos_xy": pos_xy,
            "yaw": yaw_now,
            "truncated": truncated & ~success,
            "terminated": (success | tipped | stuck_in_coll | stuck_no_prog),
            "episode_return": next_state.cumulative_reward,  # post-reset zero on done
            "episode_steps": state.step_count + 1,
            "done": done,
        }

        return next_state, obs, reward, done, info

    # ====================================================================== public API (batched)

    def reset(self, seed: int | None = None) -> tuple[jp.ndarray, dict[str, jp.ndarray]]:
        """Reset all N envs; return batched obs and info."""
        if seed is None:
            seed = int(np.random.randint(0, 1 << 31))
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, self.n_envs)
        layout_keys = jax.random.split(jax.random.fold_in(key, 7), self.n_envs)
        layout_idxs = jax.vmap(
            lambda k: jax.random.randint(k, (), 0, self._layout_pool["start_xys"].shape[0])
        )(layout_keys)

        base_data = mjx.make_data(self.mx_model)
        state = self._jit_reset_envs(base_data, keys, layout_idxs)
        self.state = state

        # Compute initial obs.
        obs = jax.vmap(self._build_obs)(state, state.data)
        info = {
            "distance_to_target": jax.vmap(
                lambda s: jp.linalg.norm(s.targets[s.wp_idx] - self._pose_xy_yaw(s.data)[0])
            )(state),
        }
        return obs, info

    def step(self, actions: jp.ndarray):
        """Step all N envs with `actions` (N, 2). Returns (obs, reward, done, info)."""
        assert self.state is not None, "call reset() first"
        new_state, obs, reward, done, info = self._jit_step(self.state, actions)
        self.state = new_state
        return obs, reward, done, info


def make_mjx_env(terrain: str, n_envs: int = 64, seed: int = 0, **kwargs) -> MjxNavEnv:
    return MjxNavEnv(terrain=terrain, n_envs=n_envs, seed=seed, **kwargs)
