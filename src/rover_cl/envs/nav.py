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

import heapq
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .terrains import TerrainSpec, compile_scene, compose_scene, get_terrain


class NavField:
    """Obstacle-aware distance-to-target field over a grid (a navigation
    function). Cell value = shortest collision-free path length from that cell
    to the target, computed by Dijkstra on an 8-connected grid where cells
    covered by an obstacle (inflated by the rover footprint) are blocked.

    Used for `progress_reward_mode="geodesic"`: rewarding the reduction of this
    geodesic distance gives the policy a gradient that routes AROUND blocking
    obstacles, instead of the straight-line Euclidean distance whose gradient
    points through them. On obstacle-free terrain the field equals Euclidean
    distance, so open-terrain behaviour is unchanged.
    """

    _SQRT2 = float(np.sqrt(2.0))

    def __init__(self, half_extent: float, res: float,
                 obstacles_xywh: list[tuple[float, float, float, float]],
                 target_xy: tuple[float, float], inflate: float):
        self.half = float(half_extent)
        self.res = float(res)
        self.n = max(4, int(np.ceil(2 * self.half / self.res)))
        blocked = np.zeros((self.n, self.n), dtype=bool)
        for (cx, cy, sx, sy) in obstacles_xywh:
            x0 = cx - sx - inflate; x1 = cx + sx + inflate
            y0 = cy - sy - inflate; y1 = cy + sy + inflate
            i0, i1 = self._idx(x0), self._idx(x1)
            j0, j1 = self._idx(y0), self._idx(y1)
            blocked[i0:i1 + 1, j0:j1 + 1] = True
        # Dijkstra from the target cell. Force the target cell free so there is
        # always a source even if the goal sits just inside an inflated AABB.
        ti, tj = self._idx(target_xy[0]), self._idx(target_xy[1])
        blocked[ti, tj] = False
        INF = np.inf
        dist = np.full((self.n, self.n), INF, dtype=np.float64)
        dist[ti, tj] = 0.0
        heap = [(0.0, ti, tj)]
        n = self.n
        while heap:
            d, i, j = heapq.heappop(heap)
            if d > dist[i, j]:
                continue
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if ni < 0 or nj < 0 or ni >= n or nj >= n:
                        continue
                    if blocked[ni, nj]:
                        continue
                    step = self.res * (self._SQRT2 if (di and dj) else 1.0)
                    nd = d + step
                    if nd < dist[ni, nj]:
                        dist[ni, nj] = nd
                        heapq.heappush(heap, (nd, ni, nj))
        self.dist = dist

    def _idx(self, w: float) -> int:
        return int(np.clip((w + self.half) / self.res, 0, self.n - 1))

    def distance(self, x: float, y: float) -> float:
        """Geodesic distance from world (x, y) to the target; NaN if the cell
        is unreachable/blocked (caller falls back to the previous value)."""
        d = float(self.dist[self._idx(x), self._idx(y)])
        return d if np.isfinite(d) else float("nan")

    def heading(self, x: float, y: float) -> tuple[float, float] | None:
        """World-frame unit vector along the shortest collision-free path from
        (x, y) toward the target — the smooth direction of steepest geodesic
        descent, i.e. -normalize(∇dist) via central differences of the distance
        field. Continuous (not quantized to 8 compass directions), so it makes
        a clean control target. None if the local field is unusable (all
        neighbours blocked / at target). This is the 'which way to go, routing
        around obstacles' signal; the straight-line bearing points through
        blocking obstacles."""
        i, j = self._idx(x), self._idx(y)
        n = self.n
        d0 = self.dist[i, j]
        if not np.isfinite(d0):
            return None

        def _grad(a_lo, a_hi, cur):
            # central difference where both sides finite; else one-sided;
            # returns None if neither neighbour is usable.
            lo_ok = np.isfinite(a_lo)
            hi_ok = np.isfinite(a_hi)
            if lo_ok and hi_ok:
                return (a_hi - a_lo) * 0.5
            if hi_ok:
                return a_hi - cur
            if lo_ok:
                return cur - a_lo
            return None

        gx = _grad(self.dist[i - 1, j] if i > 0 else np.inf,
                   self.dist[i + 1, j] if i < n - 1 else np.inf, d0)
        gy = _grad(self.dist[i, j - 1] if j > 0 else np.inf,
                   self.dist[i, j + 1] if j < n - 1 else np.inf, d0)
        if gx is None:
            gx = 0.0
        if gy is None:
            gy = 0.0
        # descent direction = -gradient
        vx, vy = -float(gx), -float(gy)
        norm = (vx * vx + vy * vy) ** 0.5
        if norm < 1e-6:
            return None
        return (vx / norm, vy / norm)

    def lookahead_heading(self, x: float, y: float,
                          lookahead_cells: int = 8) -> tuple[float, float] | None:
        """Pure-pursuit heading: walk down the geodesic (greedy min-distance
        8-neighbour) for up to `lookahead_cells` steps and return the world
        unit vector from (x, y) to that lookahead cell. Anticipates the route
        so the rover turns EARLY around obstacles instead of chasing the local
        gradient (which only bends hard once it's already at the obstacle and
        so gets clipped). Falls back to the local gradient if the walk stalls
        immediately."""
        i0, j0 = self._idx(x), self._idx(y)
        n = self.n
        if not np.isfinite(self.dist[i0, j0]):
            return None
        i, j = i0, j0
        for _ in range(lookahead_cells):
            best = self.dist[i, j]
            bi, bj = i, j
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n and self.dist[ni, nj] < best:
                        best = self.dist[ni, nj]
                        bi, bj = ni, nj
            if bi == i and bj == j:
                break
            i, j = bi, bj
        if i == i0 and j == j0:
            return self.heading(x, y)
        vx, vy = float(i - i0), float(j - j0)
        norm = (vx * vx + vy * vy) ** 0.5
        return (vx / norm, vy / norm)


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

# Body-frame tilt features (2): the world-up vector projected into the rover
# body-XY plane. Both ~0 when upright; tilt_fwd > 0 = nose up, tilt_right > 0
# = leaning right. Range [-1, 1]. Critical on hfield terrains where the
# policy otherwise can only react to a tilt *after* the chassis has already
# rolled — these features make slopes visible before the body integrates
# them into linvel/angvel.
N_TILT_FEATURES = 2

# Previous action (2): action_{t-1} clipped to [-1, 1]. Lets the policy
# satisfy the action-jerk penalty by issuing smooth continuations instead
# of having to reconstruct prior command from velocity.
N_PREV_ACTION_FEATURES = 2

# Lidar forward-fan config (used only when RoverNavEnv(use_lidar=True)).
# A ray scan is the standard egocentric obstacle representation; the abstract
# 8×AABB slots alone proved too hard for the policy to route around blocking
# obstacles. Rays are masked to LIDAR_OBSTACLE_GROUP (obstacle geoms only).
N_LIDAR_RAYS = 15                   # 15 rays over ±90° ≈ 12.8° spacing (was 9)
LIDAR_HALF_ARC = float(np.pi / 2)   # ±90° forward fan
LIDAR_OBSTACLE_GROUP = 4            # private geom group for raycast masking
LIDAR_HEIGHT_OFFSET = 0.35          # ray height below the rover base (≈ obstacle mid-height)


def _build_obs_scale() -> np.ndarray:
    """Fixed per-component observation scaling → everything lands in ~[-3, 3].

    FIXED constants rather than running statistics (VecNormalize) — in a
    continual-learning experiment the obs distribution shifts between tasks,
    so running stats would themselves drift and re-normalize OLD tasks' obs
    differently over time. That drift is indistinguishable from forgetting
    in the retention metrics. Fixed scaling has no such confounder, needs no
    train/eval/checkpoint synchronization, and is trivially reproducible.

    Scales (divisors): positions/distances 10 m, heading π rad, linear
    velocities 1 m/s, yaw rate 2 rad/s, obstacle AABB coords 10 m, tilt and
    prev_action already in [-1, 1].
    """
    scale = np.ones(
        6 + 2 + K_OBSTACLES * OBSTACLE_FEATURES_PER_SLOT
        + N_TILT_FEATURES + N_PREV_ACTION_FEATURES,
        dtype=np.float32,
    )
    scale[0:2] = 10.0          # rel_fwd, rel_right (m)
    scale[2] = np.pi           # heading_to_target (rad)
    scale[3:5] = 1.0           # body-frame linvel (m/s)
    scale[5] = 2.0             # angvel_z (rad/s)
    scale[6:8] = 10.0          # lookahead rel_fwd_next, rel_right_next (m)
    scale[8:8 + K_OBSTACLES * OBSTACLE_FEATURES_PER_SLOT] = 10.0  # AABBs (m)
    # tilt (2) and prev_action (2) stay at 1.0
    return scale


OBS_SCALE: np.ndarray = _build_obs_scale()

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
WHEEL_RADIUS = 0.22     # m — drive wheel radius (matches rover.xml sphere collider)
# (v, ω) control-mode limits. V_MAX ≈ MAX_WHEEL_VEL·r (no-slip top speed);
# OMEGA_MAX chosen so a point-turn (corner wheels at radius ≈1.07 m) stays
# within the drive actuator limit: 0.66/1.07 ≈ 0.62 rad/s, rounded to 0.7.
V_MAX = MAX_WHEEL_VEL * WHEEL_RADIUS   # ≈ 0.66 m/s (traction-limited straight speed)
# Maneuverability upgrade (vw mode only): give the (v, ω) controller a higher
# wheel-speed headroom than MAX_WHEEL_VEL so that turning-WHILE-moving no longer
# saturates and gets scaled down (the arc-stall that made every avoidance
# controller freeze ~1.5 m short of an obstacle). Straight speed stays capped by
# traction (~0.66 m/s); this only frees up the extra wheel speed a turn needs.
# The drive actuator ctrlrange is widened to match in RoverNavEnv.__init__ when
# control_mode="vw". Does NOT affect the Ackermann path / scenario_14.
VW_MAX_WHEEL_VEL = 6.0                   # rad/s (vw normalization headroom)
OMEGA_MAX = 0.8                          # rad/s (realistic moderate; Curiosity-style point-turn)
# Full mechanical steer range of the corner knuckles (rover.xml jnt_range
# ±57°). The (v, ω) controller commands geometrically-correct per-wheel
# angles up to this; the reduced MAX_STEER_RAD only bounds the legacy
# single-steer Ackermann action, not the vw controller.
MAX_STEER_JOINT = 1.0   # rad
# Corner-steer angle at |steer|=1. Reduced 1.0 → 0.40 rad (~23°) after a
# turning-authority audit: at the old 1.0 rad (57°) the un-steered middle
# wheels bind and the rover STALLS during a turn (~-4°/s yaw at 0.09 m/s).
# Peak turning happens near 0.3-0.45 rad (~-10°/s at 0.45 m/s), so mapping
# the full action extreme to 0.40 rad makes the ENTIRE steer axis useful
# and removes the over-steer stall trap that made navigation phases
# unlearnable (the policy kept commanding max steer and freezing). The
# steer joint range is ±1.0 rad, so 0.40 is well within actuator limits.
MAX_STEER_RAD = 0.40    # rad — corner steer angle at |steer|=1
TIP_OVER_COS = 0.5      # cos(angle from upright) below which we count as tipped
GOAL_HOLD_STEPS = 5

# Wheel body-frame positions (x_right, y_forward), derived from rover.xml.
# `WHEELBASE` is the front-axle → rear-axle distance, used by the Ackermann
# differential math in `step()`.
# Order matches the `drive_*` actuator order set up in `__init__`:
#   0: right_front, 1: right_middle, 2: right_rear,
#   3: left_front,  4: left_middle,  5: left_rear.
WHEEL_POS_BODY: tuple[tuple[float, float], ...] = (
    (+0.85, +0.65),  # right_front
    (+0.95, -0.05),  # right_middle (slightly more outboard than corners)
    (+0.85, -0.75),  # right_rear
    (-0.85, +0.65),  # left_front
    (-0.95, -0.05),  # left_middle
    (-0.85, -0.75),  # left_rear
)
WHEELBASE = 1.40
REAR_AXLE_Y = -0.75  # body-frame y of the rear wheels (origin of the turn radius)


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
        # Progress-shaping mode:
        #   "delta" — reward 5.0*(prev_dist - dist) every step (symmetric).
        #             Dense, but PENALIZES any step that moves away from the
        #             target — including the lateral detour needed to get
        #             around a blocking obstacle. That local disincentive is
        #             why PPO refuses to route around obstacles and rams them.
        #   "best"  — reward 5.0*max(0, best_dist_so_far - dist): only NEW
        #             closest-approach progress is paid. Moving away/sideways
        #             during a detour costs nothing (no negative reward), so
        #             maneuvering around an obstacle is free; the reward
        #             arrives once the rover clears it and gets closer than
        #             ever. Cannot be reward-farmed (returning to an already-
        #             achieved distance pays 0). Removes the ram incentive but
        #             gives NO guidance around an obstacle (no gradient when
        #             blocked), so the rover wanders when a detour is required.
        #   "geodesic" — same "best" monotone-progress logic, but distance is
        #             the OBSTACLE-AWARE geodesic distance from a NavField
        #             (shortest collision-free path length to the target),
        #             recomputed per episode and per waypoint. Its gradient
        #             routes AROUND blocking obstacles, so the reward actively
        #             guides the detour instead of merely not punishing it. On
        #             obstacle-free terrain it equals Euclidean distance, so
        #             open-terrain behaviour matches "best"/"delta".
        progress_reward_mode: str = "geodesic",
        nav_field_res: float = 0.3,   # NavField grid cell size in metres
        goal_bonus: float = 50.0,
        # Collision penalties softened from (3.0/step + 10 hit) → (1.5/step
        # + 5 hit). Previous values made "freeze near start" a strictly
        # better local optimum than "move and risk a graze" — see the
        # phase-5/8 reports where every episode terminated at exactly
        # `stuck_window_steps` with the rover wiggling in place. Halved
        # collision deterrents combined with a much larger stuck-no-
        # progress penalty (below) reverse the gradient: any progress at
        # all dominates freezing.
        # collision_penalty disabled by default (was 1.5/step). The one-shot
        # hit_penalty is enough deterrent on its own, and the per-step
        # variant historically created a "freeze near start" local optimum
        # (any movement risked a graze worth -1.5/step, while staying still
        # paid -0.01/step until stuck_no_progress fired). Keep the kwarg in
        # place so existing callers and tests still work; set > 0 to
        # restore the old behaviour.
        collision_penalty: float = 0.0,      # per-step while in contact
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
        # waypoint_bonus halved 20 → 10. The previous value was ~80 steps
        # of typical progress-shaping reward, creating a large discontinuity
        # in the value target that PPO had to model around every waypoint
        # transition. Halving keeps the "checkpoint reached" signal clearly
        # positive but lets per-step progress carry more of the credit
        # assignment.
        waypoint_bonus: float = 10.0,
        # Speed bonus on every checkpoint (waypoint + final goal): adds an extra
        # `speed_bonus_scale * base_bonus * (1 - steps_used / max_steps)`. At
        # step 0 the rover effectively gets DOUBLE the bonus; the bonus decays
        # linearly to 0 if it takes the full episode. This rewards reaching
        # checkpoints quickly without changing the per-step economics.
        # speed_bonus_scale halved 1.0 → 0.5. With the old value reaching the
        # goal at step 0 paid 2× the base goal_bonus, which created strong
        # pressure to rush at the cost of the jerk/collision/wheels-off
        # signals. Step cost already implicitly rewards speed; the time-decay
        # bonus is now a finishing nudge rather than a 2× multiplier.
        speed_bonus_scale: float = 0.5,
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
        # (both action dims swing -1 → +1). Scaled at 0.1 the per-step
        # cost caps at -0.8 in the pathological case — modest but
        # consistent. Earlier 0.2 (with action filter alpha=0.3) over-
        # damped exploration and stalled phase 0 learning; 0.1 is the
        # 2× compromise that still shapes smoothness without freezing
        # the policy at random init.
        action_jerk_scale: float = 0.1,
        # Wheel-grounded penalty: -wheels_off_scale × n_wheels_off per step.
        # 6 wheels off (rover airborne) costs 0.18/step. Discourages
        # aggressive turns that lift the rocker-bogie clear of the ground.
        # Set to 0.0 to disable (e.g. on dunes where some lift is normal).
        wheels_off_scale: float = 0.03,
        # First-order low-pass filter on the action before it reaches the
        # MuJoCo actuators. alpha=1.0 means the filter is a no-op (commanded
        # action goes straight to ctrl). Lower values smooth PPO's per-step
        # noise but also damp early learning — at random init PPO outputs
        # are mean-zero random samples, and a filter averages them to ~0,
        # leaving the rover stationary and gradient-less. The PPO action
        # smoothness signal is already covered by `action_jerk_scale`. Keep
        # the filter machinery in place (set <1.0 if you want to experiment)
        # but default to off after observing it killed exploration in
        # scenario_10's first run with alpha=0.3.
        action_filter_alpha: float = 1.0,
        # Adaptive time budget: when a waypoint is reached, extend `max_steps`
        # by `waypoint_time_bonus_per_metre * dist_to_next_target + waypoint_time_bonus_base`.
        # Lets episodes with widely-spaced targets actually finish without
        # tuning a single huge max_steps for every terrain.
        # At top speed (~0.6 m/s, 0.025 s/step) the rover needs ~67 steps/m.
        # `40 steps/m + 200 base` ≈ 1.5× wall-clock budget per segment — slack
        # but not endless.
        waypoint_time_bonus_per_metre: float = 40.0,
        waypoint_time_bonus_base: int = 200,
        # Lidar observation (opt-in, default OFF). When True, the 5 MJCF
        # forward-fan rangefinder sensors (±60°, ±30°, 0° at 0.30 m height)
        # are appended to the observation as normalized clearances in [0, 1]
        # (1 = clear/far, small = obstacle close). This is the standard
        # egocentric range representation that makes obstacle avoidance
        # readily learnable — the abstract 8×AABB obstacle slots alone proved
        # too hard for the policy to route around blocking obstacles. Off by
        # default so open/terrain tasks and the scenario_14 CL benchmark keep
        # their 44-D obs; turn on for obstacle-navigation tasks.
        use_lidar: bool = False,
        lidar_max_range: float = 8.0,
        # Geodesic-heading observation (opt-in, default OFF). Appends 2 dims:
        # the body-frame unit vector along the shortest COLLISION-FREE path to
        # the current target (the NavField gradient). The default target obs
        # (rel_fwd/rel_right) points in a straight line — through blocking
        # obstacles — so the policy is told "goal ahead" while the geodesic
        # reward pays it to go around: a contradictory signal it can't resolve.
        # This exposes the detour direction directly, reducing obstacle
        # avoidance to "drive toward this bearing" (a skill locomotion already
        # has). Requires progress_reward_mode="geodesic" (the field is built
        # only then); falls back to the straight-line bearing otherwise.
        geo_heading_obs: bool = False,
        # Pure-pursuit lookahead for the bent bearing: 0 = local gradient
        # (bends late, gets clipped); >0 = point the bearing this many grid
        # cells ahead along the geodesic so the rover turns EARLY around
        # obstacles. At res 0.3 m, 8 cells ≈ 2.4 m lookahead.
        geo_lookahead_cells: int = 0,
        # NavField obstacle inflation (metres) for BOTH the geodesic reward and
        # the bent-bearing heading. Larger than the true footprint routes the
        # planned path WIDER around obstacles, giving the Ackermann rover
        # (turn radius ~2.75 m) margin so its imperfect arc-following doesn't
        # clip the obstacle — the dominant failure mode. Defaults to the
        # footprint radius (tight routing).
        nav_field_inflate: float | None = None,
        # Skid-steer yaw assist (opt-in, default 0 = pure Ackermann). Adds a
        # left/right wheel-speed differential ∝ steer so the rover turns
        # tighter than its ~2.75 m Ackermann radius — needed to follow the
        # routed geodesic bearing around obstacles without clipping them
        # (the dominant obstacle-episode failure).
        skid_gain: float = 0.0,
        # Control interface:
        #   "ackermann" (default, legacy) — action = (throttle, steer), a
        #     car-like model with a single mirrored steer across the 4 corner
        #     knuckles. Limited to a ~2.75 m turn radius; cannot point-turn.
        #   "vw" — action = (forward_velocity, yaw_rate); a proper rover
        #     mobility controller sets EACH corner knuckle's steer angle and
        #     EACH wheel's speed from the instantaneous-centre-of-rotation
        #     geometry (independent explicit steering, as on Curiosity). Gives
        #     the policy direct yaw-rate control incl. point-turns (v≈0, ω≠0),
        #     so it can follow the routed geodesic bearing around obstacles
        #     instead of arcing wide and clipping them.
        control_mode: str = "ackermann",
        # Clearance-speed penalty (opt-in, default 0). Penalises FORWARD SPEED
        # in proportion to obstacle proximity: `scale · |v_fwd| · max(0, 1 −
        # d/ safe)`. Unlike the proximity penalty (which punishes BEING near an
        # obstacle → the policy freezes away from it), this punishes only
        # moving FAST near one, so the rover learns to SLOW DOWN and clear
        # obstacles carefully instead of clipping them at speed — the dominant
        # obstacle-episode failure — while slow motion nearby stays free (no
        # freeze). Uses `_min_obstacle_dist` (m).
        clearance_speed_penalty_scale: float = 0.0,
        clearance_safe_dist: float = 2.0,
        seed: int = 0,
        render_mode: str | None = None,
    ):
        if isinstance(terrain, str):
            terrain = get_terrain(terrain, seed=seed)
        self.terrain = terrain
        self.max_steps = max_steps
        self.control_decimation = control_decimation
        self.progress_reward_scale = progress_reward_scale
        self.progress_reward_mode = str(progress_reward_mode)
        self.nav_field_res = float(nav_field_res)
        self.nav_field_inflate = (float(nav_field_inflate)
                                  if nav_field_inflate is not None
                                  else ROVER_FOOTPRINT_RADIUS)
        self._nav_field: NavField | None = None
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
        self.action_filter_alpha = float(np.clip(action_filter_alpha, 0.0, 1.0))
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
        # Cache obstacle geom ids up-front. `_apply_terrain_roll` runs on
        # every reset (~thousands per phase) and previously called
        # mj_name2id() per obstacle per reset. The names are stable for
        # the lifetime of a compiled model, so look them up once here.
        self._obstacle_geom_ids: list[int] = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, f"obs_{i}")
            for i in range(len(terrain.obstacles))
        ]

        # Cache the 4 corner-steering actuator IDs (env applies a single Ackermann
        # steer command across them: FR & FL get -steer, RR & RL get +steer —
        # same convention as scripts/visualize_rover.py::drive_ackermann).
        self._steer_actuator_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in ["steer_right_front", "steer_right_rear",
                      "steer_left_front",  "steer_left_rear"]
        ]

        # Lidar observation, opt-in. Implemented as our own `mj_ray` fan
        # rather than the MJCF `rangefinder` sensors, which are unusable for
        # obstacle sensing: their sites sit at z≈1.05 m (above the ~1.0 m
        # obstacle tops, so rays skim over) and self-hit the mast. We instead
        # cast N rays in a forward fan at obstacle mid-height, masked to a
        # dedicated geom group that contains ONLY the obstacle geoms — so a
        # ray can never hit the rover, ground, or heightmap, and returns the
        # clean distance to the nearest obstacle along each bearing. This is
        # the standard egocentric range scan that makes obstacle avoidance
        # learnable.
        # geo_heading_obs BENDS the target-bearing obs (obs[0:2]) along the
        # NavField geodesic instead of adding dims — no obs-dim change, so it
        # composes with the CL benchmark, and in open space it is identical to
        # the straight-line bearing (heading == straight → locomotion obs
        # unchanged). Only bends when an obstacle actually routes the path.
        self.skid_gain = float(skid_gain)
        self.control_mode = str(control_mode)
        if self.control_mode == "vw":
            # Widen the drive actuators' ctrlrange so the vw controller's
            # higher wheel-speed commands (VW_MAX_WHEEL_VEL) actually execute
            # instead of being clamped to the ±3.0 Ackermann default.
            for _n in ["drive_right_front", "drive_right_middle", "drive_right_rear",
                       "drive_left_front", "drive_left_middle", "drive_left_rear"]:
                _aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, _n)
                if _aid >= 0:
                    self._model.actuator_ctrlrange[_aid] = [-VW_MAX_WHEEL_VEL, VW_MAX_WHEEL_VEL]
        self.clearance_speed_penalty_scale = float(clearance_speed_penalty_scale)
        self.clearance_safe_dist = float(clearance_safe_dist)
        self.geo_heading_obs = bool(geo_heading_obs)
        self.geo_lookahead_cells = int(geo_lookahead_cells)
        self._n_geo_heading = 0
        self.use_lidar = bool(use_lidar)
        self.lidar_max_range = float(lidar_max_range)
        self._n_lidar = N_LIDAR_RAYS if self.use_lidar else 0
        if self.use_lidar:
            # Reassign obstacle geoms to a private raycast group (geom_group
            # only affects visualization / ray filtering, NOT collision, which
            # is governed by contype/conaffinity). Rays mask to this group.
            self._lidar_group_mask = np.zeros(6, dtype=np.uint8)
            self._lidar_group_mask[LIDAR_OBSTACLE_GROUP] = 1
            for gid in self._obstacle_geom_ids:
                if gid >= 0:
                    self._model.geom_group[gid] = LIDAR_OBSTACLE_GROUP
            # Fan angles (body-frame bearings), evenly spaced over the arc.
            self._lidar_angles = np.linspace(
                -LIDAR_HALF_ARC, LIDAR_HALF_ARC, N_LIDAR_RAYS, dtype=np.float64
            )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # obs layout (44 dims):
        #   [0..6)   pose+velocity vs current target (6 dims)
        #   [6..8)   NEXT target rel_fwd, rel_right (lookahead — sentinel
        #            (0, 0) if there's no next target, i.e. current target
        #            IS the goal). Lets the policy plan its trajectory
        #            across waypoint boundaries instead of swinging wide at
        #            every transition.
        #   [8..40)  K obstacles × (fwd_min, fwd_max, right_min, right_max)
        #   [40..42) tilt_fwd, tilt_right (body-frame world-up projection):
        #            nose pitch and side roll. Both ~0 upright, range [-1, 1].
        #   [42..44) prev_action (throttle_{t-1}, steer_{t-1}) — lets the
        #            policy smoothly continue commands instead of fighting
        #            the action-jerk penalty.
        #   [44..44+n_lidar) OPTIONAL forward-fan lidar clearances in [0, 1]
        #            (1 = clear, small = obstacle close), only when use_lidar.
        obs_dim = (
            6 + 2 + K_OBSTACLES * OBSTACLE_FEATURES_PER_SLOT
            + N_TILT_FEATURES + N_PREV_ACTION_FEATURES
            + self._n_geo_heading + self._n_lidar
        )
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
        # Read obstacle geometry from the LIVE MuJoCo model (data.geom_xpos +
        # model.geom_size), NOT from self.terrain.obstacles. On randomized
        # terrains `_apply_terrain_roll` re-rolls obstacle positions/sizes into
        # the model (geom_pos / mocap_pos) but does NOT write them back to the
        # TerrainSpec — so reading the spec showed the policy phantom obstacles
        # at their compile-time positions (origin / HIDE_Z) while the real ones
        # were invisible. That silently blinded the policy to every obstacle on
        # every RC_/RT_ terrain. geom_xpos is the ground truth for both static
        # and mocap obstacles (mj_forward/mj_step keep it current) and matches
        # what collision detection and the trajectory plotter already use.
        for gid in self._obstacle_geom_ids:
            if gid < 0:
                continue
            wpos = self._data.geom_xpos[gid]
            cz = float(wpos[2])
            if cz < -10.0:   # hidden slot parked below the floor
                continue
            cx, cy = float(wpos[0]), float(wpos[1])
            gsize = self._model.geom_size[gid]
            sx, sy = float(gsize[0]), float(gsize[1])
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

    def _cast_lidar(self, pos_xy: np.ndarray, yaw: float) -> np.ndarray:
        """Cast the forward-fan ray scan; return per-ray clearances in [0, 1]
        (1 = clear to max range, small = obstacle close). Rays are masked to
        the obstacle-only geom group, so they never hit the rover/ground."""
        base_z = float(self._sensor(self._base_pos_id, 3)[2])
        origin = np.array([float(pos_xy[0]), float(pos_xy[1]),
                           base_z - LIDAR_HEIGHT_OFFSET], dtype=np.float64)
        # body-forward world vector = (-sin yaw, cos yaw); body-right = (cos, sin)
        s, c = np.sin(yaw), np.cos(yaw)
        out = np.empty(self._n_lidar, dtype=np.float32)
        gid = np.array([-1], dtype=np.int32)
        for k, a in enumerate(self._lidar_angles):
            # bearing a measured from forward, positive = toward rover-right
            ca, sa = np.cos(a), np.sin(a)
            # forward rotated by a in the body plane, expressed in world:
            fx, fy = -s, c            # forward
            rx, ry = c, s             # right
            dvec = np.array([fx * ca + rx * sa, fy * ca + ry * sa, 0.0],
                            dtype=np.float64)
            dist = mujoco.mj_ray(self._model, self._data, origin, dvec,
                                 self._lidar_group_mask, 1, -1, gid)
            if dist < 0.0:
                out[k] = 1.0
            else:
                out[k] = min(float(dist), self.lidar_max_range) / self.lidar_max_range
        return out

    def _rebuild_nav_field(self, target_xy) -> None:
        """(Re)build the obstacle-aware NavField for the given target, reading
        live obstacle geometry from the model. Called on reset and on each
        waypoint advance (obstacles are static within an episode)."""
        obstacles_xywh: list[tuple[float, float, float, float]] = []
        for gid in self._obstacle_geom_ids:
            if gid < 0:
                continue
            wpos = self._data.geom_xpos[gid]
            if float(wpos[2]) < -10.0:   # hidden slot
                continue
            gsize = self._model.geom_size[gid]
            obstacles_xywh.append((float(wpos[0]), float(wpos[1]),
                                   float(gsize[0]), float(gsize[1])))
        self._nav_field = NavField(
            half_extent=float(self.terrain.arena_half_extent),
            res=self.nav_field_res,
            obstacles_xywh=obstacles_xywh,
            target_xy=(float(target_xy[0]), float(target_xy[1])),
            inflate=self.nav_field_inflate,
        )

    def _target_distance(self, pos_xy, euclid: float) -> float:
        """Distance used by progress shaping. Geodesic (obstacle-aware) in
        'geodesic' mode when the field has a finite value at the rover cell;
        otherwise Euclidean fallback."""
        if self.progress_reward_mode == "geodesic" and self._nav_field is not None:
            g = self._nav_field.distance(float(pos_xy[0]), float(pos_xy[1]))
            if g == g:  # not NaN
                return g
        return euclid

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

        # Optionally bend the target bearing along the obstacle-aware geodesic:
        # keep the (Euclidean) distance magnitude but point the direction the
        # way the collision-free path leaves the current cell. In open space
        # the geodesic heading equals the straight-line bearing, so this is a
        # no-op there (locomotion / obstacle-free tasks unchanged); it only
        # rotates the bearing when an obstacle actually routes the path.
        if self.geo_heading_obs and self._nav_field is not None:
            if self.geo_lookahead_cells > 0:
                gh = self._nav_field.lookahead_heading(
                    float(pos_xy[0]), float(pos_xy[1]), self.geo_lookahead_cells)
            else:
                gh = self._nav_field.heading(float(pos_xy[0]), float(pos_xy[1]))
            if gh is not None:
                wx, wy = gh
                gfwd = -s * wx + c * wy
                gright = c * wx + s * wy
                mag = float(np.hypot(rel_fwd, rel_right))
                rel_fwd, rel_right = gfwd * mag, gright * mag
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

        # Body-frame tilt: project the world-up vector into the rover's body
        # frame. Using the inverse rotation matrix's third column gives the
        # body-frame components of world-up. Signed so:
        #   tilt_fwd  > 0 → nose UP (body +Y tilts toward world up)
        #   tilt_right > 0 → leaning RIGHT (body +X tilts toward world up)
        # Both are sin(angle); range [-1, 1].
        quat = self._sensor(self._base_quat_id, 4)
        w, x, y, z = (float(quat[0]), float(quat[1]),
                      float(quat[2]), float(quat[3]))
        tilt_right = -2.0 * (x * z - w * y)
        tilt_fwd = 2.0 * (y * z + w * x)

        base_dim = (6 + 2 + K_OBSTACLES * OBSTACLE_FEATURES_PER_SLOT
                    + N_TILT_FEATURES + N_PREV_ACTION_FEATURES)
        obs = np.empty(base_dim + self._n_geo_heading + self._n_lidar,
                       dtype=np.float32)
        obs[0] = rel_fwd
        obs[1] = rel_right
        obs[2] = heading_to_goal
        obs[3] = float(linvel[0])
        obs[4] = float(linvel[1])
        obs[5] = angvel_z
        obs[6] = float(rel_fwd_next)
        obs[7] = float(rel_right_next)
        obs_tail = 8 + K_OBSTACLES * OBSTACLE_FEATURES_PER_SLOT
        obs[8:obs_tail] = obstacle_feats
        obs[obs_tail + 0] = tilt_fwd
        obs[obs_tail + 1] = tilt_right
        # prev_action is updated at the END of step(); on the first obs of
        # an episode it's the zero vector set in reset().
        obs[obs_tail + 2] = float(self._prev_action[0])
        obs[obs_tail + 3] = float(self._prev_action[1])
        # Fixed normalization of the base features — see _build_obs_scale.
        obs[:base_dim] /= OBS_SCALE
        # Lidar clearances (already in [0, 1]) appended last.
        if self._n_lidar:
            obs[base_dim:] = self._cast_lidar(pos_xy, yaw)
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
            gid = self._obstacle_geom_ids[i] if i < len(self._obstacle_geom_ids) else -1
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
        # Build the obstacle-aware navigation field for the first target (only
        # in geodesic mode; cheap Dijkstra on the arena grid). Obstacles are
        # static within an episode, so this is rebuilt only here and on each
        # waypoint advance.
        if self.progress_reward_mode == "geodesic":
            self._rebuild_nav_field(target)
        # Every-step closest-approach record for "best"/"geodesic" progress
        # shaping (see step()); distinct from the 0.5 m-quantised
        # _best_d_target above. Seeded with the shaping distance (geodesic
        # when available) so the first step measures progress correctly.
        self._reward_best_dist = self._target_distance(pos_xy, self._prev_dist)
        self._steps_since_progress = 0
        # Action smoothness: previous step's action, used by the jerk penalty.
        # First step has no prior action; init to zero (no penalty on step 1).
        self._prev_action = np.zeros(2, dtype=np.float64)
        # First-order action filter state. The filter outputs what the
        # actuators actually see — `_filtered_action` lags behind `action`,
        # smoothing PPO's step-to-step noise.
        self._filtered_action = np.zeros(2, dtype=np.float64)
        obs = self._build_obs()
        return obs, {}

    # ---------------------------------------------------------------- control

    def _apply_ackermann_control(self, throttle: float, steer: float) -> None:
        """Legacy car-like control: single mirrored steer across the 4 corner
        knuckles + per-wheel Ackermann speed differential (+ optional skid)."""
        v_base = -throttle * MAX_WHEEL_VEL  # ctrl sign: positive rolls -Y
        steer_rad = steer * MAX_STEER_RAD
        if abs(steer_rad) < 1e-3:
            wheel_cmd = np.full(6, v_base)
        else:
            R = WHEELBASE / np.tan(steer_rad)
            R_abs = abs(R)
            mults = np.empty(6)
            for i, (xw, yw) in enumerate(WHEEL_POS_BODY):
                mults[i] = np.hypot(xw - R, yw - REAR_AXLE_Y) / R_abs
            max_mult = float(mults.max())
            if max_mult > 1.0:
                mults /= max_mult
            wheel_cmd = v_base * mults
        if self.skid_gain > 0.0:
            dv = steer * self.skid_gain * MAX_WHEEL_VEL
            wheel_cmd[0:3] += dv
            wheel_cmd[3:6] -= dv
            wheel_cmd = np.clip(wheel_cmd, -MAX_WHEEL_VEL, MAX_WHEEL_VEL)
        self._data.ctrl[0:6] = wheel_cmd
        # 4-wheel counter-steer: fronts -steer, rears +steer. steer>0 = right.
        self._data.ctrl[6] = -steer_rad  # right front
        self._data.ctrl[7] = +steer_rad  # right rear
        self._data.ctrl[8] = -steer_rad  # left front
        self._data.ctrl[9] = +steer_rad  # left rear

    def _apply_vw_control(self, v: float, omega: float) -> None:
        """Rover mobility controller: realise body velocity (v forward, omega
        yaw) by setting EACH corner knuckle's steer angle and EACH wheel's
        speed from the rigid-body wheel velocities (independent explicit
        steering — Curiosity-style). Enables point-turns (v≈0, omega≠0).

        Wheel ground velocity in body frame (right, fwd):
            v_i = (-omega*y_i,  v + omega*x_i)
        Corner wheels steer to that heading and drive at its magnitude;
        headings past ±90° are folded by reversing the wheel. The two middle
        wheels can't steer (no actuator) → they roll at the forward component
        and scrub the small lateral part (physical for a fixed mid-wheel).
        """
        r = WHEEL_RADIUS
        wheel_ang = np.empty(6)
        steer_ang = np.zeros(6)
        for i, (xi, yi) in enumerate(WHEEL_POS_BODY):
            vr = -omega * yi
            vf = v + omega * xi
            is_middle = (i == 1 or i == 4)
            if is_middle:
                wheel_ang[i] = vf / r          # forward component only
                continue
            spd = float(np.hypot(vr, vf))
            ang = float(np.arctan2(vr, vf))    # heading rel forward, +=right
            if ang > np.pi / 2:
                ang -= np.pi; spd = -spd
            elif ang < -np.pi / 2:
                ang += np.pi; spd = -spd
            wheel_ang[i] = spd / r
            steer_ang[i] = ang
        # Normalise wheel speeds to the (widened) vw wheel-speed cap. Using the
        # higher VW_MAX_WHEEL_VEL headroom means a turn no longer scales the
        # forward speed down until ω is large — fixing the arc-stall.
        mx = float(np.max(np.abs(wheel_ang)))
        if mx > VW_MAX_WHEEL_VEL:
            wheel_ang *= VW_MAX_WHEEL_VEL / mx
        # Drive ctrl: positive ctrl rolls -Y, so forward motion is negative.
        self._data.ctrl[0:6] = -wheel_ang
        # Steer ctrl: joint sign is ctrl = -heading (matches the legacy
        # convention where a right heading came from a negative front ctrl).
        steer_ang = np.clip(steer_ang, -MAX_STEER_JOINT, MAX_STEER_JOINT)
        self._data.ctrl[6] = -steer_ang[0]  # right front
        self._data.ctrl[7] = -steer_ang[2]  # right rear
        self._data.ctrl[8] = -steer_ang[3]  # left front
        self._data.ctrl[9] = -steer_ang[5]  # left rear

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # `action` is what PPO commanded this step; `_filtered_action` is
        # what the actuators actually see after the low-pass filter. Jerk
        # penalty is computed on the RAW command (so the policy is taught
        # to issue smooth commands), not on the filtered value — penalising
        # the filtered signal would be a tautology, since the filter is
        # designed to suppress jerk.
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        action_jerk_sq = float(np.sum((action - self._prev_action) ** 2))
        action_jerk_penalty = self.action_jerk_scale * action_jerk_sq

        # First-order LPF: filtered_t = alpha · cmd + (1-alpha) · filtered_{t-1}.
        # Tests + the visualiser use alpha=0.3 by default.
        a = self.action_filter_alpha
        self._filtered_action = a * action + (1.0 - a) * self._filtered_action
        throttle = float(self._filtered_action[0])
        steer    = float(self._filtered_action[1])

        # Map the (filtered) action to actuator commands via the selected
        # control mode (see `_apply_vw_control` / `_apply_ackermann_control`).
        if self.control_mode == "vw":
            # (v, ω) rover mobility controller sets ctrl[0:10] directly.
            # FORWARD-ONLY linear velocity: action[0]∈[-1,1] → v∈[0, V_MAX]
            # (a=-1 → v=0 point-turn, a=+1 → full speed). Reverse is disabled —
            # it only enabled degenerate wedge/back-out behaviours and is the
            # standard choice in mapless-nav RL (linear velocity sigmoid-gated
            # to (0,1)). Turning still spans both directions via ω.
            v_fwd = 0.5 * (throttle + 1.0) * V_MAX
            self._apply_vw_control(v_fwd, steer * OMEGA_MAX)
        else:
            self._apply_ackermann_control(throttle, steer)
        # Hold the arm stowed every step so position actuators don't drift.
        for i, v in enumerate(ARM_STOW_CTRL):
            self._data.ctrl[10 + i] = v

        for _ in range(self.control_decimation):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        pos_xy, _ = self._base_pose_xy()
        target = self._current_target()
        dist = float(np.linalg.norm(target - pos_xy))

        if self.progress_reward_mode in ("best", "geodesic"):
            # Reward only NEW closest-approach progress. `_reward_best_dist`
            # holds the smallest shaping-distance seen so far, updated EVERY
            # step by any amount (distinct from the 0.5 m-quantised
            # `_best_d_target` used by the stuck guard). In "geodesic" mode the
            # shaping distance is the obstacle-aware NavField distance, whose
            # decrease guides the rover AROUND obstacles; in "best" mode it is
            # Euclidean. progress = the amount by which this step beats the
            # record; steps that don't improve it pay 0 (not a negative delta),
            # so a detour is free, and returning to an already-achieved
            # distance pays 0 so it can't be reward-farmed.
            shape_dist = self._target_distance(pos_xy, dist)
            progress = max(0.0, self._reward_best_dist - shape_dist)
            self._reward_best_dist = min(self._reward_best_dist, shape_dist)
        else:
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
                # Rebuild the nav field for the new target (obstacles unchanged
                # within the episode, only the source moves) and reset the
                # closest-approach record — otherwise the stale small record
                # from the just-reached waypoint would suppress all reward
                # toward the next one.
                if self.progress_reward_mode == "geodesic":
                    self._rebuild_nav_field(next_target)
                self._reward_best_dist = self._target_distance(pos_xy, dist_to_next)
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

        # Update _prev_action BEFORE building obs so the obs at step t carries
        # the action that was just applied (action_{t-1} from the policy's
        # next-decision perspective). Jerk was already computed above against
        # the old _prev_action, so the order is consistent: jerk uses
        # (a_now - a_prev), obs publishes a_now.
        self._prev_action = action.copy()
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

        # Clearance-speed penalty: penalise moving fast near an obstacle so the
        # rover slows to clear it instead of clipping at speed (see __init__).
        clearance_speed_penalty = 0.0
        if (self.clearance_speed_penalty_scale > 0
                and self.clearance_safe_dist > 0
                and self._min_obstacle_dist < self.clearance_safe_dist):
            lv = self._sensor(self._base_linvel_id, 3)
            spd = float(np.hypot(lv[0], lv[1]))   # planar speed magnitude
            proximity = 1.0 - self._min_obstacle_dist / self.clearance_safe_dist
            clearance_speed_penalty = (
                self.clearance_speed_penalty_scale * spd * proximity
            )

        reward = (
            self.progress_reward_scale * progress
            - self.step_cost
            - proximity_penalty
            - clearance_speed_penalty
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

        info: dict[str, Any] = {
            "distance_to_goal": dist_to_goal,
            "distance_to_target": dist,
            "waypoint_index": self._wp_idx,
            "goal_hold": int(self._goal_hold),
            "is_success": bool(success),
            "collision": bool(collision),
            "tipped": bool(tipped),
            "stuck_in_collision": bool(stuck_in_collision),
            "stuck_no_progress": bool(stuck_no_progress),
            "pos_xy": (float(pos_xy[0]), float(pos_xy[1])),
            "yaw": float(yaw_now),
            "n_wheels_off_ground": int(n_wheels_off),
            # Per-step reward decomposition. Positive numbers add to the
            # step reward, negative subtract. Sum equals `reward` for this
            # step. Used by the viewer to surface "why did the rover stop
            # before the goal?" without re-running the env in debug mode.
            "reward_terms": {
                "progress": float(self.progress_reward_scale * progress),
                "step_cost": float(-self.step_cost),
                "proximity": float(-proximity_penalty),
                "collision": float(-(self.collision_penalty if collision else 0.0)),
                "hit": float(-(self.hit_penalty if new_hit else 0.0)),
                "tipped": float(-(self.tipped_penalty if tipped else 0.0)),
                "stuck_in_collision": float(-(self.stuck_in_collision_penalty
                                              if stuck_in_collision and not success else 0.0)),
                "stuck_no_progress": float(-(self.stuck_no_progress_penalty
                                             if stuck_no_progress and not success else 0.0)),
                "action_jerk": float(-action_jerk_penalty),
                "wheels_off": float(-wheels_off_penalty),
                "waypoint_bonus": float(waypoint_reached_bonus),
                "speed_bonus": float(speed_bonus),
                "goal_bonus": float(self.goal_bonus if success else 0.0),
            },
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


