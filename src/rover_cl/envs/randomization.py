"""Domain-randomization helpers for terrain factories.

A randomized terrain re-rolls its obstacle positions, start/goal poses, and
heightmap shape every time `env.reset()` is called. This is the standard
"domain randomization" technique used to make policies that don't memorize
fixed layouts — the rover never sees the exact same configuration twice.

Design:

- `TerrainRoll` is a per-episode bundle of fresh values the env applies on
  top of a "template" `TerrainSpec`. The TerrainSpec defines the *invariants*
  (arena size, hfield extent, max obstacle count, etc.); the TerrainRoll
  fills in the *current* episode's values.

- Each randomized terrain factory returns a TerrainSpec with
  `randomize_on_reset` set to a callable `np_random -> TerrainRoll`. The env
  calls it once per reset.

- The TerrainSpec pre-allocates `len(obstacles)` geom slots so MuJoCo has
  somewhere to put each rolled obstacle. To "hide" a slot the randomizer
  returns z=HIDE_Z for that obstacle (below the floor, no contact).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# Z-coordinate used to "park" an obstacle slot below the floor when the
# randomizer wants fewer obstacles than the spec allocated.
HIDE_Z: float = -50.0


def heightmap_height_at_xy(
    heightmap: np.ndarray | None,
    extent: tuple[float, float, float],
    x: float,
    y: float,
) -> float:
    """Return the world-z elevation of the heightmap at (x, y).

    Bilinear-samples the heightmap (rows = y, cols = x) and scales by the
    extent's elevation_z. Returns 0.0 when `heightmap is None` (i.e. the
    terrain is a flat plane). Out-of-bounds (x, y) clamp to the edge cell.

    The heightmap is stored normalized to [0, 1]; the world elevation is
    `heightmap[i, j] * elevation_z`. The (i, j) → (x, y) mapping matches
    MuJoCo's hfield: row index increases with y, column with x, and the
    field spans (-half, +half) in both dimensions where half is `extent[0]`
    / `extent[1]`.
    """
    if heightmap is None:
        return 0.0
    hx, hy, ez = extent
    nrow, ncol = heightmap.shape
    if ez == 0.0 or nrow == 0 or ncol == 0:
        return 0.0
    # Normalise (x, y) into row/col index space.
    fx = (x + hx) / (2 * hx) * (ncol - 1)
    fy = (y + hy) / (2 * hy) * (nrow - 1)
    j = int(np.clip(np.floor(fx), 0, ncol - 2))
    i = int(np.clip(np.floor(fy), 0, nrow - 2))
    tx = float(np.clip(fx - j, 0.0, 1.0))
    ty = float(np.clip(fy - i, 0.0, 1.0))
    h00 = float(heightmap[i,     j])
    h01 = float(heightmap[i,     j + 1])
    h10 = float(heightmap[i + 1, j])
    h11 = float(heightmap[i + 1, j + 1])
    h0 = h00 * (1 - tx) + h01 * tx
    h1 = h10 * (1 - tx) + h11 * tx
    h = h0 * (1 - ty) + h1 * ty
    return float(h * ez)


@dataclass
class TerrainRoll:
    """One episode's randomized terrain values.

    Lengths of `obstacle_positions` and `obstacle_sizes` must equal the
    terrain's compiled obstacle count. Use HIDE_Z for slots you want
    inactive this episode.
    """

    start_pos: tuple[float, float]
    start_yaw: float
    goal_pos: tuple[float, float]
    waypoints: tuple[tuple[float, float], ...]
    obstacle_positions: list[tuple[float, float, float]]  # (x, y, z)
    obstacle_sizes: list[tuple[float, float, float]]      # half-extents
    heightmap: np.ndarray | None = None                   # optional fresh hfield


TerrainRandomizer = Callable[[np.random.Generator], TerrainRoll]


# ---------------------------------------------------------------------------
# Sampling helpers — each takes an `np.random.Generator` to keep the env's
# seeded RNG flow intact (`super().reset(seed=...)` reseeds self.np_random
# which we then pass into these).
# ---------------------------------------------------------------------------


def _sample_point_in_box(
    rng: np.random.Generator,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[float, float]:
    return (float(rng.uniform(*x_range)), float(rng.uniform(*y_range)))


def sample_start_goal_pair(
    rng: np.random.Generator,
    arena_half: float,
    min_separation: float,
    margin: float = 1.0,
    *,
    relative_bearing: str = "uniform",
    max_separation: float | None = None,
) -> tuple[tuple[float, float], float, tuple[float, float]]:
    """Pick (start, yaw, goal) with goal at least `min_separation` from start.

    `relative_bearing` controls where the goal sits relative to the rover's
    facing direction (defined by `yaw`):

    - ``"uniform"`` — default; goal direction uniform in the full circle
      (`yaw` then uniform in [-π, π] so absolute orientation also varies).
      The policy must learn that "goal ahead" is `rel_fwd > 0` regardless of
      starting orientation.
    - ``"front"`` — goal is in the front 120° cone (|bearing| ≤ π/3). The
      easy case — exercises forward driving.
    - ``"side"`` — goal lies far to the rover's side (|bearing| ∈ [π/3, 2π/3]).
      Forces the rover to execute a 90°-ish turn before driving forward.
    - ``"behind"`` — goal is in the back 120° cone (|bearing| > 2π/3). Forces
      a near-U-turn manoeuvre at episode start.
    - ``"maneuver"`` — equal mix of side and behind. Used by phases that
      specifically train manoeuvring rather than straight driving.

    `margin` keeps both points away from the arena boundary.

    `max_separation`, when set, clips the random distance to
    ``[min_separation, max_separation]`` instead of the default
    ``[min_separation, min_separation + 4]``. Used by the scenario_11
    distance-curriculum where phase 0's "short" slice wants tight 2-4 m
    goals to give PPO a fast bootstrap signal.
    """
    if relative_bearing not in {"uniform", "front", "side", "behind", "maneuver"}:
        raise ValueError(f"unknown relative_bearing {relative_bearing!r}")

    # Map mode → allowed bearing range (in radians, measured from rover +Y).
    if relative_bearing == "uniform":
        bearing_range: tuple[float, float] | None = None
    elif relative_bearing == "front":
        bearing_range = (-np.pi / 3, np.pi / 3)
    elif relative_bearing == "side":
        # |bearing| ∈ [60°, 120°] on either side. Pick a side, then a
        # magnitude in that band.
        bearing_range = ("side",)  # sentinel; sampled below
    elif relative_bearing == "behind":
        bearing_range = ("behind",)  # |bearing| ∈ [120°, 180°]
    else:  # "maneuver" — 50/50 side or behind
        bearing_range = ("maneuver",)

    bound = arena_half - margin
    for _ in range(80):
        start = _sample_point_in_box(rng, (-bound, bound), (-bound, bound))
        yaw = float(rng.uniform(-np.pi, np.pi))

        # Sample (distance, relative bearing) of the goal in rover frame.
        if bearing_range is None:
            # Original behaviour: goal anywhere in the box, yaw independent.
            # When max_separation is set, also reject samples that exceed it
            # so the distance-curriculum knobs work in this branch too.
            goal = _sample_point_in_box(rng, (-bound, bound), (-bound, bound))
            dx, dy = goal[0] - start[0], goal[1] - start[1]
            d = (dx * dx + dy * dy) ** 0.5
            if d >= min_separation and (max_separation is None or d <= max_separation):
                return start, yaw, goal
            continue

        if bearing_range == ("side",):
            sign = 1.0 if rng.uniform() < 0.5 else -1.0
            rel_bearing = sign * float(rng.uniform(np.pi / 3, 2 * np.pi / 3))
        elif bearing_range == ("behind",):
            sign = 1.0 if rng.uniform() < 0.5 else -1.0
            rel_bearing = sign * float(rng.uniform(2 * np.pi / 3, np.pi))
        elif bearing_range == ("maneuver",):
            sign = 1.0 if rng.uniform() < 0.5 else -1.0
            if rng.uniform() < 0.5:
                rel_bearing = sign * float(rng.uniform(np.pi / 3, 2 * np.pi / 3))
            else:
                rel_bearing = sign * float(rng.uniform(2 * np.pi / 3, np.pi))
        else:
            lo, hi = bearing_range
            rel_bearing = float(rng.uniform(lo, hi))

        # Convert (yaw + rel_bearing) into a world heading from start.
        # Rover's forward is +Y in body frame, so world heading is yaw + 90°
        # (i.e. +Y in body = (-sin(yaw), cos(yaw)) in world). A goal at
        # `rel_bearing=0` should sit straight ahead along that vector.
        world_bearing = yaw + np.pi / 2 + rel_bearing
        # Distance: tight band [min_separation, max_separation] when the
        # caller specified one (distance-curriculum), otherwise the legacy
        # [min_separation, min_separation + 4] range.
        if max_separation is not None:
            max_d = float(rng.uniform(min_separation, max_separation))
        else:
            max_d = min_separation + float(rng.uniform(0.0, 4.0))
        gx = start[0] + max_d * float(np.cos(world_bearing))
        gy = start[1] + max_d * float(np.sin(world_bearing))
        if abs(gx) <= bound and abs(gy) <= bound:
            return start, yaw, (gx, gy)

    # Degenerate fallback if all attempts failed (very tight bounds).
    return (0.0, 0.0), 0.0, (0.0, min_separation)


def sample_waypoints_between(
    rng: np.random.Generator,
    start: tuple[float, float],
    goal: tuple[float, float],
    n_waypoints: int,
    lateral_jitter: float = 1.5,
) -> tuple[tuple[float, float], ...]:
    """Place waypoints roughly along the start->goal line, with lateral jitter.

    Splits the segment into `n_waypoints + 1` parts; each waypoint sits at
    a fraction `(i + 1) / (n_waypoints + 1)` along the line plus a random
    perpendicular offset of up to `lateral_jitter` metres.
    """
    sx, sy = start
    gx, gy = goal
    dx, dy = gx - sx, gy - sy
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return tuple((sx, sy) for _ in range(n_waypoints))
    # Unit forward + left vectors along the line.
    fx, fy = dx / length, dy / length
    lx, ly = -fy, fx
    out: list[tuple[float, float]] = []
    for i in range(n_waypoints):
        t = (i + 1) / (n_waypoints + 1)
        along = (sx + t * dx, sy + t * dy)
        perp = float(rng.uniform(-lateral_jitter, lateral_jitter))
        out.append((along[0] + perp * lx, along[1] + perp * ly))
    return tuple(out)


def sample_twisty_waypoints(
    rng: np.random.Generator,
    start: tuple[float, float],
    goal: tuple[float, float],
    n_waypoints: int,
    arena_half: float,
    margin: float = 1.5,
    *,
    swing_min: float = 2.5,
    swing_max: float = 5.0,
    alternate_sides: bool = True,
) -> tuple[tuple[float, float], ...]:
    """Place waypoints along a snaking path with real twists between them.

    Unlike `sample_waypoints_between` (which jitters perpendicular to the
    straight start→goal line), this builds a snake: each waypoint sits at
    successive fractions along the line, but its perpendicular offset
    *alternates sign* and varies in magnitude, producing a left-right-left
    zig-zag route the rover must steer through. With many waypoints the
    route ends up looking like a sinusoidal weave.

    Parameters
    ----------
    arena_half / margin
        Used to clamp the generated waypoints inside the arena so we don't
        spawn one outside the playable region.
    swing_min / swing_max
        Lateral magnitude range (m). Larger ⇒ tighter, sharper turns.
    alternate_sides
        When True (default) sides strictly alternate (deterministic zig-zag
        skeleton, magnitudes randomized). When False each waypoint flips
        sides with probability 0.5, producing a freer random walk.
    """
    sx, sy = start
    gx, gy = goal
    dx, dy = gx - sx, gy - sy
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6 or n_waypoints <= 0:
        return ()
    fx, fy = dx / length, dy / length
    lx, ly = -fy, fx
    bound = arena_half - margin

    # First-leg side: random so we don't always twist the same direction
    # across episodes.
    side = 1.0 if rng.uniform() < 0.5 else -1.0

    out: list[tuple[float, float]] = []
    for i in range(n_waypoints):
        t = (i + 1) / (n_waypoints + 1)
        # Add some along-axis jitter too — keeps the rover from learning a
        # fixed spacing pattern.
        t_jitter = float(rng.uniform(-0.07, 0.07))
        t_eff = float(np.clip(t + t_jitter, 0.05, 0.95))
        along_x = sx + t_eff * dx
        along_y = sy + t_eff * dy
        magnitude = float(rng.uniform(swing_min, swing_max))
        wx = along_x + side * magnitude * lx
        wy = along_y + side * magnitude * ly
        # Clamp into the arena.
        wx = float(np.clip(wx, -bound, bound))
        wy = float(np.clip(wy, -bound, bound))
        out.append((wx, wy))
        if alternate_sides:
            side = -side
        else:
            if rng.uniform() < 0.5:
                side = -side
    return tuple(out)


def sample_arc_waypoints(
    rng: np.random.Generator,
    start: tuple[float, float],
    goal: tuple[float, float],
    n_waypoints: int = 2,
    arc_angle_deg: tuple[float, float] = (30.0, 90.0),
) -> tuple[tuple[float, float], ...]:
    """Place `n_waypoints` along an arc that bulges off the start→goal line.

    The arc's chord runs from start to goal; its angular sweep is drawn
    uniformly from `arc_angle_deg`. The bulge direction (left/right of the
    line) is random per call. Useful for training "lay your steering into a
    curve" — every waypoint sits at the same arc radius, so a single
    well-chosen steer angle traces them all.
    """
    sx, sy = start
    gx, gy = goal
    dx, dy = gx - sx, gy - sy
    chord = (dx * dx + dy * dy) ** 0.5
    if chord < 1e-6 or n_waypoints <= 0:
        return ()
    mx, my = (sx + gx) / 2, (sy + gy) / 2
    # Forward (chord direction) and perpendicular unit vectors.
    fx, fy = dx / chord, dy / chord
    lx, ly = -fy, fx
    # Pick the arc's angular sweep and bulge side.
    sweep = float(np.deg2rad(rng.uniform(*arc_angle_deg)))
    side = 1.0 if rng.uniform() < 0.5 else -1.0
    # Geometry: chord subtends `sweep` at the arc's centre. Radius R from
    # R = (chord/2) / sin(sweep/2). Centre lies on the perpendicular
    # bisector at distance d = (chord/2) / tan(sweep/2) from the chord.
    half_chord = chord / 2.0
    R = half_chord / np.sin(sweep / 2.0)
    d_centre = half_chord / np.tan(sweep / 2.0)
    cx = mx - side * d_centre * lx
    cy = my - side * d_centre * ly
    # Angle from centre to start, to goal.
    a_start = float(np.arctan2(sy - cy, sx - cx))
    # Walk in the side direction so waypoints sit ON the arc between start and goal.
    # Sweep angle from start toward goal is `+sweep * side` in arc-centre frame.
    out: list[tuple[float, float]] = []
    for i in range(n_waypoints):
        t = (i + 1) / (n_waypoints + 1)
        a = a_start + side * t * sweep
        out.append((cx + R * np.cos(a), cy + R * np.sin(a)))
    return tuple(out)


def sample_waypoint_at_bearing(
    rng: np.random.Generator,
    start: tuple[float, float],
    start_yaw: float,
    relative_bearing_range: tuple[float, float],
    distance_range: tuple[float, float],
    arena_half: float,
    margin: float = 1.5,
) -> tuple[float, float]:
    """Sample one waypoint at a target relative bearing in [lo, hi] (radians).

    Used to construct "90° turn", "180° U-turn", etc. terrains where the
    rover must turn through a specific angle to chase the waypoint. The
    bearing is measured in the rover's body frame; sign is randomised so
    left- and right-turn episodes are balanced.
    """
    lo, hi = relative_bearing_range
    sign = 1.0 if rng.uniform() < 0.5 else -1.0
    rel_b = sign * float(rng.uniform(lo, hi))
    # Rover forward in world: +Y after yaw rotation.
    world_b = start_yaw + np.pi / 2 + rel_b
    bound = arena_half - margin
    for _ in range(40):
        d = float(rng.uniform(*distance_range))
        wx = start[0] + d * float(np.cos(world_b))
        wy = start[1] + d * float(np.sin(world_b))
        if abs(wx) <= bound and abs(wy) <= bound:
            return (wx, wy)
    # Fall back to a point in the same direction at the minimum distance.
    d = distance_range[0]
    return (float(np.clip(start[0] + d * np.cos(world_b), -bound, bound)),
            float(np.clip(start[1] + d * np.sin(world_b), -bound, bound)))


def sample_slalom_waypoints(
    rng: np.random.Generator,
    start: tuple[float, float],
    goal: tuple[float, float],
    n_waypoints: int = 5,
    swing_min: float = 1.8,
    swing_max: float = 3.0,
) -> tuple[tuple[float, float], ...]:
    """Strict alternating-side waypoints with deterministic spacing.

    Same idea as `sample_twisty_waypoints` but with fixed alternation
    (always left-right-left-…), uniform spacing, and tighter swing range.
    Used by the slalom terrain — the rover learns a steady S-curve.
    """
    sx, sy = start
    gx, gy = goal
    dx, dy = gx - sx, gy - sy
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6 or n_waypoints <= 0:
        return ()
    fx, fy = dx / length, dy / length
    lx, ly = -fy, fx
    side = 1.0 if rng.uniform() < 0.5 else -1.0
    out: list[tuple[float, float]] = []
    for i in range(n_waypoints):
        t = (i + 1) / (n_waypoints + 1)
        along_x = sx + t * dx
        along_y = sy + t * dy
        magnitude = float(rng.uniform(swing_min, swing_max))
        out.append((along_x + side * magnitude * lx, along_y + side * magnitude * ly))
        side = -side
    return tuple(out)


def sample_obstacles_along_path(
    rng: np.random.Generator,
    start: tuple[float, float],
    goal: tuple[float, float],
    n_obstacles: int,
    max_slots: int,
    size_range: tuple[float, float],
    lateral_jitter: float,
    along_jitter_fraction: float = 0.7,
    heightmap: np.ndarray | None = None,
    heightmap_extent: tuple[float, float, float] = (15.0, 15.0, 0.0),
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Place `n_obstacles` random boxes roughly between start and goal.

    Each obstacle is positioned at a random fraction along the start-goal
    line (within `along_jitter_fraction * length` of the midpoint), with a
    random perpendicular offset and a random size in `size_range`.

    When `heightmap` is given, the obstacle's z is offset so its bottom
    rests on the terrain surface at (x, y). Without this offset, obstacles
    on heightmap terrains end up half-buried (centre at z = half-extent,
    bottom at z = 0 ≪ terrain surface), which: (a) corrupts the visible
    geometry the rover learns to dodge, (b) hides part of the collider so
    the rover tunnels into the buried half before the broad-phase fires.

    Slots beyond `n_obstacles` are filled with the HIDE_Z sentinel so the
    pre-compiled geoms stay inert.
    """
    sx, sy = start
    gx, gy = goal
    dx, dy = gx - sx, gy - sy
    length = (dx * dx + dy * dy) ** 0.5
    fx, fy = (dx / length, dy / length) if length > 1e-6 else (0.0, 1.0)
    lx, ly = -fy, fx

    positions: list[tuple[float, float, float]] = []
    sizes: list[tuple[float, float, float]] = []
    for _ in range(n_obstacles):
        s = float(rng.uniform(*size_range))
        # `t` in [0.5 - jitter, 0.5 + jitter] keeps the obstacles near the
        # middle 70% of the path (jitter=0.35 means [0.15, 0.85]).
        t = 0.5 + float(rng.uniform(-along_jitter_fraction / 2, along_jitter_fraction / 2))
        along_x = sx + t * dx
        along_y = sy + t * dy
        perp = float(rng.uniform(-lateral_jitter, lateral_jitter))
        ox = along_x + perp * lx
        oy = along_y + perp * ly
        # Place box centre at terrain_surface + half_extent so the box bottom
        # rests on the surface. Flat-plane terrains pass `heightmap=None` →
        # surface=0 → centre=s (the original behaviour).
        surface_z = heightmap_height_at_xy(heightmap, heightmap_extent, ox, oy)
        positions.append((ox, oy, surface_z + s))
        sizes.append((s, s, s))

    # Pad with hidden slots.
    for _ in range(max_slots - n_obstacles):
        positions.append((0.0, 0.0, HIDE_Z))
        sizes.append((0.1, 0.1, 0.1))

    return positions, sizes
