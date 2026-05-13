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
) -> tuple[tuple[float, float], float, tuple[float, float]]:
    """Pick (start, yaw, goal) with goal at least `min_separation` from start.

    Start yaw is uniform in [-π, π]; the policy must learn that "goal ahead"
    is rel_fwd > 0 regardless of starting orientation.

    `margin` keeps both points away from the arena boundary.
    """
    bound = arena_half - margin
    for _ in range(50):
        start = _sample_point_in_box(rng, (-bound, bound), (-bound, bound))
        goal = _sample_point_in_box(rng, (-bound, bound), (-bound, bound))
        dx, dy = goal[0] - start[0], goal[1] - start[1]
        if (dx * dx + dy * dy) ** 0.5 >= min_separation:
            yaw = float(rng.uniform(-np.pi, np.pi))
            return start, yaw, goal
    # Degenerate fallback if 50 samples failed (shouldn't happen for sane bounds).
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


def sample_obstacles_along_path(
    rng: np.random.Generator,
    start: tuple[float, float],
    goal: tuple[float, float],
    n_obstacles: int,
    max_slots: int,
    size_range: tuple[float, float],
    lateral_jitter: float,
    along_jitter_fraction: float = 0.7,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Place `n_obstacles` random boxes roughly between start and goal.

    Each obstacle is positioned at a random fraction along the start-goal
    line (within `along_jitter_fraction * length` of the midpoint), with a
    random perpendicular offset and a random size in `size_range`.

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
        positions.append((along_x + perp * lx, along_y + perp * ly, s))
        sizes.append((s, s, s))

    # Pad with hidden slots.
    for _ in range(max_slots - n_obstacles):
        positions.append((0.0, 0.0, HIDE_Z))
        sizes.append((0.1, 0.1, 0.1))

    return positions, sizes
