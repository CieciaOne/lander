"""Terrain framework.

A terrain is a Python function that returns a `TerrainSpec`. The env composes
the terrain with the rover MJCF into a single MuJoCo XML string.

Adding a new terrain = writing one function that returns a `TerrainSpec`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from rover_cl.envs.randomization import (
    HIDE_Z,
    TerrainRandomizer,
    TerrainRoll,
    sample_arc_waypoints,
    sample_obstacles_along_path,
    sample_slalom_waypoints,
    sample_start_goal_pair,
    sample_twisty_waypoints,
    sample_waypoint_at_bearing,
    sample_waypoints_between,
)

ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"


@dataclass
class Obstacle:
    """A static box obstacle in the arena."""
    pos: tuple[float, float, float]
    size: tuple[float, float, float]  # half-extents
    rgba: tuple[float, float, float, float] = (0.4, 0.35, 0.3, 1.0)

    def to_mjcf(self, name: str) -> str:
        return (
            f'<geom name="{name}" type="box" pos="{self.pos[0]} {self.pos[1]} {self.pos[2]}" '
            f'size="{self.size[0]} {self.size[1]} {self.size[2]}" '
            f'rgba="{self.rgba[0]} {self.rgba[1]} {self.rgba[2]} {self.rgba[3]}" '
            f'contype="1" conaffinity="1" friction="1.0 0.05 0.001"/>'
        )


@dataclass
class TerrainSpec:
    """Declarative description of a terrain."""
    name: str
    arena_half_extent: float                       # plane half-size in X and Y
    obstacles: list[Obstacle]
    start_pos: tuple[float, float]                 # rover spawn (x, y)
    start_yaw: float                               # rover spawn yaw (rad)
    goal_pos: tuple[float, float]                  # final goal (x, y)
    goal_radius: float = 0.8
    # Intermediate waypoints visited in order BEFORE the rover heads to
    # goal_pos. Empty = single-stop terrain (current behavior). Used to bias
    # navigation around blocking obstacles (e.g. T1_blocked_arc).
    waypoints: tuple[tuple[float, float], ...] = ()
    waypoint_radius: float = 1.5
    friction: tuple[float, float, float] = (1.0, 0.05, 0.001)
    ground_rgba: tuple[float, float, float, float] = (0.65, 0.45, 0.3, 1.0)
    skybox_rgb1: tuple[float, float, float] = (0.8, 0.6, 0.5)
    skybox_rgb2: tuple[float, float, float] = (0.4, 0.3, 0.3)
    # Optional heightmap; when set, the ground geom switches from `plane` to `hfield`.
    heightmap: np.ndarray | None = None
    heightmap_extent: tuple[float, float, float] = (15.0, 15.0, 1.0)
    # Per-reset domain randomization. When set, the env calls this with its
    # seeded `np_random` on every reset and applies the returned `TerrainRoll`
    # to model.geom_pos / model.geom_size / model.hfield_data plus
    # start_pos / goal_pos / waypoints. The number of obstacles in this
    # TerrainSpec is the MAX slot count — the randomizer can hide extras by
    # returning z=HIDE_Z for them.
    randomize_on_reset: TerrainRandomizer | None = None

    def __post_init__(self) -> None:
        if self.heightmap is None:
            return
        hm = np.asarray(self.heightmap)
        if hm.ndim != 2:
            raise ValueError(
                f"heightmap must be 2D (nrow, ncol); got shape {hm.shape}"
            )
        try:
            hm_f = hm.astype(np.float32, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"heightmap must be float-compatible: {exc}") from exc
        if not np.isfinite(hm_f).all():
            raise ValueError("heightmap contains non-finite values (NaN/Inf)")
        # small tolerance for floating-point noise around 0/1
        if hm_f.min() < -1e-6 or hm_f.max() > 1.0 + 1e-6:
            raise ValueError(
                f"heightmap values must lie in [0, 1]; got "
                f"[{hm_f.min():.4f}, {hm_f.max():.4f}]"
            )
        # store normalized float32, clipped exactly into [0, 1]
        self.heightmap = np.clip(hm_f, 0.0, 1.0)


def _ground_geom_xml(spec: TerrainSpec) -> tuple[str, str]:
    """Return (extra_asset_xml, ground_geom_xml) for the given spec."""
    fric = spec.friction
    g_rgba = spec.ground_rgba

    if spec.heightmap is None:
        ground_geom = (
            f'<geom name="ground" type="plane"\n'
            f'          size="{spec.arena_half_extent} {spec.arena_half_extent} 0.1"\n'
            f'          material="ground_mat"\n'
            f'          friction="{fric[0]} {fric[1]} {fric[2]}"\n'
            f'          contype="1" conaffinity="1"/>'
        )
        return "", ground_geom

    nrow, ncol = spec.heightmap.shape
    rx, ry, ez = spec.heightmap_extent
    base_z = ez * 0.1
    hfield_asset = (
        f'<hfield name="terrain_hf" '
        f'nrow="{nrow}" ncol="{ncol}" '
        f'size="{rx} {ry} {ez} {base_z}"/>'
    )
    ground_geom = (
        f'<geom name="ground" type="hfield" hfield="terrain_hf"\n'
        f'          rgba="{g_rgba[0]} {g_rgba[1]} {g_rgba[2]} {g_rgba[3]}"\n'
        f'          friction="{fric[0]} {fric[1]} {fric[2]}"\n'
        f'          contype="1" conaffinity="1"/>'
    )
    return hfield_asset, ground_geom


def compose_scene(terrain: TerrainSpec, include_rover: bool = True) -> str:
    """Return a full MuJoCo MJCF that includes the rover and adds the terrain."""
    rover_include = f'<include file="{ASSETS_DIR / "rover.xml"}"/>' if include_rover else ""

    # When the terrain is randomized, each obstacle needs to be movable at
    # runtime. MuJoCo's *static-geom* broad-phase BVH is built at compile
    # time from model.geom_pos and is NOT refreshed when geom_pos is later
    # mutated — so an obstacle relocated via geom_pos write becomes invisible
    # to collision queries. The canonical workaround is to attach each
    # movable obstacle to a `mocap="true"` body, then update its position via
    # `data.mocap_pos[mocap_id]`. Mocap bodies use the dynamic-AABB path and
    # broad phase tracks them correctly.
    if terrain.randomize_on_reset is not None:
        obstacle_xml = "\n        ".join(
            f'<body mocap="true" name="obs_body_{i}" '
            f'pos="{ob.pos[0]} {ob.pos[1]} {ob.pos[2]}">'
            f'<geom name="obs_{i}" type="box" '
            f'size="{ob.size[0]} {ob.size[1]} {ob.size[2]}" '
            f'rgba="{ob.rgba[0]} {ob.rgba[1]} {ob.rgba[2]} {ob.rgba[3]}" '
            f'contype="1" conaffinity="1" friction="1.0 0.05 0.001"/>'
            f'</body>'
            for i, ob in enumerate(terrain.obstacles)
        )
    else:
        obstacle_xml = "\n        ".join(
            ob.to_mjcf(f"obs_{i}") for i, ob in enumerate(terrain.obstacles)
        )

    # Marker height: for flat-plane terrains a thin disk at ground level
    # reads fine; for heightmap terrains the disk gets buried under any
    # dune the marker happens to land on (e.g. RT_dunes goal sitting under
    # a 0.4 m dune crest). Fix: render markers as TALL VERTICAL POSTS
    # centred at z = (elevation_z + 0.4), spanning from below the dunes up
    # to clearly above. Always visible regardless of where the marker lands.
    if terrain.heightmap is not None:
        ez = float(terrain.heightmap_extent[2])
        marker_centre_z = ez + 0.6
        marker_half_height = ez + 0.8     # post spans z ∈ [-0.2, 2*ez + 1.4]
        marker_radius_scale = 0.18         # thin posts, not disks
        wp_radius_scale = 0.14
        wp_centre_z = ez + 0.5
    else:
        marker_centre_z = 0.02
        marker_half_height = 0.02
        marker_radius_scale = 1.0
        wp_radius_scale = 1.0
        wp_centre_z = 0.04

    waypoint_xml = "\n        ".join(
        f'<site name="waypoint_{i}" pos="{wx} {wy} {wp_centre_z}" '
        f'size="{terrain.waypoint_radius * wp_radius_scale} {marker_half_height}" '
        f'type="cylinder" rgba="0.3 0.55 0.95 0.55"/>'
        for i, (wx, wy) in enumerate(terrain.waypoints)
    )

    g_rgba = terrain.ground_rgba
    sx, sy = terrain.start_pos
    gx, gy = terrain.goal_pos
    start_radius = 0.4 * marker_radius_scale
    goal_radius = terrain.goal_radius * marker_radius_scale

    hfield_asset_xml, ground_geom_xml = _ground_geom_xml(terrain)

    return f"""<mujoco model="{terrain.name}">
  {rover_include}

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.85 0.7 0.55 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient"
             rgb1="{terrain.skybox_rgb1[0]} {terrain.skybox_rgb1[1]} {terrain.skybox_rgb1[2]}"
             rgb2="{terrain.skybox_rgb2[0]} {terrain.skybox_rgb2[1]} {terrain.skybox_rgb2[2]}"
             width="512" height="512"/>
    <texture type="2d" name="ground_tex" builtin="checker"
             rgb1="{g_rgba[0]} {g_rgba[1]} {g_rgba[2]}"
             rgb2="{g_rgba[0]*0.85:.3f} {g_rgba[1]*0.85:.3f} {g_rgba[2]*0.85:.3f}"
             width="200" height="200"/>
    <material name="ground_mat" texture="ground_tex" texrepeat="20 20"
              specular="0.05" shininess="0.05"/>
    {hfield_asset_xml}
  </asset>

  <worldbody>
    {ground_geom_xml}
    <light name="sun" pos="0 0 15" dir="0 0 -1" diffuse="0.6 0.6 0.6" castshadow="false"/>

    <site name="start_marker" pos="{sx} {sy} {marker_centre_z}"
          size="{start_radius} {marker_half_height}" type="cylinder"
          rgba="0.4 0.7 1.0 0.55"/>
    {waypoint_xml}
    <site name="goal_marker" pos="{gx} {gy} {marker_centre_z}"
          size="{goal_radius} {marker_half_height}"
          type="cylinder" rgba="0.2 0.9 0.2 0.7"/>

    {obstacle_xml}
  </worldbody>
</mujoco>
"""


def compile_scene(spec: TerrainSpec) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compose the scene XML, load it into MuJoCo, and apply any heightmap data."""
    xml = compose_scene(spec)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    if spec.heightmap is not None:
        flat = np.asarray(spec.heightmap, dtype=np.float32).ravel(order="C")
        if model.hfield_data.shape != flat.shape:
            raise AssertionError(
                f"hfield size mismatch: model has {model.hfield_data.shape}, "
                f"heightmap is {flat.shape}"
            )
        model.hfield_data[:] = flat
        # propagate the data to the GPU/visual buffers and recompute derived quantities
        mujoco.mj_forward(model, data)
    return model, data


# ---------------------------------------------------------------------------
# Heightmap generators
# ---------------------------------------------------------------------------


def generate_heightmap_perlin(
    nrow: int = 64,
    ncol: int = 64,
    scale: float = 0.1,
    octaves: int = 3,
    seed: int = 0,
) -> np.ndarray:
    """Smoothly varying [0, 1] heightmap from OpenSimplex noise.

    Sums `octaves` octaves with halving amplitude and doubling frequency, then
    rescales the result to the full [0, 1] range so the hfield always uses the
    requested elevation_z.
    """
    import opensimplex

    noise = opensimplex.OpenSimplex(seed=seed)
    out = np.zeros((nrow, ncol), dtype=np.float32)
    for o in range(octaves):
        freq = scale * (2.0 ** o)
        amp = 0.5 ** o
        for i in range(nrow):
            for j in range(ncol):
                out[i, j] += amp * noise.noise2(j * freq, i * freq)
    mn, mx = float(out.min()), float(out.max())
    if mx - mn < 1e-9:
        return np.zeros_like(out)
    return ((out - mn) / (mx - mn)).astype(np.float32)


def generate_heightmap_slope(
    nrow: int = 64,
    ncol: int = 64,
    grade: float = 0.15,
    axis: str = "y",
) -> np.ndarray:
    """Linear ramp normalized to [0, 1]; useful for slope / incline scenarios.

    The `grade` parameter (rise / run) is preserved by the elevation_z field of
    the heightmap_extent; here we just emit a monotonic [0, 1] ramp along the
    requested axis. The caller is responsible for choosing an extent that
    yields the desired physical grade.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y'; got {axis!r}")
    ramp_row = np.linspace(0.0, 1.0, nrow, dtype=np.float32)
    ramp_col = np.linspace(0.0, 1.0, ncol, dtype=np.float32)
    if axis == "y":
        out = np.broadcast_to(ramp_row[:, None], (nrow, ncol)).copy()
    else:
        out = np.broadcast_to(ramp_col[None, :], (nrow, ncol)).copy()
    # `grade` does not change the [0, 1] shape but is part of the function's
    # public contract so callers can stash it for documentation / tuning.
    _ = grade
    return out


# ---------------------------------------------------------------------------
# Terrain catalog
# ---------------------------------------------------------------------------

TerrainFactory = Callable[[int], TerrainSpec]


def terrain_T1_flat(seed: int = 0) -> TerrainSpec:
    """T1: flat plain with 3 boulders in a staggered S-pattern.

    Each obstacle sits to alternating sides of the rover's start→goal line so
    the rover must steer to navigate around them, but the center corridor is
    never fully blocked. Obstacle half-sizes are 0.5–0.6 m so their tops sit
    at z ≈ 1.0–1.2 m, above the lidar origin (z=0.95) — i.e. the policy can
    actually *see* them before colliding.
    """
    rng = random.Random(seed)
    # Flip the zig-zag chirality based on seed so the policy can't lock in a
    # single hard-coded S-curve across seeds.
    signs = [+1, -1, +1] if rng.random() > 0.5 else [-1, +1, -1]
    y_centers = [3.5, 6.0, 8.5]
    obstacles: list[Obstacle] = []
    for sign, y_c in zip(signs, y_centers):
        # Center range 1.8–2.6 keeps the obstacle's inner face ≥ 1.2 m from
        # the rover's start-axis: rocker arms extend ≈ 0.9 m from the chassis
        # center, so this leaves ~0.3 m of clearance for a rover going dead
        # straight — visible to the lidar but not a physical wall.
        x = sign * rng.uniform(1.8, 2.6)
        y = y_c + rng.uniform(-0.4, 0.4)
        s = rng.uniform(0.5, 0.6)
        obstacles.append(Obstacle(pos=(x, y, s), size=(s, s, s)))
    return TerrainSpec(
        name="T1_flat",
        arena_half_extent=15.0,
        obstacles=obstacles,
        start_pos=(0.0, 0.0),
        start_yaw=0.0,
        goal_pos=(0.0, 10.0),
        goal_radius=1.0,
    )


def terrain_T1_blocked_arc(seed: int = 0) -> TerrainSpec:
    """T1 variant: the middle obstacle sits dead-center on the start→goal line.

    Forces the rover to deviate around it. A waypoint at (-2, 6) — to the LEFT
    of the blocker — pulls the heading-to-goal signal left, so the natural
    gradient of the progress reward shapes a counter-clockwise arc rather than
    a 50/50 left-or-right toss-up. The first and third boulders are still
    off-axis (same staggered logic as T1_flat) so the rover meets the blocker
    in a mostly-clean corridor.
    """
    rng = random.Random(seed)
    # Endpoints stay off-center (left / right alternating); the middle obstacle
    # is the new pin in the rover's path.
    first_sign = +1 if rng.random() > 0.5 else -1
    obstacles: list[Obstacle] = [
        Obstacle(
            pos=(first_sign * rng.uniform(1.8, 2.6),
                 3.5 + rng.uniform(-0.4, 0.4),
                 0.55),
            size=(0.55, 0.55, 0.55),
        ),
        Obstacle(
            pos=(rng.uniform(-0.25, 0.25),
                 6.0 + rng.uniform(-0.2, 0.2),
                 0.55),
            size=(0.55, 0.55, 0.55),
        ),
        Obstacle(
            pos=(-first_sign * rng.uniform(1.8, 2.6),
                 8.5 + rng.uniform(-0.4, 0.4),
                 0.55),
            size=(0.55, 0.55, 0.55),
        ),
    ]
    return TerrainSpec(
        name="T1_blocked_arc",
        arena_half_extent=15.0,
        obstacles=obstacles,
        start_pos=(0.0, 0.0),
        start_yaw=0.0,
        goal_pos=(0.0, 10.0),
        goal_radius=1.0,
        waypoints=((-2.2, 6.0),),  # arc-left bias around the center blocker
        waypoint_radius=1.5,
    )


def terrain_T1_blocked_arc_hills(seed: int = 0) -> TerrainSpec:
    """T1_blocked_arc with a gentle perlin-noise heightmap (≤ 0.15 m bumps).

    Same obstacle layout + waypoint as T1_blocked_arc — the rover still has to
    arc-left around a centerline blocker — but the ground is mildly uneven so
    suspension and wheel-slip dynamics come into play. Bump amplitude is small
    enough that flat-terrain policies should mostly transfer.
    """
    rng = random.Random(seed + 9001)  # offset so the hfield doesn't co-vary with obstacles
    first_sign = +1 if rng.random() > 0.5 else -1
    obstacles: list[Obstacle] = [
        Obstacle(
            pos=(first_sign * rng.uniform(1.8, 2.6),
                 3.5 + rng.uniform(-0.4, 0.4),
                 0.55),
            size=(0.55, 0.55, 0.55),
        ),
        Obstacle(
            pos=(rng.uniform(-0.25, 0.25),
                 6.0 + rng.uniform(-0.2, 0.2),
                 0.55),
            size=(0.55, 0.55, 0.55),
        ),
        Obstacle(
            pos=(-first_sign * rng.uniform(1.8, 2.6),
                 8.5 + rng.uniform(-0.4, 0.4),
                 0.55),
            size=(0.55, 0.55, 0.55),
        ),
    ]
    hm = generate_heightmap_perlin(nrow=48, ncol=48, scale=0.10, octaves=2, seed=seed)
    return TerrainSpec(
        name="T1_blocked_arc_hills",
        arena_half_extent=15.0,
        obstacles=obstacles,
        start_pos=(0.0, 0.0),
        start_yaw=0.0,
        goal_pos=(0.0, 10.0),
        goal_radius=1.0,
        waypoints=((-2.2, 6.0),),
        waypoint_radius=1.5,
        ground_rgba=(0.60, 0.45, 0.35, 1.0),
        heightmap=hm,
        heightmap_extent=(15.0, 15.0, 0.15),
    )


def terrain_T2_corridor(seed: int = 0) -> TerrainSpec:
    """T2: a narrow corridor between two long walls. Tests precision."""
    rng = random.Random(seed)
    wall_h = 0.6
    wall_len = 8.0
    corridor_half_width = rng.uniform(1.2, 1.6)
    obstacles = [
        Obstacle(pos=(-corridor_half_width - 0.3, 5.0, wall_h),
                 size=(0.3, wall_len / 2, wall_h),
                 rgba=(0.45, 0.35, 0.3, 1.0)),
        Obstacle(pos=(corridor_half_width + 0.3, 5.0, wall_h),
                 size=(0.3, wall_len / 2, wall_h),
                 rgba=(0.45, 0.35, 0.3, 1.0)),
    ]
    return TerrainSpec(
        name="T2_corridor",
        arena_half_extent=15.0,
        obstacles=obstacles,
        start_pos=(0.0, 0.0),
        start_yaw=0.0,
        goal_pos=(0.0, 10.0),
        goal_radius=0.7,
        ground_rgba=(0.55, 0.4, 0.35, 1.0),
    )


def terrain_T3_obstacle_field(seed: int = 0) -> TerrainSpec:
    """T3: dense scattered obstacles. Harder navigation."""
    rng = random.Random(seed)
    obstacles: list[Obstacle] = []
    for _ in range(14):
        x = rng.uniform(-5.0, 5.0)
        y = rng.uniform(2.0, 9.0)
        s = rng.uniform(0.25, 0.55)
        obstacles.append(Obstacle(pos=(x, y, s), size=(s, s, s)))
    return TerrainSpec(
        name="T3_obstacle_field",
        arena_half_extent=15.0,
        obstacles=obstacles,
        start_pos=(0.0, 0.0),
        start_yaw=0.0,
        goal_pos=(0.0, 11.0),
        goal_radius=0.8,
        ground_rgba=(0.5, 0.35, 0.3, 1.0),
    )


def terrain_T4_dunes(seed: int = 0) -> TerrainSpec:
    """T4: rolling dunes generated from OpenSimplex noise. No box obstacles."""
    hm = generate_heightmap_perlin(
        nrow=64, ncol=64, scale=0.08, octaves=4, seed=seed
    )
    return TerrainSpec(
        name="T4_dunes",
        arena_half_extent=15.0,
        obstacles=[],
        start_pos=(0.0, 0.0),
        start_yaw=0.0,
        goal_pos=(0.0, 10.0),
        goal_radius=0.8,
        ground_rgba=(0.7, 0.5, 0.35, 1.0),
        heightmap=hm,
        heightmap_extent=(15.0, 15.0, 0.5),
    )


def terrain_T6_slope(seed: int = 0) -> TerrainSpec:
    """T6: gentle incline (slope) along the +Y axis."""
    hm = generate_heightmap_slope(nrow=64, ncol=64, grade=0.10, axis="y")
    return TerrainSpec(
        name="T6_slope",
        arena_half_extent=15.0,
        obstacles=[],
        start_pos=(0.0, 0.0),
        start_yaw=0.0,
        goal_pos=(0.0, 10.0),
        goal_radius=0.8,
        ground_rgba=(0.6, 0.45, 0.35, 1.0),
        heightmap=hm,
        heightmap_extent=(15.0, 15.0, 1.5),
    )


# ---------------------------------------------------------------------------
# Randomized terrains (RT_*) — for the robust curriculum in scenario_10.
#
# Each factory builds a "template" TerrainSpec with N obstacle slots compiled
# in (or zero for hfield-only / open-arena phases) and a `randomize_on_reset`
# closure that re-rolls obstacle positions, start, goal, and (where relevant)
# the heightmap shape from the env's seeded np_random on every reset.
#
# Difficulty ramps across the 8 phases — see docs/design/scenarios.md.
# ---------------------------------------------------------------------------


def _flat_template(name: str, max_obstacles: int = 0,
                   arena_half: float = 15.0,
                   ground_rgba: tuple[float, float, float, float] = (0.65, 0.45, 0.3, 1.0)
                   ) -> TerrainSpec:
    """Build an empty flat-terrain template with N hidden obstacle slots.

    The TerrainSpec returned has all obstacle positions at HIDE_Z initially;
    each reset, the `randomize_on_reset` closure overwrites them.
    """
    obstacles = [
        Obstacle(pos=(0.0, 0.0, HIDE_Z), size=(0.5, 0.5, 0.5))
        for _ in range(max_obstacles)
    ]
    return TerrainSpec(
        name=name, arena_half_extent=arena_half, obstacles=obstacles,
        start_pos=(0.0, 0.0), start_yaw=0.0,
        goal_pos=(0.0, 8.0), goal_radius=1.0,
        ground_rgba=ground_rgba,
    )


def _hfield_template(name: str, max_obstacles: int,
                     elevation_z: float, hm_seed: int = 0,
                     arena_half: float = 15.0,
                     ground_rgba: tuple[float, float, float, float] = (0.7, 0.5, 0.35, 1.0)
                     ) -> TerrainSpec:
    """Same as `_flat_template` but compiles in a heightmap of the given max elevation."""
    obstacles = [
        Obstacle(pos=(0.0, 0.0, HIDE_Z), size=(0.5, 0.5, 0.5))
        for _ in range(max_obstacles)
    ]
    hm = generate_heightmap_perlin(nrow=48, ncol=48, scale=0.10, octaves=3, seed=hm_seed)
    return TerrainSpec(
        name=name, arena_half_extent=arena_half, obstacles=obstacles,
        start_pos=(0.0, 0.0), start_yaw=0.0,
        goal_pos=(0.0, 8.0), goal_radius=1.0,
        ground_rgba=ground_rgba,
        heightmap=hm,
        heightmap_extent=(arena_half, arena_half, elevation_z),
    )


def terrain_RT_drive_random(seed: int = 0) -> TerrainSpec:
    """Phase 0: drive from random start to random goal on a flat plane.

    No obstacles. Goal is at least 5 m from start. Tests "basic locomotion"
    over a wide range of starting orientations and goal directions.

    20% of episodes spawn with the goal in the *side* cone (90°-ish manoeuvre)
    and another 20% put it *behind* the rover (near-U-turn). The rest stay
    uniform. This trains turning-from-rest in the very first phase so later
    obstacle / waypoint phases inherit a policy that can already reorient.
    """
    spec = _flat_template("RT_drive_random", max_obstacles=0)

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        r = rng.uniform()
        if r < 0.2:
            bearing = "behind"
        elif r < 0.4:
            bearing = "side"
        else:
            bearing = "uniform"
        start, yaw, goal = sample_start_goal_pair(
            rng, arena_half=12.0, min_separation=5.0, margin=2.0,
            relative_bearing=bearing,
        )
        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=goal,
            waypoints=(),
            obstacle_positions=[], obstacle_sizes=[],
        )
    spec.randomize_on_reset = _roll
    return spec


def terrain_RT_with_waypoint(seed: int = 0) -> TerrainSpec:
    """Phase 1: drive through 1 random waypoint to a random goal. Flat, no obstacles.

    Mixes spawn orientations: 60% the goal is straight ahead (`front`), 20%
    it's off to a side, 20% it's behind. With the manoeuvre-biased starts
    the rover must turn first, then chase the waypoint. The waypoint itself
    keeps low lateral jitter so it stays roughly on the post-turn line.
    """
    spec = _flat_template("RT_with_waypoint", max_obstacles=0)

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        r = rng.uniform()
        if r < 0.6:
            bearing = "front"
        elif r < 0.8:
            bearing = "side"
        else:
            bearing = "behind"
        start, yaw, goal = sample_start_goal_pair(
            rng, arena_half=12.0, min_separation=6.0, margin=2.0,
            relative_bearing=bearing,
        )
        # Lateral jitter reduced 2.0 → 0.8 so the waypoint sits close to the
        # obvious start→goal line. With wider jitter the rover's "drive
        # straight" prior missed the waypoint entirely and never learned the
        # task. Once it's reliably hit, jitter can grow in later phases.
        wps = sample_waypoints_between(rng, start, goal, n_waypoints=1, lateral_jitter=0.8)
        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=goal,
            waypoints=wps,
            obstacle_positions=[], obstacle_sizes=[],
        )
    spec.randomize_on_reset = _roll
    return spec


def terrain_RT_with_two_waypoints(seed: int = 0) -> TerrainSpec:
    """Phase 2: drive through 2-4 random waypoints along a TWISTY route.

    The waypoint count itself randomizes (2..4) so the policy has to handle
    variable-length chains. Routes use `sample_twisty_waypoints` which puts
    waypoints alternately to the left / right of the straight start→goal
    line — a real zig-zag the rover must steer through, not jitter around a
    straight line. Forces the policy to learn segment-by-segment heading
    changes, not "drive straight and hope a waypoint is close enough".
    """
    spec = _flat_template("RT_with_two_waypoints", max_obstacles=0)

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        # Mix start orientations so the policy also learns to turn-then-chase.
        r = rng.uniform()
        if r < 0.5:
            bearing = "front"
        elif r < 0.8:
            bearing = "side"
        else:
            bearing = "behind"
        start, yaw, goal = sample_start_goal_pair(
            rng, arena_half=12.0, min_separation=8.0, margin=2.0,
            relative_bearing=bearing,
        )
        n_wp = int(rng.integers(2, 5))  # 2..4 inclusive
        wps = sample_twisty_waypoints(
            rng, start, goal, n_waypoints=n_wp,
            arena_half=12.0, margin=2.0,
            swing_min=1.8, swing_max=3.2,
            alternate_sides=True,
        )
        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=goal,
            waypoints=wps,
            obstacle_positions=[], obstacle_sizes=[],
        )
    spec.randomize_on_reset = _roll
    return spec


_NO_OBSTACLE_PROB = 0.20  # fraction of episodes that get zero obstacles


# ---------------------------------------------------------------------------
# Navigation-skill curriculum (no obstacles). Each terrain trains ONE specific
# manoeuvre — arc steering, sharp 90° turn, U-turn, slalom. Used before the
# obstacle-avoidance phases so the policy enters those with a solid driving
# foundation. Bounded randomisation per phase: layout pattern is fixed (e.g.
# "2 waypoints on an arc"), specifics (arc angle, start orientation, start
# and goal positions) vary per episode.
# ---------------------------------------------------------------------------


def terrain_RT_waypoints_arc(seed: int = 0) -> TerrainSpec:
    """Phase: 2 waypoints arranged along an arc bulging off the start→goal line.

    Arc sweep is uniform in 30°–90° per episode. Tests "lay your steering
    into a curve" — every waypoint sits at the same arc radius, so a single
    well-chosen steer angle should trace through all of them.
    """
    spec = _flat_template("RT_waypoints_arc", max_obstacles=0)

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        start, yaw, goal = sample_start_goal_pair(
            rng, arena_half=12.0, min_separation=7.0, margin=2.0,
        )
        wps = sample_arc_waypoints(
            rng, start, goal, n_waypoints=2, arc_angle_deg=(30.0, 90.0),
        )
        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=goal,
            waypoints=wps,
            obstacle_positions=[], obstacle_sizes=[],
        )
    spec.randomize_on_reset = _roll
    return spec


def terrain_RT_waypoint_90(seed: int = 0) -> TerrainSpec:
    """Phase: 1 waypoint at ~90° from the rover's starting heading.

    Two waypoints total: the 90°-turn waypoint, then the final goal beyond
    it (in roughly the post-turn direction). Trains sharp directional
    changes from rest. Sign (left vs right) is randomised per episode.
    """
    spec = _flat_template("RT_waypoint_90", max_obstacles=0)

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        start, yaw, _goal = sample_start_goal_pair(
            rng, arena_half=12.0, min_separation=5.0, margin=2.0,
        )
        wp = sample_waypoint_at_bearing(
            rng, start, yaw,
            relative_bearing_range=(np.deg2rad(75), np.deg2rad(105)),
            distance_range=(4.0, 6.0),
            arena_half=12.0, margin=2.0,
        )
        # Goal: roughly in the post-turn direction from the waypoint.
        # Use the waypoint's relative bearing from start to find the turn
        # direction, then extrapolate.
        dx, dy = wp[0] - start[0], wp[1] - start[1]
        # Project further along that vector for the final goal.
        goal_dist = float(rng.uniform(3.0, 5.0))
        norm = (dx * dx + dy * dy) ** 0.5
        bound = 12.0 - 2.0
        gx = float(np.clip(wp[0] + goal_dist * dx / norm, -bound, bound))
        gy = float(np.clip(wp[1] + goal_dist * dy / norm, -bound, bound))
        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=(gx, gy),
            waypoints=(wp,),
            obstacle_positions=[], obstacle_sizes=[],
        )
    spec.randomize_on_reset = _roll
    return spec


def terrain_RT_waypoint_180(seed: int = 0) -> TerrainSpec:
    """Phase: 1 waypoint behind the rover, forcing a near-U-turn.

    Bearing uniform in 135°–180° relative to start yaw. Trains the rover
    to reorient when the target is behind it from rest.
    """
    spec = _flat_template("RT_waypoint_180", max_obstacles=0)

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        start, yaw, _ = sample_start_goal_pair(
            rng, arena_half=12.0, min_separation=5.0, margin=2.0,
        )
        wp = sample_waypoint_at_bearing(
            rng, start, yaw,
            relative_bearing_range=(np.deg2rad(135), np.deg2rad(180)),
            distance_range=(4.0, 7.0),
            arena_half=12.0, margin=2.0,
        )
        # Goal further along same vector.
        dx, dy = wp[0] - start[0], wp[1] - start[1]
        norm = (dx * dx + dy * dy) ** 0.5
        goal_dist = float(rng.uniform(3.0, 5.0))
        bound = 12.0 - 2.0
        gx = float(np.clip(wp[0] + goal_dist * dx / norm, -bound, bound))
        gy = float(np.clip(wp[1] + goal_dist * dy / norm, -bound, bound))
        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=(gx, gy),
            waypoints=(wp,),
            obstacle_positions=[], obstacle_sizes=[],
        )
    spec.randomize_on_reset = _roll
    return spec


def terrain_RT_slalom(seed: int = 0) -> TerrainSpec:
    """Phase: 4-5 alternating-side waypoints in an S-curve / slalom pattern.

    Fixed-side alternation (left-right-left…), randomised swing magnitude
    and starting side per episode. Trains rhythmic steering reversals
    — the rover learns to plan one segment ahead.
    """
    spec = _flat_template("RT_slalom", max_obstacles=0)

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        start, yaw, goal = sample_start_goal_pair(
            rng, arena_half=12.0, min_separation=9.0, margin=2.0,
        )
        n_wp = int(rng.integers(4, 6))  # 4 or 5
        wps = sample_slalom_waypoints(
            rng, start, goal, n_waypoints=n_wp,
            swing_min=1.8, swing_max=3.0,
        )
        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=goal,
            waypoints=wps,
            obstacle_positions=[], obstacle_sizes=[],
        )
    spec.randomize_on_reset = _roll
    return spec


def terrain_RT_one_obstacle(seed: int = 0) -> TerrainSpec:
    """Phase 3: random start/goal with up to one random obstacle in the path.

    20% of episodes generate ZERO obstacles. This is critical for the policy
    to learn that "no obstacle visible" → "drive straight" — otherwise it
    bakes a "veer wide" prior from the obstacle distribution and overshoots
    every goal on no-obstacle eval (see report_phase_3_..._on_RT_drive_random).
    """
    spec = _flat_template("RT_one_obstacle", max_obstacles=1)

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        start, yaw, goal = sample_start_goal_pair(
            rng, arena_half=12.0, min_separation=6.0, margin=2.0,
        )
        n_obs = 0 if rng.uniform() < _NO_OBSTACLE_PROB else 1
        pos, sz = sample_obstacles_along_path(
            rng, start, goal,
            n_obstacles=n_obs, max_slots=1,
            size_range=(0.45, 0.65),
            lateral_jitter=0.6,
        )
        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=goal, waypoints=(),
            obstacle_positions=pos, obstacle_sizes=sz,
        )
    spec.randomize_on_reset = _roll
    return spec


def terrain_RT_obstacle_field(seed: int = 0) -> TerrainSpec:
    """Phase 4: random start/goal with 0-6 random obstacles between them.

    20% no-obstacle episodes (see terrain_RT_one_obstacle docstring).
    """
    spec = _flat_template("RT_obstacle_field", max_obstacles=6,
                          ground_rgba=(0.55, 0.4, 0.3, 1.0))

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        start, yaw, goal = sample_start_goal_pair(
            rng, arena_half=12.0, min_separation=7.0, margin=2.0,
        )
        if rng.uniform() < _NO_OBSTACLE_PROB:
            n = 0
        else:
            n = int(rng.integers(3, 7))  # 3..6 inclusive
        pos, sz = sample_obstacles_along_path(
            rng, start, goal,
            n_obstacles=n, max_slots=6,
            size_range=(0.4, 0.7),
            lateral_jitter=1.4,
        )
        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=goal, waypoints=(),
            obstacle_positions=pos, obstacle_sizes=sz,
        )
    spec.randomize_on_reset = _roll
    return spec


def terrain_RT_dense_obstacles(seed: int = 0) -> TerrainSpec:
    """Phase 5: dense field — 0-12 random obstacles. Tests precision navigation.

    20% no-obstacle episodes (see terrain_RT_one_obstacle docstring).
    """
    spec = _flat_template("RT_dense_obstacles", max_obstacles=12,
                          ground_rgba=(0.5, 0.35, 0.3, 1.0))

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        start, yaw, goal = sample_start_goal_pair(
            rng, arena_half=12.0, min_separation=8.0, margin=2.0,
        )
        if rng.uniform() < _NO_OBSTACLE_PROB:
            n = 0
        else:
            n = int(rng.integers(8, 13))  # 8..12 inclusive
        pos, sz = sample_obstacles_along_path(
            rng, start, goal,
            n_obstacles=n, max_slots=12,
            size_range=(0.35, 0.6),
            lateral_jitter=2.5,
        )
        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=goal, waypoints=(),
            obstacle_positions=pos, obstacle_sizes=sz,
        )
    spec.randomize_on_reset = _roll
    return spec


def terrain_RT_gentle_hills(seed: int = 0) -> TerrainSpec:
    """Phase 6: random start/goal on a gentle randomized heightmap (≤ 0.2 m)."""
    spec = _hfield_template(
        "RT_gentle_hills", max_obstacles=0,
        elevation_z=0.2, hm_seed=seed,
        ground_rgba=(0.65, 0.48, 0.36, 1.0),
    )

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        start, yaw, goal = sample_start_goal_pair(
            rng, arena_half=11.0, min_separation=6.0, margin=2.5,
        )
        # Fresh perlin heightmap each episode for variety.
        hm = generate_heightmap_perlin(
            nrow=48, ncol=48, scale=0.10, octaves=3,
            seed=int(rng.integers(0, 10_000_000)),
        )
        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=goal, waypoints=(),
            obstacle_positions=[], obstacle_sizes=[],
            heightmap=hm,
        )
    spec.randomize_on_reset = _roll
    return spec


def terrain_RT_dunes(seed: int = 0) -> TerrainSpec:
    """Phase 7: bigger dunes — randomized heightmap up to 0.6 m. Tests suspension."""
    spec = _hfield_template(
        "RT_dunes", max_obstacles=0,
        elevation_z=0.6, hm_seed=seed,
        ground_rgba=(0.7, 0.5, 0.35, 1.0),
    )

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        start, yaw, goal = sample_start_goal_pair(
            rng, arena_half=11.0, min_separation=6.0, margin=2.5,
        )
        # More octaves + lower scale → more rolling structure. Shape must
        # match the template's hfield slot (48×48 — see _hfield_template).
        hm = generate_heightmap_perlin(
            nrow=48, ncol=48, scale=0.08, octaves=4,
            seed=int(rng.integers(0, 10_000_000)),
        )
        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=goal, waypoints=(),
            obstacle_positions=[], obstacle_sizes=[],
            heightmap=hm,
        )
    spec.randomize_on_reset = _roll
    return spec


def terrain_RT_mixed(seed: int = 0) -> TerrainSpec:
    """Phase 8 (capstone): everything at once — obstacles + waypoints + heightmap."""
    spec = _hfield_template(
        "RT_mixed", max_obstacles=8,
        elevation_z=0.35, hm_seed=seed,
        ground_rgba=(0.6, 0.45, 0.34, 1.0),
    )

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        # 30% manoeuvre start (side/behind), 70% uniform.
        bearing = "maneuver" if rng.uniform() < 0.3 else "uniform"
        start, yaw, goal = sample_start_goal_pair(
            rng, arena_half=11.0, min_separation=7.0, margin=2.5,
            relative_bearing=bearing,
        )
        # 0-2 twisty waypoints, 4-8 obstacles.
        n_wp = int(rng.integers(0, 3))
        wps = (
            sample_twisty_waypoints(
                rng, start, goal, n_waypoints=n_wp,
                arena_half=11.0, margin=2.5,
                swing_min=1.5, swing_max=2.8,
                alternate_sides=True,
            )
            if n_wp > 0 else ()
        )
        # Generate the heightmap first so obstacles can sit ON the terrain
        # surface, not buried at z=0. RT_mixed elevation_z = 0.35 m so this
        # is a non-trivial correction.
        hm = generate_heightmap_perlin(
            nrow=48, ncol=48, scale=0.10, octaves=3,
            seed=int(rng.integers(0, 10_000_000)),
        )
        hm_extent = (11.0, 11.0, 0.35)
        n_obs = int(rng.integers(4, 9))  # 4..8 inclusive
        pos, sz = sample_obstacles_along_path(
            rng, start, goal,
            n_obstacles=n_obs, max_slots=8,
            size_range=(0.4, 0.65),
            lateral_jitter=2.0,
            heightmap=hm, heightmap_extent=hm_extent,
        )
        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=goal, waypoints=wps,
            obstacle_positions=pos, obstacle_sizes=sz,
            heightmap=hm,
        )
    spec.randomize_on_reset = _roll
    return spec


# ---------------------------------------------------------------------------
# scenario_11 robust-generalist curriculum (RC_*) — mixed-distribution phases.
#
# Each terrain below SAMPLES INTERNALLY over a non-trivial sub-distribution,
# rather than fixing one specific manoeuvre per phase. Two goals:
#   1) Anchor sub-distribution per phase: every phase keeps a fraction of
#      its earlier "trivial" case alive (e.g. RC_obstacle_avoidance has 25%
#      zero-obstacle episodes). Within-phase mixing is much stronger
#      protection against catastrophic forgetting than EWC alone.
#   2) Hard composition is *built up* from already-trained sub-pieces, not
#      introduced as a new task. The capstone RC_full_random is a uniform
#      draw over the cross-product of all prior axes; held-out evaluation
#      samples from the same distribution, so generalisation is just
#      "sample-the-same-way at test time".
# ---------------------------------------------------------------------------


def terrain_RC_locomotion(seed: int = 0) -> TerrainSpec:
    """Phase 0: drive to a single goal. No waypoints, no obstacles. Flat.

    Distance-curriculum WITHIN this single phase (a "near-to-far"
    bootstrap):

        50% short  (2-4 m, front cone)     — fast goal_bonus discovery
        25% medium (4-7 m, front cone)     — standard difficulty
        15% medium (4-7 m, side cone)      — short turn-from-rest
        10% long   (5-10 m, mixed bearing) — full-spec hardest case

    Rationale: random-init PPO needs to *experience* the +50 goal bonus
    early to learn the value of moving toward goals. With 8-12 m goals
    only (v1/v2), the rover essentially never reached the target during
    the first ~100k steps — the value function stayed flat and the
    policy gradient was driven entirely by per-step progress reward,
    which is dwarfed by the stuck-no-progress penalty when the rover
    can't make significant headway.

    With 50% of episodes at 2-4 m, even a noisy "drift forward" policy
    hits the goal often enough to anchor the value function. Once the
    policy learns "forward = good when target is ahead", the longer-
    distance slice generalises that knowledge.

    The 15% short-side slice keeps the turn-from-rest skill in the mix
    without trying to learn it from scratch on a 10 m + 90° goal
    (which is the hardest possible task and disastrous as a bootstrap).
    """
    spec = _flat_template("RC_locomotion", max_obstacles=0)

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        r = rng.uniform()
        if r < 0.50:
            bearing, dmin, dmax = "front", 2.0, 4.0
        elif r < 0.75:
            bearing, dmin, dmax = "front", 4.0, 7.0
        elif r < 0.90:
            bearing, dmin, dmax = "side", 4.0, 7.0
        else:
            # Hardest slice: full bearing mix at longer distances.
            r2 = rng.uniform()
            bearing = "front" if r2 < 0.5 else ("side" if r2 < 0.85 else "behind")
            dmin, dmax = 5.0, 10.0
        start, yaw, goal = sample_start_goal_pair(
            rng, arena_half=12.0, min_separation=dmin, max_separation=dmax,
            margin=2.0, relative_bearing=bearing,
        )
        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=goal,
            waypoints=(),
            obstacle_positions=[], obstacle_sizes=[],
        )
    spec.randomize_on_reset = _roll
    return spec


def terrain_RC_path_following(seed: int = 0) -> TerrainSpec:
    """Phase 1: waypoint chains of variable length and shape. Flat, no obstacles.

    Within-phase mix on chain configuration:
        25% 0 waypoints                    — anchor: keeps phase-0 alive
        25% 1 waypoint on-line (jitter ≤ 0.8 m)
        20% 2–3 twisty waypoints (alternating sides, swing 1.5–3.0 m)
        15% 2 waypoints on an arc (sweep 30°–90°, sign random)
        15% 4–5 slalom waypoints (swing 1.5–3.0 m)

    Distance curriculum (orthogonal to chain config):
        40% short paths  (start-goal distance 4–6 m)  — bootstrap waypoint
                                                        bonus discovery
        60% long paths   (start-goal distance 6–10 m) — generalisation

    The short slice lets PPO hit waypoint bonuses (+20) and goal bonuses
    (+50) often during the first ~100k steps — without it, the same
    bootstrap problem as phase 0 hits at the waypoint level.

    Mixed with 15% manoeuvre starts (side/behind cones) so the policy
    can't just learn "always start moving forward".
    """
    spec = _flat_template("RC_path_following", max_obstacles=0)

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        # Distance band: 40% short, 60% long.
        if rng.uniform() < 0.40:
            dmin, dmax = 4.0, 6.0
        else:
            dmin, dmax = 6.0, 10.0
        # Start orientation: 85% forward, 15% manoeuvre.
        bearing = "uniform" if rng.uniform() < 0.85 else "maneuver"
        start, yaw, goal = sample_start_goal_pair(
            rng, arena_half=12.0, min_separation=dmin, max_separation=dmax,
            margin=2.0, relative_bearing=bearing,
        )
        r = rng.uniform()
        if r < 0.25:
            wps = ()
        elif r < 0.50:
            wps = sample_waypoints_between(
                rng, start, goal, n_waypoints=1, lateral_jitter=0.8,
            )
        elif r < 0.70:
            n_wp = int(rng.integers(2, 4))
            wps = sample_twisty_waypoints(
                rng, start, goal, n_waypoints=n_wp,
                arena_half=12.0, margin=2.0,
                swing_min=1.5, swing_max=3.0,
                alternate_sides=True,
            )
        elif r < 0.85:
            wps = sample_arc_waypoints(
                rng, start, goal, n_waypoints=2,
                arc_angle_deg=(30.0, 90.0),
            )
        else:
            n_wp = int(rng.integers(4, 6))
            wps = sample_slalom_waypoints(
                rng, start, goal, n_waypoints=n_wp,
                swing_min=1.5, swing_max=3.0,
            )
        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=goal,
            waypoints=wps,
            obstacle_positions=[], obstacle_sizes=[],
        )
    spec.randomize_on_reset = _roll
    return spec


def terrain_RC_obstacle_avoidance(seed: int = 0) -> TerrainSpec:
    """Phase 2: obstacle dodging — density + distance curriculum.

    The v2/v3 runs showed phase 2 obliterating phases 0-1 (mean return
    went −30 → −100 in the first 200k steps) because the jump from
    "no obstacles" to "up to 10 obstacles" was too steep. The 32 obstacle
    obs-slots are padded sentinels for phases 0-1; they suddenly carry
    real signal here, and the policy can't learn to attend to them while
    also keeping the locomotion skill — it crashes both.

    Fix: a 2-axis curriculum WITHIN this phase. Mostly easy episodes
    (short goals, 1-2 obstacles) for bootstrap; some hard ones for
    generalisation.

    Within-phase mix:
        30% no obstacles, short goals (2-4 m)      — locomotion anchor.
                                                     50% of these get 1-2
                                                     waypoints to keep
                                                     path-following alive.
        35% 1-2 obstacles, short-medium (3-6 m)   — easiest avoidance.
        20% 2-3 obstacles, medium (4-7 m)         — standard.
        10% 4-6 obstacles, medium-long (5-9 m)    — harder field.
         5% 7-10 obstacles, long (6-10 m)         — dense field anchor.

    Also reverts ent_coef from 0.03 → default (handled in scenario_11
    phase plan, NOT here). Same lesson as phase 0: forced exploration
    actively prevents the policy from committing to a working strategy.
    """
    spec = _flat_template("RC_obstacle_avoidance", max_obstacles=10,
                          ground_rgba=(0.55, 0.40, 0.32, 1.0))

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        r = rng.uniform()
        wps: tuple = ()
        if r < 0.30:
            n, jitter, dmin, dmax = 0, 1.0, 2.0, 4.0
        elif r < 0.65:
            n = int(rng.integers(1, 3))   # 1 or 2
            jitter, dmin, dmax = 0.9, 3.0, 6.0
        elif r < 0.85:
            n = int(rng.integers(2, 4))   # 2 or 3
            jitter, dmin, dmax = 1.2, 4.0, 7.0
        elif r < 0.95:
            n = int(rng.integers(4, 7))   # 4..6
            jitter, dmin, dmax = 1.6, 5.0, 9.0
        else:
            n = int(rng.integers(7, 11))  # 7..10
            jitter, dmin, dmax = 2.2, 6.0, 10.0
        start, yaw, goal = sample_start_goal_pair(
            rng, arena_half=12.0,
            min_separation=dmin, max_separation=dmax,
            margin=2.0,
        )
        # Anchor sub-case: 50% of the no-obstacle slice gets waypoints
        # (= 15% of all episodes hold the path-following skill alive).
        if r < 0.30 and rng.uniform() < 0.5:
            wps = sample_waypoints_between(
                rng, start, goal,
                n_waypoints=int(rng.integers(1, 3)),
                lateral_jitter=0.6,
            )
        pos, sz = sample_obstacles_along_path(
            rng, start, goal,
            n_obstacles=n, max_slots=10,
            size_range=(0.4, 0.6),
            lateral_jitter=jitter,
        )
        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=goal, waypoints=wps,
            obstacle_positions=pos, obstacle_sizes=sz,
        )
    spec.randomize_on_reset = _roll
    return spec


def terrain_RC_path_and_obstacles(seed: int = 0) -> TerrainSpec:
    """Phase 3: compose waypoints + obstacles. Flat.

    Same 2-axis curriculum approach as RC_obstacle_avoidance — most
    episodes are easy compositions (short goal, 1-2 obstacles, 0-1
    waypoint); some are harder.

    Within-phase mix:
        25% waypoints only, short paths (4-6 m)        — phase-1 anchor.
        30% 1-2 obstacles only, short (3-5 m)          — phase-2 anchor.
        25% 1 waypoint + 1-2 obstacles, short (4-6 m)  — easy compose.
        15% 1-2 waypoints + 2-4 obstacles, med (5-8 m) — medium compose.
         5% 2 waypoints + 3-5 obstacles, longer (6-9m) — full compose.
    """
    spec = _flat_template("RC_path_and_obstacles", max_obstacles=8,
                          ground_rgba=(0.58, 0.42, 0.30, 1.0))

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        r = rng.uniform()
        if r < 0.25:
            # Phase-1 anchor: waypoints only.
            dmin, dmax = 4.0, 6.0
            n_obs = 0
            n_wp = int(rng.integers(1, 3))
            use_twisty = False
        elif r < 0.55:
            # Phase-2 anchor: obstacles only, easy density.
            dmin, dmax = 3.0, 5.0
            n_obs = int(rng.integers(1, 3))
            n_wp = 0
            use_twisty = False
        elif r < 0.80:
            # Easy compose: 1 waypoint + 1-2 obstacles.
            dmin, dmax = 4.0, 6.0
            n_obs = int(rng.integers(1, 3))
            n_wp = 1
            use_twisty = False
        elif r < 0.95:
            # Medium compose.
            dmin, dmax = 5.0, 8.0
            n_obs = int(rng.integers(2, 5))
            n_wp = int(rng.integers(1, 3))
            use_twisty = True
        else:
            # Full compose, harder.
            dmin, dmax = 6.0, 9.0
            n_obs = int(rng.integers(3, 6))
            n_wp = 2
            use_twisty = True
        start, yaw, goal = sample_start_goal_pair(
            rng, arena_half=12.0,
            min_separation=dmin, max_separation=dmax, margin=2.0,
        )
        if n_wp > 0:
            if use_twisty:
                wps = sample_twisty_waypoints(
                    rng, start, goal, n_waypoints=n_wp,
                    arena_half=12.0, margin=2.0,
                    swing_min=1.0, swing_max=2.2, alternate_sides=True,
                )
            else:
                wps = sample_waypoints_between(
                    rng, start, goal, n_waypoints=n_wp,
                    lateral_jitter=0.8,
                )
        else:
            wps = ()
        pos, sz = sample_obstacles_along_path(
            rng, start, goal,
            n_obstacles=n_obs, max_slots=8,
            size_range=(0.4, 0.6),
            lateral_jitter=1.3,
        )
        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=goal, waypoints=wps,
            obstacle_positions=pos, obstacle_sizes=sz,
        )
    spec.randomize_on_reset = _roll
    return spec


def terrain_RC_terrain(seed: int = 0) -> TerrainSpec:
    """Phase 4: drive over height. No obstacles, no waypoints.

    Within-phase mix on terrain elevation (heightmap scales linearly with
    `elevation_z` because the field itself is in [0, 1] — we control roughness
    by re-rolling the heightmap, and amplitude by SCALING in the roll):
        30% essentially flat (heightmap rolled but pre-scaled by 0.02)
        40% gentle hills (peak ≤ 0.2 m)
        30% bigger dunes (peak ≤ 0.5 m)

    Note: the compiled spec must declare ONE elevation_z (the maximum the
    hfield can address), so we set it to 0.5 m here and rescale the rolled
    heightmap to use 4%, 40%, or 100% of the available range per episode.
    """
    spec = _hfield_template(
        "RC_terrain", max_obstacles=0,
        elevation_z=0.5, hm_seed=seed,
        ground_rgba=(0.66, 0.48, 0.36, 1.0),
    )

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        start, yaw, goal = sample_start_goal_pair(
            rng, arena_half=11.0, min_separation=6.0, margin=2.5,
        )
        hm = generate_heightmap_perlin(
            nrow=48, ncol=48, scale=0.10, octaves=3,
            seed=int(rng.integers(0, 10_000_000)),
        )
        r = rng.uniform()
        if r < 0.30:
            hm = hm * 0.04           # nearly flat
        elif r < 0.70:
            hm = hm * 0.40           # gentle hills
        # else: full 100% of elevation_z=0.5 m → big dunes
        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=goal, waypoints=(),
            obstacle_positions=[], obstacle_sizes=[],
            heightmap=hm,
        )
    spec.randomize_on_reset = _roll
    return spec


def terrain_RC_terrain_plus(seed: int = 0) -> TerrainSpec:
    """Phase 5: terrain + obstacles. No waypoints.

    Within-phase mix:
        40% flat ground + 3–6 obstacles    — phase-2 anchor
        40% hfield (0.2 m) + 2–4 obstacles — composed
        20% hfield (0.3 m) only            — phase-4 anchor

    Obstacles sit on the local terrain surface (sample_obstacles_along_path
    queries the heightmap when one is provided).
    """
    spec = _hfield_template(
        "RC_terrain_plus", max_obstacles=6,
        elevation_z=0.4, hm_seed=seed,
        ground_rgba=(0.62, 0.46, 0.34, 1.0),
    )

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        start, yaw, goal = sample_start_goal_pair(
            rng, arena_half=11.0, min_separation=7.0, margin=2.5,
        )
        hm = generate_heightmap_perlin(
            nrow=48, ncol=48, scale=0.10, octaves=3,
            seed=int(rng.integers(0, 10_000_000)),
        )
        r = rng.uniform()
        if r < 0.40:
            hm = hm * 0.04           # essentially flat
            n_obs = int(rng.integers(3, 7))
        elif r < 0.80:
            hm = hm * 0.50           # ~0.2 m peaks (50% of 0.4)
            n_obs = int(rng.integers(2, 5))
        else:
            hm = hm * 0.75           # ~0.3 m peaks
            n_obs = 0
        # Always route through the sampler so empty branches still produce
        # `max_slots` HIDE_Z entries — the env's TerrainRoll length check
        # is strict.
        pos, sz = sample_obstacles_along_path(
            rng, start, goal,
            n_obstacles=n_obs, max_slots=6,
            size_range=(0.4, 0.65), lateral_jitter=1.6,
            heightmap=hm, heightmap_extent=(11.0, 11.0, 0.4),
        )
        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=goal, waypoints=(),
            obstacle_positions=pos, obstacle_sizes=sz,
            heightmap=hm,
        )
    spec.randomize_on_reset = _roll
    return spec


def terrain_RC_full_random(seed: int = 0) -> TerrainSpec:
    """Phase 6 (capstone): uniform over (waypoints × obstacles × terrain).

    By construction this is the distribution the deployed policy will be
    evaluated on. The phase-6 train and the held-out eval draw from the
    same generator with different seeds.

    Per-roll:
        n_waypoints  ~ {0: 0.30, 1: 0.20, 2: 0.20, 3: 0.15, 4: 0.10, 5: 0.05}
        n_obstacles  ~ {0: 0.20, 1-2: 0.20, 3-5: 0.30, 6-8: 0.30}
        hfield scale ~ {flat 0.02: 0.25, mild 0.30: 0.45, big 0.60: 0.30}
        start type   ~ {uniform: 0.60, manoeuvre: 0.40}
    """
    spec = _hfield_template(
        "RC_full_random", max_obstacles=8,
        elevation_z=0.5, hm_seed=seed,
        ground_rgba=(0.60, 0.45, 0.34, 1.0),
    )

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        # start type
        bearing = "maneuver" if rng.uniform() < 0.40 else "uniform"
        start, yaw, goal = sample_start_goal_pair(
            rng, arena_half=11.0, min_separation=6.0, margin=2.5,
            relative_bearing=bearing,
        )

        # waypoint count via the documented distribution.
        r = rng.uniform()
        cum = np.cumsum([0.30, 0.20, 0.20, 0.15, 0.10, 0.05])
        n_wp = int(np.searchsorted(cum, r))  # 0..5
        if n_wp == 0:
            wps = ()
        elif rng.uniform() < 0.5 and n_wp >= 2:
            # arc / slalom mix
            if rng.uniform() < 0.5:
                wps = sample_arc_waypoints(
                    rng, start, goal, n_waypoints=min(n_wp, 3),
                    arc_angle_deg=(30.0, 90.0),
                )
            else:
                wps = sample_slalom_waypoints(
                    rng, start, goal, n_waypoints=n_wp,
                    swing_min=1.4, swing_max=2.8,
                )
        else:
            wps = sample_twisty_waypoints(
                rng, start, goal, n_waypoints=n_wp,
                arena_half=11.0, margin=2.5,
                swing_min=1.4, swing_max=2.8, alternate_sides=True,
            )

        # obstacle count
        r = rng.uniform()
        if r < 0.20:
            n_obs = 0
        elif r < 0.40:
            n_obs = int(rng.integers(1, 3))
        elif r < 0.70:
            n_obs = int(rng.integers(3, 6))
        else:
            n_obs = int(rng.integers(6, 9))

        # terrain scale
        r = rng.uniform()
        hm = generate_heightmap_perlin(
            nrow=48, ncol=48, scale=0.10, octaves=3,
            seed=int(rng.integers(0, 10_000_000)),
        )
        if r < 0.25:
            hm = hm * 0.04
            terrain_z = 0.02
        elif r < 0.70:
            hm = hm * 0.60
            terrain_z = 0.30
        else:
            hm = hm * 1.0
            terrain_z = 0.50  # full elevation_z

        # Always route through the sampler (n=0 returns max_slots HIDE_Z
        # entries) so the env's strict length check passes.
        pos, sz = sample_obstacles_along_path(
            rng, start, goal,
            n_obstacles=n_obs, max_slots=8,
            size_range=(0.4, 0.65),
            lateral_jitter=1.4 + 0.3 * (n_obs / 8.0),
            heightmap=hm, heightmap_extent=(11.0, 11.0, 0.5),
        )

        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=goal, waypoints=wps,
            obstacle_positions=pos, obstacle_sizes=sz,
            heightmap=hm,
        )
    spec.randomize_on_reset = _roll
    return spec


# ---------------------------------------------------------------------------
# scenario_13 integrated curriculum (RC_foundation/navigation/...).
#
# The scenario_11 curriculum had pure single-skill phases (path-only, then
# obstacle-only). The transition from "no obstacles" to "many obstacles"
# wiped path-following — exactly the catastrophic-forgetting failure mode
# CL is supposed to mitigate, but more than EWC alone can recover from.
#
# Design principle here: NO phase trains a single skill in isolation.
# Every phase after the foundation has obstacles AND waypoints in every
# episode. The CL method's job becomes "preserve smoothly improving
# skills" rather than "preserve skill A while task switches to skill B".
# ---------------------------------------------------------------------------


def terrain_RC_foundation(seed: int = 0) -> TerrainSpec:
    """Phase 0 of the integrated curriculum: locomotion + simple paths.

    Within-phase mix:
        50% short goal, 0 waypoints   (2-4 m)  — locomotion bootstrap
        25% medium goal, 0 waypoints  (4-7 m)  — extended driving
        25% short goal, 1 waypoint    (3-6 m total) — light path follow

    Distance-curriculum bootstrap (same trick that worked on RC_locomotion).
    The 25% one-waypoint slice introduces the waypoint mechanic gently so
    the next phase doesn't see it for the first time.
    """
    spec = _flat_template("RC_foundation", max_obstacles=0)

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        r = rng.uniform()
        if r < 0.50:
            dmin, dmax, has_wp = 2.0, 4.0, False
            bearing = "front"
        elif r < 0.75:
            dmin, dmax, has_wp = 4.0, 7.0, False
            bearing = "front" if rng.uniform() < 0.75 else "side"
        else:
            dmin, dmax, has_wp = 3.0, 6.0, True
            bearing = "uniform"
        start, yaw, goal = sample_start_goal_pair(
            rng, arena_half=12.0, min_separation=dmin, max_separation=dmax,
            margin=2.0, relative_bearing=bearing,
        )
        wps = (
            sample_waypoints_between(rng, start, goal, n_waypoints=1, lateral_jitter=0.6)
            if has_wp else ()
        )
        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=goal,
            waypoints=wps,
            obstacle_positions=[], obstacle_sizes=[],
        )
    spec.randomize_on_reset = _roll
    return spec


def terrain_RC_navigation(seed: int = 0) -> TerrainSpec:
    """Phase 1: integrated path+obstacle navigation on flat ground.

    EVERY episode has BOTH waypoints AND obstacles. The "skills together
    from day one" principle — never train obstacle avoidance without
    something to reach, never train path following without something
    to avoid. This is the headline change vs scenario_11.

    Within-phase difficulty mix:
        30%  1 obstacle, 0-1 waypoint, short (3-5 m)
        30%  1-2 obstacles, 1 waypoint, medium (4-7 m)
        25%  2-3 obstacles, 1-2 waypoints, medium (5-8 m)
        15%  3-5 obstacles, 2 waypoints, longer (6-9 m)
    """
    spec = _flat_template("RC_navigation", max_obstacles=5,
                          ground_rgba=(0.58, 0.42, 0.30, 1.0))

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        r = rng.uniform()
        if r < 0.30:
            n_obs, dmin, dmax = 1, 3.0, 5.0
            n_wp = 0 if rng.uniform() < 0.5 else 1
        elif r < 0.60:
            n_obs, dmin, dmax = int(rng.integers(1, 3)), 4.0, 7.0
            n_wp = 1
        elif r < 0.85:
            n_obs, dmin, dmax = int(rng.integers(2, 4)), 5.0, 8.0
            n_wp = int(rng.integers(1, 3))
        else:
            n_obs, dmin, dmax = int(rng.integers(3, 6)), 6.0, 9.0
            n_wp = 2
        start, yaw, goal = sample_start_goal_pair(
            rng, arena_half=12.0, min_separation=dmin, max_separation=dmax,
            margin=2.0,
        )
        if n_wp > 0:
            wps = sample_waypoints_between(
                rng, start, goal, n_waypoints=n_wp, lateral_jitter=0.9,
            )
        else:
            wps = ()
        pos, sz = sample_obstacles_along_path(
            rng, start, goal,
            n_obstacles=n_obs, max_slots=5,
            size_range=(0.4, 0.6), lateral_jitter=1.3,
        )
        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=goal, waypoints=wps,
            obstacle_positions=pos, obstacle_sizes=sz,
        )
    spec.randomize_on_reset = _roll
    return spec


def terrain_RC_navigation_terrain(seed: int = 0) -> TerrainSpec:
    """Phase 2: phase 1 + heightmap. Same skill mix, varied terrain.

    All paths/obstacles like RC_navigation, layered on:
        30% essentially flat   (hfield × 0.04)
        50% gentle hills       (hfield × 0.4)
        20% bigger dunes       (hfield × 0.8)

    Obstacles sit on the local terrain surface.
    """
    spec = _hfield_template(
        "RC_navigation_terrain", max_obstacles=5,
        elevation_z=0.5, hm_seed=seed,
        ground_rgba=(0.62, 0.46, 0.34, 1.0),
    )

    def _roll(rng: np.random.Generator) -> TerrainRoll:
        # Same path+obstacle distribution as RC_navigation.
        r = rng.uniform()
        if r < 0.30:
            n_obs, dmin, dmax = 1, 3.0, 5.0
            n_wp = 0 if rng.uniform() < 0.5 else 1
        elif r < 0.60:
            n_obs, dmin, dmax = int(rng.integers(1, 3)), 4.0, 7.0
            n_wp = 1
        elif r < 0.85:
            n_obs, dmin, dmax = int(rng.integers(2, 4)), 5.0, 8.0
            n_wp = int(rng.integers(1, 3))
        else:
            n_obs, dmin, dmax = int(rng.integers(3, 6)), 6.0, 9.0
            n_wp = 2

        # Terrain scale mix.
        tr = rng.uniform()
        hm = generate_heightmap_perlin(
            nrow=48, ncol=48, scale=0.10, octaves=3,
            seed=int(rng.integers(0, 10_000_000)),
        )
        if tr < 0.30:
            hm = hm * 0.04
        elif tr < 0.80:
            hm = hm * 0.40
        else:
            hm = hm * 0.80

        start, yaw, goal = sample_start_goal_pair(
            rng, arena_half=11.0, min_separation=dmin, max_separation=dmax,
            margin=2.5,
        )
        if n_wp > 0:
            wps = sample_waypoints_between(
                rng, start, goal, n_waypoints=n_wp, lateral_jitter=0.9,
            )
        else:
            wps = ()
        pos, sz = sample_obstacles_along_path(
            rng, start, goal,
            n_obstacles=n_obs, max_slots=5,
            size_range=(0.4, 0.6), lateral_jitter=1.3,
            heightmap=hm, heightmap_extent=(11.0, 11.0, 0.5),
        )
        return TerrainRoll(
            start_pos=start, start_yaw=yaw, goal_pos=goal, waypoints=wps,
            obstacle_positions=pos, obstacle_sizes=sz,
            heightmap=hm,
        )
    spec.randomize_on_reset = _roll
    return spec


TERRAIN_CATALOG: dict[str, TerrainFactory] = {
    "T1_flat": terrain_T1_flat,
    "T1_blocked_arc": terrain_T1_blocked_arc,
    "T1_blocked_arc_hills": terrain_T1_blocked_arc_hills,
    "T2_corridor": terrain_T2_corridor,
    "T3_obstacle_field": terrain_T3_obstacle_field,
    "T4_dunes": terrain_T4_dunes,
    "T6_slope": terrain_T6_slope,
    # Randomized curriculum terrains (used by scenario_10_robust_curriculum):
    "RT_drive_random": terrain_RT_drive_random,
    "RT_with_waypoint": terrain_RT_with_waypoint,
    "RT_with_two_waypoints": terrain_RT_with_two_waypoints,
    "RT_waypoints_arc": terrain_RT_waypoints_arc,
    "RT_waypoint_90": terrain_RT_waypoint_90,
    "RT_waypoint_180": terrain_RT_waypoint_180,
    "RT_slalom": terrain_RT_slalom,
    "RT_one_obstacle": terrain_RT_one_obstacle,
    "RT_obstacle_field": terrain_RT_obstacle_field,
    "RT_dense_obstacles": terrain_RT_dense_obstacles,
    "RT_gentle_hills": terrain_RT_gentle_hills,
    "RT_dunes": terrain_RT_dunes,
    "RT_mixed": terrain_RT_mixed,
    # scenario_11 robust-generalist curriculum (mixed-distribution phases):
    "RC_locomotion":          terrain_RC_locomotion,
    "RC_path_following":      terrain_RC_path_following,
    "RC_obstacle_avoidance":  terrain_RC_obstacle_avoidance,
    "RC_path_and_obstacles":  terrain_RC_path_and_obstacles,
    "RC_terrain":             terrain_RC_terrain,
    "RC_terrain_plus":        terrain_RC_terrain_plus,
    "RC_full_random":         terrain_RC_full_random,
    # scenario_13 integrated curriculum (no-pure-single-skill phases):
    "RC_foundation":          terrain_RC_foundation,
    "RC_navigation":          terrain_RC_navigation,
    "RC_navigation_terrain":  terrain_RC_navigation_terrain,
}


def get_terrain(name: str, seed: int = 0) -> TerrainSpec:
    if name not in TERRAIN_CATALOG:
        raise KeyError(f"Unknown terrain {name!r}. Known: {sorted(TERRAIN_CATALOG)}")
    return TERRAIN_CATALOG[name](seed)
