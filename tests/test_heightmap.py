"""Tests for the MuJoCo HField (heightmap) terrain support."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco

from rover_cl.envs.terrains import (
    compile_scene,
    generate_heightmap_perlin,
    generate_heightmap_slope,
    get_terrain,
)

pytestmark = pytest.mark.sim


def _hfield_surface_z(spec, x: float, y: float) -> float:
    """Sample the heightmap z at a world (x, y) point using nearest-neighbour lookup."""
    hm = spec.heightmap
    rx, ry, ez = spec.heightmap_extent
    nrow, ncol = hm.shape
    j = int(round(np.clip((x + rx) / (2 * rx) * (ncol - 1), 0, ncol - 1)))
    i = int(round(np.clip((y + ry) / (2 * ry) * (nrow - 1), 0, nrow - 1)))
    return float(hm[i, j]) * ez


def test_perlin_deterministic() -> None:
    a = generate_heightmap_perlin(nrow=32, ncol=32, seed=42)
    b = generate_heightmap_perlin(nrow=32, ncol=32, seed=42)
    np.testing.assert_array_equal(a, b)


def test_perlin_range_and_shape() -> None:
    hm = generate_heightmap_perlin(nrow=48, ncol=48, seed=1)
    assert hm.shape == (48, 48)
    assert hm.dtype in (np.float32, np.float64)
    eps = 1e-6
    assert float(hm.min()) >= -eps
    assert float(hm.max()) <= 1.0 + eps


def test_slope_monotonic() -> None:
    hm = generate_heightmap_slope(nrow=32, ncol=32, grade=0.10, axis="y")
    # Monotonically non-decreasing along axis 0 (rows).
    row_means = hm.mean(axis=1)
    diffs = np.diff(row_means)
    assert (diffs >= -1e-7).all(), f"slope not monotonic: min diff = {diffs.min()}"
    # First and last rows differ significantly.
    assert row_means[-1] - row_means[0] > 0.5


def test_T4_dunes_compiles_and_settles() -> None:
    spec = get_terrain("T4_dunes", seed=0)
    model, data = compile_scene(spec)
    assert model.nu == 14
    assert model.nhfield == 1
    assert model.hfield_data.size == spec.heightmap.size

    # Drop the rover above the surface at the start point and let it settle.
    surf_z = _hfield_surface_z(spec, *spec.start_pos)
    mujoco.mj_resetData(model, data)
    data.qpos[0] = spec.start_pos[0]
    data.qpos[1] = spec.start_pos[1]
    data.qpos[2] = surf_z + 1.5
    data.qpos[3] = 1.0  # identity quaternion (w, x, y, z) = (1, 0, 0, 0)
    data.qpos[4:7] = 0.0
    data.ctrl[:] = 0.0
    for _ in range(600):
        mujoco.mj_step(model, data)

    qw, qx, qy, qz = (
        float(data.qpos[3]),
        float(data.qpos[4]),
        float(data.qpos[5]),
        float(data.qpos[6]),
    )
    # Quaternion w > 0.9 means tilt < ~52 degrees; upright.
    assert qw > 0.9, f"rover did not settle upright (qw={qw:.3f})"
    assert np.isfinite(data.qpos[:3]).all()
    settled_z = float(data.qpos[2])
    assert -5.0 < settled_z < 5.0, f"rover at implausible height z={settled_z}"


def test_T6_slope_rover_rolls_downhill() -> None:
    spec = get_terrain("T6_slope", seed=0)
    model, data = compile_scene(spec)

    # Spawn at high Y (the top of the slope; slope rises in +Y direction)
    # and watch the rover drift toward -Y under gravity.
    start_x, start_y = 0.0, 10.0
    surf_z = _hfield_surface_z(spec, start_x, start_y)
    mujoco.mj_resetData(model, data)
    data.qpos[0] = start_x
    data.qpos[1] = start_y
    data.qpos[2] = surf_z + 1.5
    data.qpos[3] = 1.0
    data.qpos[4:7] = 0.0
    data.ctrl[:] = 0.0

    # The rover's velocity-controlled wheel actuators actively brake at ctrl=0,
    # so pure "rolling" on the gentle slope is suppressed. Run a longer rollout
    # to give settling + drift enough wall time to register a clear direction.
    for _ in range(1500):
        mujoco.mj_step(model, data)

    dy = float(data.qpos[1] - start_y)
    # Must drift downhill (toward -Y, away from start), with at least a few cm
    # of net motion to distinguish from numerical jitter.
    assert dy < -0.05, f"rover did not roll downhill: dy={dy:.4f}"


def test_T1_flat_still_compiles() -> None:
    """Sanity: the legacy plane code path is unaffected by the hfield refactor."""
    spec = get_terrain("T1_flat", seed=0)
    model, data = compile_scene(spec)
    assert model.nhfield == 0
    assert model.nu == 14
    # And it has the plane ground geom.
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    assert ground_id >= 0
    assert int(model.geom_type[ground_id]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
