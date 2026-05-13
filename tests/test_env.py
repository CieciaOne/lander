"""Smoke tests for RoverNavEnv and the terrain catalog."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rover_cl.envs import RoverNavEnv, TERRAIN_CATALOG, compose_scene, get_terrain

pytestmark = pytest.mark.sim


@pytest.mark.parametrize("terrain", sorted(TERRAIN_CATALOG))
def test_terrain_compiles(terrain: str) -> None:
    """Every registered terrain produces valid MJCF that MuJoCo can load."""
    import mujoco
    spec = get_terrain(terrain, seed=0)
    xml = compose_scene(spec)
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.nbody > 5  # rover bodies + obstacles + world
    assert model.nu == 14    # 6 drive + 4 corner-steering + 4 arm actuators


def test_env_reset_step_shapes() -> None:
    env = RoverNavEnv(terrain="T1_flat", max_steps=30, seed=0)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32

    obs2, reward, term, trunc, info = env.step(np.array([0.5, 0.5], dtype=np.float32))
    assert obs2.shape == obs.shape
    assert np.isfinite(reward)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert "distance_to_goal" in info
    assert "is_success" in info
    env.close()


def test_env_truncates_at_max_steps() -> None:
    env = RoverNavEnv(terrain="T1_flat", max_steps=5, seed=0)
    env.reset()
    for i in range(5):
        _, _, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
        if term or trunc:
            break
    assert trunc is True
    assert "episode" in info
    env.close()


def test_env_success_when_spawned_near_goal() -> None:
    """Manually teleport the rover to the goal; episode must terminate as success."""
    env = RoverNavEnv(terrain="T1_flat", max_steps=30, seed=0)
    obs, info = env.reset()
    gx, gy = env.terrain.goal_pos
    env._data.qpos[0] = gx
    env._data.qpos[1] = gy
    import mujoco
    mujoco.mj_forward(env._model, env._data)
    env._prev_dist = 0.0
    succeeded = False
    for _ in range(20):
        _, _, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
        if info["is_success"]:
            succeeded = True
            break
        if term or trunc:
            break
    assert succeeded, "rover at goal should yield is_success=True within a few steps"
    env.close()


def test_rocker_bogie_differential() -> None:
    """The rocker differential must counter-rotate the two rockers.

    Real-rover physics: a bump under the right-front wheel pushes the right rocker
    up; the differential bar/gearbox forces the left rocker down by the same angle
    so the chassis ends up at the average pitch instead of the full bump height.
    Verified here as `rocker_right + rocker_left ≈ 0` under an asymmetric obstacle.
    """
    import mujoco
    from rover_cl.envs.terrains import Obstacle, compose_scene, get_terrain

    t = get_terrain("T1_flat", seed=0)
    # The right-front wheel center is at world (0.81, 0.65, 0.25) after settling;
    # put a box under it so only that wheel rides up.
    t.obstacles = [Obstacle(pos=(0.81, 0.65, 0.15), size=(0.30, 0.30, 0.15))]
    m = mujoco.MjModel.from_xml_string(compose_scene(t))
    d = mujoco.MjData(m)
    # spawn the chassis above the box so the rover lands on it cleanly instead of
    # sliding off from a low default spawn
    d.qpos[2] = 1.3
    for _ in range(600):
        mujoco.mj_step(m, d)

    def jval(name: str) -> float:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        return float(d.qpos[m.jnt_qposadr[jid]])

    rr = jval("rocker_right_joint")
    rl = jval("rocker_left_joint")
    assert abs(rr) > 0.05, f"right rocker barely moved ({rr:.3f}); is the obstacle in the right spot?"
    # Constraint residual must be tiny (default MuJoCo solver tolerance).
    assert abs(rr + rl) < 0.01, f"differential broken: rocker_right + rocker_left = {rr+rl:+.4f}"


def test_env_drives_forward_when_actuated() -> None:
    """Two seconds of forward drive should reduce distance to goal."""
    env = RoverNavEnv(terrain="T1_flat", max_steps=60, seed=0)
    env.reset()
    d0 = env._prev_dist
    for _ in range(40):
        _, _, term, trunc, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        if term or trunc:
            break
    d1 = info["distance_to_goal"]
    assert d1 < d0, f"expected rover to move closer to goal, but {d1:.2f} >= {d0:.2f}"
    env.close()
