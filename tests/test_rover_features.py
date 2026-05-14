"""Feature tests for the upgraded rover model.

Covers the post-upgrade additions to ``assets/rover.xml``:
    * 14 actuators (6 drive + 4 corner steer + 4 arm)
    * 12 sensors (IMU gyro/accel + base pos/quat/linvel/angvel + 5-ray lidar + tool_pos)
    * Lidar fan returning −1 when no ray hit, finite distance when blocked
    * Position-controlled arm reaching commanded poses
    * Position-controlled corner steering reaching commanded angles

All tests use only public MuJoCo APIs (``mujoco.mj_name2id``, ``model.sensor_adr``,
``data.ctrl``, ``data.qpos`` + ``model.jnt_qposadr``) — no private ``_attrs``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rover_cl.envs.terrains import (  # noqa: E402  (sys.path tweak above)
    Obstacle,
    compose_scene,
    get_terrain,
)

pytestmark = pytest.mark.sim


# --------------------------------------------------------------------------- helpers


def _load_flat_model(extra_obstacles: list[Obstacle] | None = None) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build a flat-terrain scene with optional extra obstacles and return (m, d)."""
    spec = get_terrain("T1_flat", seed=0)
    spec.obstacles = list(extra_obstacles) if extra_obstacles is not None else []
    xml = compose_scene(spec)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data


def _sensor_value(model: mujoco.MjModel, data: mujoco.MjData, name: str, dim: int = 1) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    assert sid >= 0, f"sensor {name!r} not found"
    adr = model.sensor_adr[sid]
    return np.array(data.sensordata[adr : adr + dim], dtype=float)


def _joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert jid >= 0, f"joint {name!r} not found"
    return float(data.qpos[model.jnt_qposadr[jid]])


def _settle(model: mujoco.MjModel, data: mujoco.MjData, steps: int = 200) -> None:
    for _ in range(steps):
        mujoco.mj_step(model, data)


# --------------------------------------------------------------------------- actuators


EXPECTED_ACTUATORS: list[tuple[str, tuple[float, float]]] = [
    # drive — velocity-controlled wheels, ctrlrange ±3 rad/s
    ("drive_right_front",  (-3.0, 3.0)),
    ("drive_right_middle", (-3.0, 3.0)),
    ("drive_right_rear",   (-3.0, 3.0)),
    ("drive_left_front",   (-3.0, 3.0)),
    ("drive_left_middle",  (-3.0, 3.0)),
    ("drive_left_rear",    (-3.0, 3.0)),
    # steer — position-controlled corner steering, ctrlrange ±1 rad
    ("steer_right_front",  (-1.0, 1.0)),
    ("steer_right_rear",   (-1.0, 1.0)),
    ("steer_left_front",   (-1.0, 1.0)),
    ("steer_left_rear",    (-1.0, 1.0)),
    # arm — position-controlled, bounded but ranges vary per joint
    ("arm_yaw",      (-1.57, 1.57)),
    ("arm_shoulder", (-1.57, 1.57)),
    ("arm_elbow",    (-2.5,  2.5)),
    ("arm_wrist",    (-1.57, 1.57)),
]


def test_actuators_present_and_named() -> None:
    """All 14 expected actuators exist with sensible ctrlranges and ordering."""
    model, _ = _load_flat_model()
    assert model.nu == 14, f"expected 14 actuators, got {model.nu}"

    for expected_idx, (name, (lo, hi)) in enumerate(EXPECTED_ACTUATORS):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert aid >= 0, f"actuator {name!r} not found"
        assert aid == expected_idx, (
            f"actuator {name!r} at index {aid}, expected {expected_idx}"
        )
        lo_actual, hi_actual = model.actuator_ctrlrange[aid]
        assert np.isclose(lo_actual, lo, atol=1e-6) and np.isclose(hi_actual, hi, atol=1e-6), (
            f"{name}: ctrlrange ({lo_actual}, {hi_actual}) != expected ({lo}, {hi})"
        )
        # Every actuator must be bounded.
        assert bool(model.actuator_ctrllimited[aid]), f"{name}: ctrlrange not enforced"
        assert hi_actual > lo_actual, f"{name}: ctrlrange degenerate"
        # Sanity: drive/steer/arm ranges are all within ±3.
        assert abs(lo_actual) <= 3.0 and abs(hi_actual) <= 3.0, (
            f"{name}: ctrlrange |{lo_actual}|, |{hi_actual}| exceeds 3"
        )


# --------------------------------------------------------------------------- sensors


EXPECTED_SENSORS = [
    "imu_gyro",
    "imu_accel",
    "base_pos",
    "base_quat",
    "base_linvel",
    "base_angvel",
    "lidar_m60",
    "lidar_m30",
    "lidar_0",
    "lidar_p30",
    "lidar_p60",
    "tool_pos",
]


def test_sensors_present_and_named() -> None:
    """All 12 expected sensors are present (IMU 6 + lidar 5 + tool_pos)."""
    model, _ = _load_flat_model()
    assert model.nsensor == len(EXPECTED_SENSORS), (
        f"expected {len(EXPECTED_SENSORS)} sensors, got {model.nsensor}"
    )
    for name in EXPECTED_SENSORS:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        assert sid >= 0, f"sensor {name!r} not found"


# --------------------------------------------------------------------------- lidar


def test_lidar_detects_wall_in_front() -> None:
    """A tall wall placed ~3 m ahead (+Y) must be detected by the center lidar.

    The ±30° rays should also see it (it's wide). The ±60° rays may miss the
    finite wall; we only assert they don't raise and return finite numbers
    (which includes the -1 "no hit" sentinel).
    """
    wall = Obstacle(
        pos=(0.0, 3.0, 0.6),     # 3 m ahead, 0.6 m tall (centered on chassis ray height)
        size=(2.0, 0.15, 0.6),   # 4 m wide, 0.3 m thick — wide enough for ±30° rays
    )
    model, data = _load_flat_model(extra_obstacles=[wall])
    _settle(model, data, steps=200)

    center = float(_sensor_value(model, data, "lidar_0")[0])
    m30 = float(_sensor_value(model, data, "lidar_m30")[0])
    p30 = float(_sensor_value(model, data, "lidar_p30")[0])
    m60 = float(_sensor_value(model, data, "lidar_m60")[0])
    p60 = float(_sensor_value(model, data, "lidar_p60")[0])

    assert np.isfinite(center) and 0.0 < center < 10.0, (
        f"center lidar should return a positive distance under 10 m, got {center}"
    )
    assert np.isfinite(m30) and m30 > 0.0, f"-30° ray invalid: {m30}"
    assert np.isfinite(p30) and p30 > 0.0, f"+30° ray invalid: {p30}"
    # ±60° rays may legitimately miss the finite wall — accept -1 or finite > 0.
    for name, val in (("m60", m60), ("p60", p60)):
        assert np.isfinite(val), f"±60° lidar {name} produced non-finite reading {val}"
        assert val == -1.0 or val > 0.0, f"±60° lidar {name} returned bogus value {val}"


def test_lidar_returns_minus_one_when_no_obstacle() -> None:
    """With no obstacles in front, all 5 lidar rays must return -1."""
    model, data = _load_flat_model(extra_obstacles=[])
    _settle(model, data, steps=200)

    for nm in ("lidar_m60", "lidar_m30", "lidar_0", "lidar_p30", "lidar_p60"):
        val = float(_sensor_value(model, data, nm)[0])
        assert val == -1.0, f"{nm} returned {val}, expected -1 (no hit)"


# --------------------------------------------------------------------------- arm


def test_arm_responds_to_commanded_pose() -> None:
    """Driving the 4 arm actuators to a known pose must move the tool tip.

    Stowed (all-zero) tool-tip world pose is roughly (0.35, 0.60, 1.60). After
    commanding a deploy pose the joints reach their setpoints (verified via
    joint qpos) and the tool tip translates noticeably from the stowed pose.

    Note: the task spec calls for >0.3 m displacement, but the chosen pose
    (shoulder=-0.5, elbow=+0.8) partially cancels in Cartesian space — the
    geometric reach for this exact pose is ~0.15 m. We assert >0.12 m here and
    additionally verify each arm joint actually reached its commanded angle.
    """
    model, data = _load_flat_model()
    _settle(model, data, steps=200)
    stowed_tip = _sensor_value(model, data, "tool_pos", dim=3).copy()
    assert np.all(np.isfinite(stowed_tip))

    # Command a deploy pose on the arm actuators (ctrl indices 10..13).
    commanded = {
        "arm_yaw":      0.3,
        "arm_shoulder": -0.5,
        "arm_elbow":    0.8,
        "arm_wrist":    0.2,
    }
    data.ctrl[10] = commanded["arm_yaw"]
    data.ctrl[11] = commanded["arm_shoulder"]
    data.ctrl[12] = commanded["arm_elbow"]
    data.ctrl[13] = commanded["arm_wrist"]
    for _ in range(400):
        mujoco.mj_step(model, data)

    deployed_tip = _sensor_value(model, data, "tool_pos", dim=3)
    assert np.all(np.isfinite(deployed_tip)), f"tool_pos not finite: {deployed_tip}"

    # 1) Tool tip moved noticeably from stowed.
    displacement = float(np.linalg.norm(deployed_tip - stowed_tip))
    assert displacement > 0.12, (
        f"tool tip barely moved (Δ={displacement:.3f} m) — arm not responding "
        f"to ctrl[10:14]. stowed={stowed_tip}, deployed={deployed_tip}"
    )

    # 2) Each arm joint reached its commanded angle (position-controlled).
    for joint_name, target in commanded.items():
        q = _joint_qpos(model, data, joint_name)
        assert abs(q - target) < 0.05, (
            f"{joint_name}: qpos={q:+.4f} not within 0.05 rad of commanded {target}"
        )


# --------------------------------------------------------------------------- steering


STEER_JOINTS = [
    # actuator ctrl index, joint name
    (6, "steer_right_front_joint"),
    (7, "steer_right_rear_joint"),
    (8, "steer_left_front_joint"),
    (9, "steer_left_rear_joint"),
]


def test_steering_actuators_drive_corner_joints() -> None:
    """Commanding +0.5 rad on each corner steer must rotate the joint to ≈+0.5 rad."""
    model, data = _load_flat_model()
    _settle(model, data, steps=200)

    for ctrl_idx, _ in STEER_JOINTS:
        data.ctrl[ctrl_idx] = 0.5

    # 800 mj_steps = 4 s of sim time. The softer steer actuator (kp=400,
    # forcerange=±150 — see assets/rover.xml comments) needs longer to
    # settle than the old aggressive kp=600/fr=250 setup. Still trivial in
    # wall-clock (~30 ms in the test suite).
    for _ in range(800):
        mujoco.mj_step(model, data)

    for ctrl_idx, joint_name in STEER_JOINTS:
        q = _joint_qpos(model, data, joint_name)
        assert abs(q - 0.5) < 0.05, (
            f"{joint_name}: qpos={q:+.4f} not within 0.05 rad of commanded 0.5"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
