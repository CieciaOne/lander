"""Rover feature demo viewer + trained-policy replay.

This script has two modes:

1. **Scripted demo (default).** Cycles through every active feature of the rover
   so you can see them working:

    1. Settle
    2. Drive forward (straight)
    3. Ackermann turn left (smooth arc)
    4. Spin in place (corner-tangent steering)
    5. Ackermann turn right
    6. Arm deploy / sweep / stow
    7. Sensors readout (lidar fan + tool-tip + IMU) printed each phase

   All control inputs go through a slew-rate limiter, so commanded transitions
   are smooth instead of step changes — the rover doesn't jerk between phases.

2. **Trained-policy replay (`--policy <path>`).** Loads an SB3 PPO checkpoint
   (`stable_baselines3.PPO.load`) and drives a `RoverNavEnv` with it inside
   the viewer. Use `--terrain-name <id>` to pick a terrain from the catalog
   (`T1_flat`, `T2_corridor`, `T3_obstacle_field`) instead of an XML file.
   Episode summaries print on each `terminated/truncated` (no per-phase lidar).

The viewer uses a TRACKING camera mounted on `base_link`: the camera follows
the rover's position but you can still orbit/zoom/elevate around it with the
mouse. Switch to the on-rover `chase` or `navcam` cameras with TAB.

Run from the project root with the venv active:

    source .venv/bin/activate
    # macOS:
    mjpython scripts/visualize_rover.py
    mjpython scripts/visualize_rover.py --policy results/.../final.zip --terrain-name T3_obstacle_field
    # Linux:
    python   scripts/visualize_rover.py
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TERRAIN = PROJECT_ROOT / "assets" / "terrain_flat.xml"

# Make sure `src/` is importable when this script runs from the project root.
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Geometry constants
WHEEL_RADIUS = 0.25
CORNER_RIGHT_FRONT = (0.65, 0.65)
CORNER_RIGHT_REAR  = (0.65, -0.75)
CORNER_LEFT_FRONT  = (-0.65, 0.65)
CORNER_LEFT_REAR   = (-0.65, -0.75)
MIDDLE_HALF_TRACK  = 0.65

# ctrl indices (see assets/rover.xml). nu=14.
#   0..5  drive_{R|L}_{front|middle|rear}
#   6..9  steer_{R|L}_{front|rear}   (R_front, R_rear, L_front, L_rear)
#  10..13 arm_yaw, arm_shoulder, arm_elbow, arm_wrist
N_CTRL = 14

# Sign convention: rover "forward" is the +Y direction (front wheels and mast).
# A positive `wheel_ctrl` produces wheel motion in -Y (because of the wheel axle
# orientation), so to drive +Y we use NEGATIVE wheel ctrl. The helpers below
# expose user-facing semantics (forward_vel > 0 = move forward) and negate
# internally.

# Per-channel slew rates (units/sec). Generous enough that the rover feels
# responsive but not snappy.
DRIVE_RATE = 4.0    # rad/s per sec
STEER_RATE = 1.2    # rad/s per sec
ARM_RATE   = 1.2    # rad/s per sec
RATES = (
    [DRIVE_RATE] * 6
    + [STEER_RATE] * 4
    + [ARM_RATE] * 4
)


# --------------------------------------------------------------------------- helpers

def drive_straight(target: np.ndarray, forward_vel: float) -> None:
    """forward_vel > 0 -> rover moves +Y (its forward)."""
    target[6:10] = 0.0
    target[0:6] = -forward_vel / WHEEL_RADIUS


def drive_ackermann(target: np.ndarray, forward_vel: float, steer_angle: float) -> None:
    """Forward drive plus 4-wheel Ackermann counter-steer.

    forward_vel > 0 -> move forward (+Y).
    steer_angle  > 0 -> turn right (CW yaw); < 0 -> turn left.
    """
    target[0:6] = -forward_vel / WHEEL_RADIUS
    target[6] = -steer_angle   # FR steer
    target[8] = -steer_angle   # FL steer  (fronts: opposite-sign of steer for forward-right convention)
    target[7] = +steer_angle   # RR steer  (rears: counter-steer)
    target[9] = +steer_angle   # RL steer


def drive_spin_in_place(target: np.ndarray, omega: float) -> None:
    """omega > 0 -> CCW spin (positive yaw rate).

    Each wheel is commanded at the angular velocity that matches its circular
    motion about the rover center. The middle wheels are stuck at δ=0, so they
    can only roll in Y; we command them at the Y-velocity their center actually
    follows during the spin. Setting middle ctrl=0 instead would make the
    velocity actuator actively brake them, generating big lateral friction
    forces that pump the bogie up. Commanding the matching velocity keeps the
    bogie quiet — the only slip left is the small X-component of the tangent.
    """
    delta_front = np.arctan2(abs(CORNER_RIGHT_FRONT[1]), abs(CORNER_RIGHT_FRONT[0]))  # ≈ 45°
    delta_rear  = np.arctan2(abs(CORNER_RIGHT_REAR[1]),  abs(CORNER_RIGHT_REAR[0]))   # ≈ 49°
    r_front = float(np.hypot(*CORNER_RIGHT_FRONT))
    r_rear  = float(np.hypot(*CORNER_RIGHT_REAR))
    s = 1.0 if omega >= 0 else -1.0
    w_front = r_front * abs(omega) / WHEEL_RADIUS
    w_rear  = r_rear  * abs(omega) / WHEEL_RADIUS
    # Middle-wheel velocity = ω × |x_pos|; commanded wheel ω = that / R. With low
    # drive gain this avoids actively braking the wheel (which would scrub).
    w_mid = MIDDLE_HALF_TRACK * abs(omega) / WHEEL_RADIUS

    target[6] = +s * delta_front
    target[7] = -s * delta_rear
    target[8] = -s * delta_front
    target[9] = +s * delta_rear

    target[0] = -s * w_front   # drive_right_front
    target[1] = -s * w_mid     # drive_right_middle (matches circular Y-velocity)
    target[2] = -s * w_rear    # drive_right_rear
    target[3] = +s * w_front   # drive_left_front
    target[4] = +s * w_mid     # drive_left_middle
    target[5] = +s * w_rear    # drive_left_rear


def stop(target: np.ndarray) -> None:
    target[0:10] = 0.0


def arm_pose(target: np.ndarray, yaw: float, shoulder: float, elbow: float, wrist: float) -> None:
    target[10] = yaw
    target[11] = shoulder
    target[12] = elbow
    target[13] = wrist


# --------------------------------------------------------------------------- demo schedule

@dataclass
class Phase:
    label: str
    duration: float
    apply: callable  # (target: np.ndarray) -> None


def make_schedule() -> list[Phase]:
    def settle(t): stop(t); arm_pose(t, 0, 0, 0, 0)
    def forward(t): drive_straight(t, 0.40); arm_pose(t, 0, 0, 0, 0)
    def ack_left(t): drive_ackermann(t, 0.30, -np.radians(22)); arm_pose(t, 0, 0, 0, 0)
    def spin_ccw(t): drive_spin_in_place(t, 0.15); arm_pose(t, 0, 0, 0, 0)
    def ack_right(t): drive_ackermann(t, 0.30, +np.radians(22)); arm_pose(t, 0, 0, 0, 0)
    # Arm sequence sweeps each joint through its full range one at a time, then
    # combines them into a "deploy" reach toward the ground and stows back.
    def arm_yaw_right(t):   stop(t); arm_pose(t, +np.radians(85), 0, 0, 0)
    def arm_yaw_left(t):    stop(t); arm_pose(t, -np.radians(85), 0, 0, 0)
    def arm_shoulder_up(t): stop(t); arm_pose(t, 0, +np.radians(80), 0, 0)
    def arm_shoulder_dn(t): stop(t); arm_pose(t, 0, -np.radians(80), 0, 0)
    def arm_elbow_fold(t):  stop(t); arm_pose(t, 0, -np.radians(40), +np.radians(140), 0)
    def arm_wrist_roll(t):  stop(t); arm_pose(t, 0, -np.radians(40), +np.radians(140), +np.radians(140))
    def arm_deploy(t):
        stop(t)
        arm_pose(t, np.radians(20), -np.radians(70), np.radians(110), np.radians(45))
    def arm_stow(t): stop(t); arm_pose(t, 0, 0, 0, 0)
    return [
        Phase("settle",            3.0, settle),
        Phase("forward (straight)",5.0, forward),
        Phase("Ackermann LEFT",    5.0, ack_left),
        Phase("spin in place CCW", 3.5, spin_ccw),
        Phase("Ackermann RIGHT",   5.0, ack_right),
        Phase("stop",              1.5, lambda t: (stop(t), arm_pose(t, 0, 0, 0, 0))),
        Phase("arm: yaw RIGHT",    3.0, arm_yaw_right),
        Phase("arm: yaw LEFT",     3.0, arm_yaw_left),
        Phase("arm: shoulder UP",  3.0, arm_shoulder_up),
        Phase("arm: shoulder DOWN",3.0, arm_shoulder_dn),
        Phase("arm: elbow FOLD",   3.0, arm_elbow_fold),
        Phase("arm: wrist ROLL",   3.0, arm_wrist_roll),
        Phase("arm: deploy",       3.5, arm_deploy),
        Phase("arm: stow",         3.0, arm_stow),
    ]


# --------------------------------------------------------------------------- sensors

LIDAR_NAMES = ["lidar_m60", "lidar_m30", "lidar_0", "lidar_p30", "lidar_p60"]


def read_lidar(model: mujoco.MjModel, data: mujoco.MjData) -> list[float]:
    out = []
    for nm in LIDAR_NAMES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, nm)
        out.append(float(data.sensordata[model.sensor_adr[sid]]))
    return out


def fmt_lidar(rs: list[float]) -> str:
    return "  ".join(f"{a:>4s}: " + (f"{r:5.2f}" if r >= 0 else "  ∞  ")
                     for a, r in zip(["-60", "-30", "  0", "+30", "+60"], rs))


# --------------------------------------------------------------------------- mac guard

def _guard_macos_mjpython() -> None:
    if sys.platform != "darwin":
        return
    if getattr(mujoco.viewer, "_MJPYTHON", None) is not None:
        return
    print(
        "ERROR: on macOS the MuJoCo viewer must be launched with `mjpython`.\n"
        "       Try:  mjpython scripts/visualize_rover.py",
        file=sys.stderr,
    )
    sys.exit(1)


# --------------------------------------------------------------------------- policy replay helpers

def build_env_from_terrain_name(name: str):
    """Build a `RoverNavEnv` from a terrain catalog id (e.g. "T1_flat").

    Raises `KeyError` if the terrain name is not in the catalog.
    """
    # Imported lazily so the scripted-demo mode doesn't pay the gymnasium/SB3
    # import cost when the user just wants to watch the feature demo.
    from rover_cl.envs.nav import RoverNavEnv

    return RoverNavEnv(terrain=name)


def policy_step(env, policy, obs):
    """One env step driven by the policy.

    Returns ``(next_obs, terminated, truncated, info)``. Reward and the
    deterministic action used are tucked into ``info`` under the keys
    ``"_reward"`` and ``"_action"`` so callers (and tests) can introspect.
    """
    action, _ = policy.predict(obs, deterministic=True)
    next_obs, reward, terminated, truncated, info = env.step(action)
    info = dict(info)
    info["_reward"] = float(reward)
    info["_action"] = action
    return next_obs, terminated, truncated, info


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Rover viewer. Default: scripted feature demo. "
            "With --policy: replay a trained SB3 PPO checkpoint inside a RoverNavEnv."
        ),
    )
    ap.add_argument("--terrain", type=Path, default=None,
                    help="Path to a static terrain XML (default assets/terrain_flat.xml). "
                         "Mutually exclusive with --terrain-name.")
    ap.add_argument("--terrain-name", type=str, default=None,
                    help="Terrain catalog id (T1_flat / T2_corridor / T3_obstacle_field). "
                         "Mutually exclusive with --terrain.")
    ap.add_argument("--policy", type=Path, default=None,
                    help="Path to an SB3 PPO .zip checkpoint. Enables policy-replay mode "
                         "(skips the scripted demo).")
    ap.add_argument("--realtime", action="store_true", default=True)
    ap.add_argument("--fast", dest="realtime", action="store_false")
    ap.add_argument("--free-camera", action="store_true",
                    help="Start with a free (untracked) camera instead of tracking the rover.")
    args = ap.parse_args()

    if args.terrain is not None and args.terrain_name is not None:
        print("ERROR: pass only one of --terrain or --terrain-name, not both.",
              file=sys.stderr)
        sys.exit(2)

    _guard_macos_mjpython()

    if args.policy is not None:
        _run_policy_viewer(args)
    else:
        _run_demo_viewer(args)


def _run_demo_viewer(args) -> None:
    terrain_path = args.terrain or DEFAULT_TERRAIN
    if args.terrain_name is not None:
        # Build the scene from the catalog instead of loading a static XML.
        from rover_cl.envs.terrains import compose_scene, get_terrain
        spec = get_terrain(args.terrain_name)
        model = mujoco.MjModel.from_xml_string(compose_scene(spec))
        terrain_label = args.terrain_name
    else:
        model = mujoco.MjModel.from_xml_path(str(terrain_path))
        terrain_label = str(terrain_path)
    data = mujoco.MjData(model)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    print(f"loaded {terrain_label}")
    print(f"  nq={model.nq} nv={model.nv} nu={model.nu} ncam={model.ncam} nsensor={model.nsensor}")
    print("Demo phases:")
    schedule = make_schedule()
    cycle_len = sum(p.duration for p in schedule)
    t_off = 0.0
    for p in schedule:
        print(f"  [{t_off:5.1f}–{t_off + p.duration:5.1f} s] {p.label}")
        t_off += p.duration
    print(f"Cycle length: {cycle_len:.1f} s.  Lidar readings print on every phase change.")
    print("TAB to cycle cameras; mouse to orbit/zoom; ESC to quit.\n")

    target_ctrl = np.zeros(N_CTRL)
    rates = np.array(RATES)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        _set_camera(viewer, base_id, free=args.free_camera)

        last_phase_label = ""
        while viewer.is_running():
            step_start = time.time()
            cycle_t = data.time % cycle_len
            # find current phase + commanded targets
            acc = 0.0
            current_phase = schedule[-1]
            for p in schedule:
                if cycle_t < acc + p.duration:
                    current_phase = p
                    break
                acc += p.duration
            current_phase.apply(target_ctrl)

            # slew-rate-limited ctrl update
            dt = model.opt.timestep
            max_step = rates * dt
            delta = target_ctrl - data.ctrl
            delta_clip = np.clip(delta, -max_step, max_step)
            data.ctrl[:] = data.ctrl + delta_clip

            if current_phase.label != last_phase_label:
                rs = read_lidar(model, data)
                yaw_vel = float(data.qvel[5])
                pos = data.qpos[0:2]
                print(f"[t={data.time:6.1f}s] {current_phase.label:25s} "
                      f"pos=({pos[0]:+.2f},{pos[1]:+.2f}) yaw_rate={yaw_vel:+.2f}rad/s  "
                      f"lidar [{fmt_lidar(rs)}]")
                last_phase_label = current_phase.label

            mujoco.mj_step(model, data)
            viewer.sync()
            if args.realtime:
                slack = model.opt.timestep - (time.time() - step_start)
                if slack > 0:
                    time.sleep(slack)


def _set_camera(viewer, base_id: int, *, free: bool) -> None:
    if free:
        viewer.cam.distance = 6.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 110
    else:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = base_id
        viewer.cam.distance = 5.0
        viewer.cam.elevation = -18
        viewer.cam.azimuth = 110


def _run_policy_viewer(args) -> None:
    # SB3 is only needed in policy mode; import lazily.
    from stable_baselines3 import PPO

    if args.terrain_name is not None:
        env = build_env_from_terrain_name(args.terrain_name)
        terrain_label = args.terrain_name
    elif args.terrain is not None:
        # Build a RoverNavEnv-shaped wrapper around a custom XML is out of scope;
        # for policy replay we require a catalog terrain so obs/action match training.
        print("ERROR: --policy currently requires --terrain-name (a catalog id), "
              "not a static --terrain XML, so the obs/action match how the policy was trained.",
              file=sys.stderr)
        sys.exit(2)
    else:
        env = build_env_from_terrain_name("T1_flat")
        terrain_label = "T1_flat"

    policy = PPO.load(str(args.policy))
    print(f"loaded policy {args.policy}, terrain {terrain_label}")

    model = env._model
    data = env._data
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    obs, _ = env.reset()
    episode_idx = 0
    ep_reward = 0.0
    ep_steps = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        _set_camera(viewer, base_id, free=args.free_camera)

        while viewer.is_running():
            step_start = time.time()
            obs, terminated, truncated, info = policy_step(env, policy, obs)
            ep_reward += float(info.get("_reward", 0.0))
            ep_steps += 1
            viewer.sync()

            if terminated or truncated:
                success = bool(info.get("is_success", False))
                print(f"[ep {episode_idx}] reward={ep_reward:.1f} steps={ep_steps} "
                      f"success={success}")
                episode_idx += 1
                ep_reward = 0.0
                ep_steps = 0
                obs, _ = env.reset()

            if args.realtime:
                # RoverNavEnv steps `control_decimation` physics steps per env step.
                target_dt = model.opt.timestep * getattr(env, "control_decimation", 1)
                slack = target_dt - (time.time() - step_start)
                if slack > 0:
                    time.sleep(slack)


if __name__ == "__main__":
    main()
