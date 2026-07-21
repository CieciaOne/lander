"""Cycle through every phase's trained policy + terrain, one viewer per phase.

Walks the per-phase checkpoints under a results dir
(`results/<scenario>/<method>/seed_<N>/`) in numerical phase order and opens a
MuJoCo passive viewer for each. The viewer auto-closes after
`--seconds-per-phase` seconds and the next phase opens. Close any window
early (Cmd+Q / window close button) to skip to the next phase. Press Ctrl+C
in the terminal to abort the whole sequence.

Examples
--------
    # macOS — note `mjpython`
    mjpython scripts/visualize_all_phases.py \\
        results/scenario_10_robust_curriculum/ewc/seed_0

    # Linger 60 s on each phase instead of the default 30
    mjpython scripts/visualize_all_phases.py \\
        results/scenario_10_robust_curriculum/ewc/seed_0 \\
        --seconds-per-phase 60

    # Pin to a specific phase range
    mjpython scripts/visualize_all_phases.py \\
        results/scenario_10_robust_curriculum/ewc/seed_0 \\
        --phases 3-5

Linux: same commands but `python scripts/...` (no `mjpython`).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Reuse the per-phase viewer helpers from visualize_rover.py instead of
# duplicating the env wiring + camera setup.
from visualize_rover import (
    _guard_macos_mjpython,
    _set_camera,
    build_env_from_terrain_name,
    policy_step,
)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# `ckpt_phase_<idx>_after_<terrain>.zip`
_CKPT_RE = re.compile(r"^ckpt_phase_(?P<idx>\d+)_after_(?P<terrain>.+)\.zip$")


def _detect_run(results_dir: Path):
    """From results/<scenario>/<method>[__<perception>]/seed_<N>/ recover
    (scenario, method, perception, seed) so each phase env can be built EXACTLY
    as the scenario trained it (obstacle-capable config + perception mode).
    Returns None if the path doesn't match, so callers fall back to the old
    terrain-name env builder."""
    try:
        seed = int(results_dir.name.split("_")[1])
        method_dir = results_dir.parent.name
        scenario = results_dir.parent.parent.name
    except (IndexError, ValueError):
        return None
    if "__" in method_dir:
        method, perception = method_dir.split("__", 1)
    else:
        method, perception = method_dir, None
    from rover_cl.missions.scenarios import SCENARIO_REGISTRY
    if scenario not in SCENARIO_REGISTRY:
        return None
    return scenario, method, perception, seed


def _build_scenario_phase_env(run, phase_idx: int):
    """Build the phase's env via get_scenario so obs/action/perception match how
    the policy was trained (returns None if unavailable)."""
    scenario, method, perception, seed = run
    from rover_cl.missions.scenarios import get_scenario
    mission = get_scenario(scenario, cl_method=method, perception=perception,
                           seed=seed)
    if phase_idx >= len(mission.tasks):
        return None
    return mission.tasks[phase_idx].env_factory(seed)


def discover_phases(results_dir: Path) -> list[tuple[int, str, Path]]:
    """Return [(phase_idx, terrain_name, checkpoint_path), ...] sorted by phase."""
    phases: list[tuple[int, str, Path]] = []
    for f in sorted(results_dir.glob("ckpt_phase_*_after_*.zip")):
        m = _CKPT_RE.match(f.name)
        if not m:
            continue
        phases.append((int(m["idx"]), m["terrain"], f))
    phases.sort(key=lambda r: r[0])
    return phases


def filter_phases(
    phases: list[tuple[int, str, Path]], spec: str | None
) -> list[tuple[int, str, Path]]:
    """`spec` examples: "3", "3-5", "0,2,4". None = no filter."""
    if not spec:
        return phases
    keep: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if "-" in token:
            a, b = token.split("-", 1)
            keep.update(range(int(a), int(b) + 1))
        else:
            keep.add(int(token))
    return [p for p in phases if p[0] in keep]


# ---------------------------------------------------------------------------
# Per-phase viewer loop
# ---------------------------------------------------------------------------


def _draw_markers(viewer, env) -> None:
    """Goal (green) + waypoints (blue) overlay spheres at the real rolled
    positions (the built-in start/goal sites are stale on randomised terrains)."""
    scn = viewer.user_scn
    scn.ngeom = 0

    def add(pos, size, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom], int(mujoco.mjtGeom.mjGEOM_SPHERE),
            np.array([size, size, size], float),
            np.array([pos[0], pos[1], pos[2]], float),
            np.eye(3).flatten(), np.array(rgba, np.float32))
        scn.ngeom += 1

    gx, gy = env.terrain.goal_pos
    add((gx, gy, 0.3), 0.5, (0.2, 0.9, 0.2, 0.9))
    for (wx, wy) in env.terrain.waypoints:
        add((wx, wy, 0.25), 0.35, (0.3, 0.55, 1.0, 0.9))


def _run_one_phase(
    env,
    ckpt_path: Path,
    seconds: float,
    realtime: bool,
    free_camera: bool,
) -> None:
    """Open the viewer on `env` driven by `ckpt_path`, run for `seconds`."""
    from stable_baselines3 import PPO

    try:
        policy = PPO.load(str(ckpt_path))
    except (ValueError, RuntimeError, KeyError) as exc:
        # obs_dim or action_space mismatch → the checkpoint pre-dates an env
        # change. Skip with a clear message instead of crashing the whole tour.
        print(f"  [skip] checkpoint {ckpt_path.name} won't load: {exc!r}")
        return

    model = env._model
    data = env._data
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    obstacle_env = bool(getattr(env, "use_lidar", False))

    obs, _ = env.reset()
    ep_reward = 0.0
    ep_steps = 0
    episode_idx = 0
    phase_start = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        _set_camera(viewer, base_id, free=free_camera)
        if obstacle_env:
            # obstacles live in the private lidar geom group when use_lidar;
            # show all geom groups and hide the stale built-in site markers.
            viewer.opt.geomgroup[:] = 1
            viewer.opt.sitegroup[:] = 0

        while viewer.is_running():
            if time.time() - phase_start >= seconds:
                break

            step_start = time.time()
            obs, terminated, truncated, info = policy_step(env, policy, obs)
            ep_reward += float(info.get("_reward", 0.0))
            ep_steps += 1
            if obstacle_env:
                _draw_markers(viewer, env)
            viewer.sync()

            if terminated or truncated:
                success = bool(info.get("is_success", False))
                print(f"  ep {episode_idx}: reward={ep_reward:+.1f} "
                      f"steps={ep_steps} success={success}")
                episode_idx += 1
                ep_reward = 0.0
                ep_steps = 0
                obs, _ = env.reset()

            if realtime:
                target_dt = model.opt.timestep * getattr(env, "control_decimation", 1)
                slack = target_dt - (time.time() - step_start)
                if slack > 0:
                    time.sleep(slack)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("results_dir", type=Path,
                    help="Path to results/<scenario>/<method>/seed_<N>/")
    ap.add_argument("--seconds-per-phase", type=float, default=30.0,
                    help="Wall-clock seconds to spend on each phase (default 30).")
    ap.add_argument("--phases", type=str, default=None,
                    help="Phase index filter, e.g. '3', '3-5', '0,2,4'. "
                         "Default: all phases.")
    ap.add_argument("--fast", dest="realtime", action="store_false",
                    help="Run the sim as fast as possible instead of realtime.")
    ap.add_argument("--realtime", dest="realtime", action="store_true",
                    default=True)
    ap.add_argument("--free-camera", action="store_true",
                    help="Start each phase with a free orbit camera (default: "
                         "TRACKING camera following the rover).")
    args = ap.parse_args()

    if not args.results_dir.is_dir():
        print(f"ERROR: results dir not found: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    # Detects `mjpython` via mujoco.viewer._MJPYTHON — the same hook the
    # MuJoCo team uses internally; `sys.executable` doesn't reliably name
    # `mjpython` because the wrapper re-execs a regular Python.
    _guard_macos_mjpython()

    phases = filter_phases(discover_phases(args.results_dir), args.phases)
    if not phases:
        print(f"ERROR: no `ckpt_phase_*_after_*.zip` checkpoints in "
              f"{args.results_dir}", file=sys.stderr)
        sys.exit(1)

    # Recover (scenario, method, perception, seed) from the path so each phase
    # env is built exactly as trained (obstacle config + perception mode). Falls
    # back to the terrain-name builder for plain scenarios.
    run = _detect_run(args.results_dir)
    if run is not None:
        _, _, perception, _ = run
        print(f"detected run: scenario={run[0]} method={run[1]} "
              f"perception={perception} seed={run[3]}")

    print(f"Touring {len(phases)} phase(s) from {args.results_dir} "
          f"@ {args.seconds_per_phase:.0f} s each. "
          f"Close a window early to skip; Ctrl+C in terminal to abort.")
    for phase_idx, terrain_name, ckpt in phases:
        banner = f"=== Phase {phase_idx}: {terrain_name} ({ckpt.name}) ==="
        print(banner, flush=True)
        env = None
        if run is not None:
            try:
                env = _build_scenario_phase_env(run, phase_idx)
            except Exception as exc:  # noqa: BLE001 — fall back on any build error
                print(f"  [warn] scenario env build failed ({exc!r}); "
                      "falling back to terrain-name env")
        if env is None:
            env = build_env_from_terrain_name(terrain_name)
        try:
            _run_one_phase(
                env=env,
                ckpt_path=ckpt,
                seconds=args.seconds_per_phase,
                realtime=args.realtime,
                free_camera=args.free_camera,
            )
        except KeyboardInterrupt:
            print("\naborted by user (Ctrl+C).")
            sys.exit(0)


if __name__ == "__main__":
    main()
