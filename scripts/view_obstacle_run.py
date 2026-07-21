"""Watch the obstacle-nav policy run the SLALOM map episodes in the 3D viewer.

Rebuilds the exact same episodes shown in the map PNGs
(scripts/plot_obstacle_maps.py) — same seeds, same slalom obstacles and in-gap
waypoints, same env config (lidar + Curiosity-style vw control + geodesic
reward) — and replays the trained policy inside the MuJoCo viewer. Episodes play
one after another for the given seeds; it loops back to the first when done.

The existing visualize_rover.py can't do this: it builds the env with defaults
(no lidar / ackermann → 44-D obs), which mismatches the obstacle policy (59-D,
vw), and it can't reproduce a specific seed's slalom layout.

macOS: the viewer needs the main thread, so launch with `mjpython`:

    mjpython scripts/view_obstacle_run.py                 # HARD maps (8100-8105)
    mjpython scripts/view_obstacle_run.py --seeds 8200-8205 --n-waypoints 0   # field maps
    mjpython scripts/view_obstacle_run.py --seed 8100     # just one episode, looped

Linux: use plain `python`. Controls: TAB cycles cameras, mouse orbits/zooms,
ESC quits.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rover_cl.envs.nav import RoverNavEnv
from rover_cl.envs.terrains import _flat_template
from rover_cl.envs.randomization import (
    TerrainRoll, sample_start_goal_pair, sample_obstacles_slalom,
)

# Must match scripts/plot_obstacle_maps.py + eval so the 3D run == the map.
EK = dict(
    use_lidar=True, geo_heading_obs=False, control_mode="vw", max_steps=2400,
    progress_reward_mode="geodesic", collision_penalty=0.0, hit_penalty=8.0,
    stuck_in_collision_penalty=25.0, proximity_penalty_scale=0.28,
    proximity_safety_dist=1.8, collision_terminate_steps=1,
)


def _gap_waypoints(gaps, n_wp):
    if not gaps or n_wp <= 0:
        return ()
    idx = np.linspace(0, len(gaps) - 1, min(n_wp, len(gaps))).round().astype(int)
    return tuple(gaps[i] for i in sorted(set(int(k) for k in idx)))


def _terrain(seed, n_obs, n_wp, plo, phi):
    spec = _flat_template("view", max_obstacles=8)

    def _roll(rng):
        st, yaw, goal = sample_start_goal_pair(
            rng, arena_half=12.0, min_separation=plo, max_separation=phi,
            margin=2.0, relative_bearing="front")
        pos, sz, gaps = sample_obstacles_slalom(
            rng, st, goal, n_obs, 8, size_range=(0.4, 0.6), return_gaps=True)
        return TerrainRoll(start_pos=st, start_yaw=yaw, goal_pos=goal,
                           waypoints=_gap_waypoints(gaps, n_wp),
                           obstacle_positions=pos, obstacle_sizes=sz)
    spec.randomize_on_reset = _roll
    return spec


def _parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def _guard_macos_mjpython():
    if sys.platform == "darwin" and getattr(mujoco.viewer, "_MJPYTHON", None) is None:
        print("ERROR: on macOS launch with `mjpython scripts/view_obstacle_run.py` "
              "(the viewer needs the main thread).", file=sys.stderr)
        sys.exit(2)


def _load_policy(path):
    """Load an SB3 PPO or an sb3-contrib RecurrentPPO checkpoint."""
    from stable_baselines3 import PPO
    try:
        return PPO.load(str(path)), False
    except Exception:
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(str(path)), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy",
                    default="results/_obstacle_nav/slalom_field_hard_best.zip")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--seeds", default="8100-8105", help="range a-b or list a,b,c")
    grp.add_argument("--seed", type=int, help="single seed (looped)")
    ap.add_argument("--n-obstacles", type=int, default=5)
    ap.add_argument("--n-waypoints", type=int, default=2,
                    help="2 = HARD maps; 0 = field maps")
    ap.add_argument("--path-lo", type=float, default=9.0)
    ap.add_argument("--path-hi", type=float, default=14.0)
    ap.add_argument("--fast", action="store_true", help="run faster than real time")
    args = ap.parse_args()

    _guard_macos_mjpython()
    seeds = [args.seed] if args.seed is not None else _parse_seeds(args.seeds)
    policy, recurrent = _load_policy(args.policy)
    print(f"policy: {args.policy}  ({'RecurrentPPO' if recurrent else 'PPO'})")
    print(f"seeds: {seeds}  (n_obstacles={args.n_obstacles}, "
          f"n_waypoints={args.n_waypoints})")

    # One env; we re-roll its terrain per seed by rebuilding. Simpler: build a
    # fresh env per episode (cheap) and relaunch the viewer model each time is
    # not possible mid-session, so we keep ONE model shape (max_obstacles=8) and
    # just reset with each seed — the slalom layout is baked per reset.
    idx = {"i": 0}

    def build_env(seed):
        return RoverNavEnv(
            terrain=_terrain(seed, args.n_obstacles, args.n_waypoints,
                             args.path_lo, args.path_hi), **EK)

    env = build_env(seeds[0])
    obs, _ = env.reset(seed=seeds[0])
    model, data = env._model, env._data
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    lstm_state = None
    ep_starts = np.ones((1,), dtype=bool)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = base_id
        viewer.cam.distance = 6.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 110
        print(f"\n>>> seed {seeds[0]} — watch the rover weave the slalom. "
              "TAB: cameras, mouse: orbit, ESC: quit.\n")
        steps = 0
        while viewer.is_running():
            t0 = time.time()
            if recurrent:
                action, lstm_state = policy.predict(
                    obs, state=lstm_state, episode_start=ep_starts,
                    deterministic=True)
            else:
                action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            ep_starts = np.array([terminated or truncated])
            steps += 1
            viewer.sync()

            if terminated or truncated:
                res = ("SUCCESS" if info.get("is_success") else
                       ("collided" if info.get("collision") else "timeout/stuck"))
                print(f"[seed {seeds[idx['i']]}] {res} in {steps} steps")
                idx["i"] = (idx["i"] + 1) % len(seeds)
                nxt = seeds[idx["i"]]
                # The slalom layout is a function of the reset SEED (the terrain's
                # randomize_on_reset reads the env RNG), so reset(seed=nxt) alone
                # reproduces map episode `nxt` on the SAME compiled model — no
                # terrain swap / recompile, so the viewer's model/data stay valid.
                obs, _ = env.reset(seed=nxt)
                lstm_state = None
                ep_starts = np.ones((1,), dtype=bool)
                steps = 0
                print(f"\n>>> seed {nxt}\n")

            if not args.fast:
                target_dt = model.opt.timestep * getattr(env, "control_decimation", 1)
                slack = target_dt - (time.time() - t0)
                if slack > 0:
                    time.sleep(slack)


if __name__ == "__main__":
    main()
