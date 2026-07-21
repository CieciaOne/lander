"""Train a RECURRENT (LSTM) policy on the feasible slalom obstacle-nav task.

Motivation: the feedforward MLP policy plateaus at HARD ~0.65-0.75; its failures
are timeouts / getting stuck at tight gates and inconsistent multi-waypoint
sequencing. An LSTM policy carries hidden state across timesteps, giving it
(a) memory of obstacles that have left the forward lidar arc, and (b) a sense of
"where I am in the waypoint sequence" — both of which target that bottleneck.

Cannot warm-start from the MLP policy (different architecture), so this trains
from scratch: locomotion bootstrap -> obstacle curriculum -> full slalom +
in-gap waypoints. Same env config as the MLP best (geodesic reward, proximity
shaping, strict collision-terminate=1, Curiosity-style vw control, lidar).

Checkpoint selection uses a FIXED 100-episode eval (unbiased). Recurrent eval
carries the LSTM state across steps and resets it at episode boundaries.

Usage:
    python scripts/train_obstacle_recurrent.py [--chunks 60] [--n-envs 6]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from rover_cl.envs.nav import RoverNavEnv
from rover_cl.envs.terrains import _flat_template
from rover_cl.envs.randomization import (
    TerrainRoll, sample_start_goal_pair, sample_obstacles_slalom,
)

EK = dict(
    use_lidar=True, geo_heading_obs=False, control_mode="vw", max_steps=2400,
    progress_reward_mode="geodesic", collision_penalty=0.0, hit_penalty=8.0,
    stuck_in_collision_penalty=25.0, proximity_penalty_scale=0.28,
    proximity_safety_dist=1.8, clearance_speed_penalty_scale=0.10,
    clearance_safe_dist=2.0, collision_terminate_steps=1,
)
MAX_STEPS = EK["max_steps"]
CKPT = "/tmp/obstacle_recurrent_best"


def _gap_waypoints(gaps, n_wp):
    if not gaps or n_wp <= 0:
        return ()
    idx = np.linspace(0, len(gaps) - 1, min(n_wp, len(gaps))).round().astype(int)
    return tuple(gaps[i] for i in sorted(set(int(k) for k in idx)))


# (path_lo, path_hi) per mode; obstacle/waypoint counts set in _roll.
CFG = {"boot": (4.0, 8.0), "train": (7.0, 14.0),
       "loco": (6.0, 12.0), "field": (9.0, 14.0), "hard": (9.0, 14.0)}


def _terrain(seed, mode, stage):
    plo, phi = CFG[mode]
    spec = _flat_template("rec", max_obstacles=8)

    def _roll(rng):
        st, yaw, goal = sample_start_goal_pair(
            rng, arena_half=12.0, min_separation=plo, max_separation=phi,
            margin=2.0, relative_bearing="front")
        if mode == "boot" or mode == "loco":
            n_obs, n_wp = 0, 0
        elif mode == "field":
            n_obs, n_wp = 5, 0
        elif mode == "hard":
            n_obs, n_wp = 5, 2
        else:  # train: difficulty ramps with `stage`
            r = rng.uniform()
            if r < 0.20:
                n_obs = 0
            elif stage == "A":
                n_obs = 2
            elif stage == "B":
                n_obs = 4
            else:
                n_obs = 5
            n_wp = int(rng.integers(1, 3)) if (stage == "C" and rng.uniform() < 0.5) else 0
        pos, sz, gaps = sample_obstacles_slalom(
            rng, st, goal, n_obs, 8, size_range=(0.4, 0.6), return_gaps=True)
        return TerrainRoll(start_pos=st, start_yaw=yaw, goal_pos=goal,
                           waypoints=_gap_waypoints(gaps, n_wp),
                           obstacle_positions=pos, obstacle_sizes=sz)
    spec.randomize_on_reset = _roll
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=60, help="150k-step chunks after bootstrap")
    ap.add_argument("--boot-chunks", type=int, default=6)
    ap.add_argument("--n-envs", type=int, default=6)
    args = ap.parse_args()

    stage_holder = {"stage": "A"}

    def make_venv(mode, base):
        def mk(s):
            def _m():
                return Monitor(RoverNavEnv(
                    terrain=_terrain(base + s, mode, stage_holder["stage"]),
                    seed=base + s, **EK))
            return _m
        return SubprocVecEnv([mk(i) for i in range(args.n_envs)])

    ev = {m: RoverNavEnv(terrain=_terrain(700 + k, m, "C"), **EK)
          for k, m in enumerate(["loco", "field", "hard"])}

    def rate(env, n, base):
        s = 0
        for ep in range(n):
            obs, _ = env.reset(seed=base + ep)
            state = None
            starts = np.ones((1,), dtype=bool)
            for _ in range(MAX_STEPS):
                action, state = model.predict(
                    obs, state=state, episode_start=starts, deterministic=True)
                obs, _, tm, tr, info = env.step(action)
                starts = np.array([tm or tr])
                if tm or tr:
                    break
            s += int(info["is_success"])
        return s / n

    policy_kwargs = dict(lstm_hidden_size=128, n_lstm_layers=1,
                         net_arch=dict(pi=[128], vf=[128]), enable_critic_lstm=True)
    print("=== RecurrentPPO (LSTM) obstacle-nav, from scratch ===", flush=True)
    model = RecurrentPPO(
        "MlpLstmPolicy", make_venv("boot", 100), n_steps=256, batch_size=256,
        gamma=0.995, ent_coef=0.01, learning_rate=3e-4, target_kl=0.05,
        policy_kwargs=policy_kwargs, device="cpu", seed=0, verbose=0)

    print(f"--- bootstrap locomotion ({args.boot_chunks} chunks) ---", flush=True)
    for c in range(args.boot_chunks):
        model.learn(total_timesteps=150_000, reset_num_timesteps=False)
        if c % 2 == 1 or c == args.boot_chunks - 1:
            print(f"  boot+{(c+1)*150000}: loco={rate(ev['loco'],40,4000):.2f}", flush=True)

    best = -1.0
    schedule = [("A", 0.30), ("B", 0.60), ("C", 1.0)]  # (stage, fraction cutoff)
    for c in range(1, args.chunks + 1):
        frac = c / args.chunks
        stage_holder["stage"] = next(s for s, cut in schedule if frac <= cut)
        model.set_env(make_venv("train", 200 + c))  # fresh env each chunk (leak-free)
        model.learn(total_timesteps=150_000, reset_num_timesteps=False)
        if c % 3 == 0 or c == 1:
            lo = rate(ev["loco"], 40, 4000)
            fd = rate(ev["field"], 100, 5000)
            hd = rate(ev["hard"], 100, 6000)
            star = ""
            if lo >= 0.85 and (hd + 0.5 * fd) > best:
                best = hd + 0.5 * fd
                model.save(CKPT)
                star = " *saved"
            print(f"  [{stage_holder['stage']}] +{c*150000}: loco={lo:.2f} "
                  f"field={fd:.3f} HARD={hd:.3f}{star}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
