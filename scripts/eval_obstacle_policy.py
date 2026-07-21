"""Evaluate an obstacle-navigation policy across difficulty tiers and emit a
success-rate bar chart + a JSON of the numbers.

Tiers (all on the guaranteed-feasible slalom task, strict collision-terminate=1
so ANY contact counts as failure — the honest metric):

* loco ...... no obstacles (locomotion floor)
* field ..... 3-4 genuinely-blocking obstacles, no waypoints
* dense ..... 5 obstacles on a short path (tight)
* hard ...... 3-4 obstacles + 1-2 in-gap waypoints
* hard_long . obstacles + 2 waypoints on a long (11-14 m) path

Every reported success is additionally checked to be collision-free.

Usage:
    python scripts/eval_obstacle_policy.py \
        [--policy results/_obstacle_nav/slalom_field_hard_best.zip] \
        [--episodes 60] [--out-dir results/_obstacle_nav]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rover_cl.envs.nav import RoverNavEnv
from rover_cl.envs.terrains import _flat_template
from rover_cl.envs.randomization import (
    TerrainRoll, sample_start_goal_pair, sample_obstacles_slalom,
)

EK = dict(
    use_lidar=True, geo_heading_obs=False, control_mode="vw", max_steps=2400,
    progress_reward_mode="geodesic", collision_penalty=0.0, hit_penalty=8.0,
    stuck_in_collision_penalty=25.0, proximity_penalty_scale=0.28,
    proximity_safety_dist=1.8, collision_terminate_steps=1,
)

# (label, path_lo, path_hi, n_obstacles, n_waypoints, eval-seed base)
TIERS = [
    ("loco",      6.0, 12.0, 0, 0, 4000),
    ("field",     9.0, 14.0, 5, 0, 5000),
    ("dense",     7.0, 11.0, 5, 0, 5500),
    ("hard",      9.0, 14.0, 5, 2, 6000),
    ("hard_long", 11.0, 14.0, 5, 2, 6500),
]


def _gap_waypoints(gaps, n_wp):
    if not gaps or n_wp <= 0:
        return ()
    idx = np.linspace(0, len(gaps) - 1, min(n_wp, len(gaps))).round().astype(int)
    return tuple(gaps[i] for i in sorted(set(int(k) for k in idx)))


def _terrain(seed, plo, phi, n_obs, n_wp):
    spec = _flat_template("eval", max_obstacles=8)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy",
                    default="results/_obstacle_nav/slalom_field_hard_best.zip")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--out-dir", default="results/_obstacle_nav")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.policy, device="cpu")

    results = {}
    for label, plo, phi, n_obs, n_wp, base in TIERS:
        env = RoverNavEnv(terrain=_terrain(base, plo, phi, n_obs, n_wp), **EK)
        succ = clean = 0
        for ep in range(args.episodes):
            o, _ = env.reset(seed=base + ep)
            touched = False
            for _ in range(EK["max_steps"]):
                a, _ = model.predict(o, deterministic=True)
                o, _, tm, tr, info = env.step(a)
                touched = touched or bool(info.get("collision"))
                if tm or tr:
                    break
            s = int(info["is_success"])
            succ += s
            clean += int(s and not touched)
        results[label] = {"success": succ / args.episodes,
                          "collision_free": clean / args.episodes,
                          "n": args.episodes}
        print(f"  {label:10s}: success {succ}/{args.episodes} = "
              f"{succ/args.episodes*100:.0f}%  (collision-free "
              f"{clean/args.episodes*100:.0f}%)", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "obstacle_eval.json").write_text(json.dumps(results, indent=2))

    # Bar chart.
    labels = [t[0] for t in TIERS]
    vals = [results[l]["success"] * 100 for l in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, vals, color=["#4c72b0", "#55a868", "#8172b3",
                                       "#c44e52", "#937860"])
    ax.axhspan(80, 90, color="green", alpha=0.08, label="80-90% target")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%",
                ha="center", fontsize=10, weight="bold")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Obstacle-nav success by tier ({args.episodes} eps, strict "
                 "no-touch)\npolicy: " + Path(args.policy).name, fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "obstacle_success_by_tier.png", dpi=110)
    print(f"wrote {out/'obstacle_success_by_tier.png'} and obstacle_eval.json")


if __name__ == "__main__":
    main()
