"""Accurate top-down maps for the slalom obstacle-navigation task.

Unlike the generic report plot, this renders the REAL collision geometry so a
reader can judge feasibility by eye:

* solid brown box .......... the obstacle geom (true size)
* hatched halo ............. the C-space obstacle = box grown by the rover
                             footprint radius (Minkowski sum with a disc). The
                             rover CENTRE may not enter this region — this is
                             what actually causes a collision, and it is why a
                             plain box looks passable when it is not.
* green disc (to scale) .... the rover footprint (0.9 m radius ⇒ 1.8 m across),
                             drawn at the start so the scale is obvious.
* grey dashed line ......... a provably collision-free geodesic path
                             (start → waypoints → goal), traced on the env's
                             NavField. Its presence proves the episode is
                             SOLVABLE; a "FAIL" with a dashed line present is a
                             policy failure, not an impossible map.
* navy line ................ the policy's actual trajectory.

Usage:
    python scripts/plot_obstacle_maps.py \
        [--policy results/_obstacle_nav/slalom_field_hard_best.zip] \
        [--seeds 8100-8105] [--n-waypoints 2] [--out results/_obstacle_nav/slalom_hard_maps.png]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from rover_cl.envs.nav import NavField, RoverNavEnv, ROVER_FOOTPRINT_RADIUS as F
from rover_cl.envs.terrains import _flat_template
from rover_cl.envs.randomization import (
    TerrainRoll, sample_start_goal_pair, sample_obstacles_slalom,
)

EK = dict(
    use_lidar=True, geo_heading_obs=False, control_mode="vw", max_steps=2200,
    progress_reward_mode="geodesic", collision_penalty=0.0, hit_penalty=5.0,
    stuck_in_collision_penalty=25.0, proximity_penalty_scale=0.20,
    proximity_safety_dist=1.7, collision_terminate_steps=1,
)

# Perception overrides so a policy is visualised with the SAME observation it was
# trained on (a slam policy in a privileged env would see the wrong obs).
PERCEPTION_EK = {
    "privileged": dict(obstacle_obs_mode="privileged", geo_heading_obs=True,
                       geo_heading_source="truth"),
    "reactive": dict(obstacle_obs_mode="none", geo_heading_obs=False),
    "slam": dict(obstacle_obs_mode="none", geo_heading_obs=True,
                 geo_heading_source="slam"),
}


def _gap_waypoints(gaps, n_wp):
    if not gaps or n_wp <= 0:
        return ()
    idx = np.linspace(0, len(gaps) - 1, min(n_wp, len(gaps))).round().astype(int)
    return tuple(gaps[i] for i in sorted(set(int(k) for k in idx)))


def _make_terrain(seed, n_obstacles, n_wp, plo, phi):
    spec = _flat_template("slalom_map", max_obstacles=8)

    def _roll(rng):
        st, yaw, goal = sample_start_goal_pair(
            rng, arena_half=12.0, min_separation=plo, max_separation=phi,
            margin=2.0, relative_bearing="front")
        pos, sz, gaps = sample_obstacles_slalom(
            rng, st, goal, n_obstacles, 8, size_range=(0.4, 0.6),
            return_gaps=True)
        return TerrainRoll(start_pos=st, start_yaw=yaw, goal_pos=goal,
                           waypoints=_gap_waypoints(gaps, n_wp),
                           obstacle_positions=pos, obstacle_sizes=sz)
    spec.randomize_on_reset = _roll
    return spec


def _live_obstacles(env):
    """(cx, cy, half) for every visible obstacle geom, read live from MuJoCo."""
    out = []
    for gid in env._obstacle_geom_ids:
        p = env._data.geom_xpos[gid]
        if float(p[2]) < -10.0:
            continue
        out.append((float(p[0]), float(p[1]), float(env._model.geom_size[gid][0])))
    return out


def _grid_descent(nf, start, max_steps=40000):
    """Extract the geodesic route on a NavField by walking, cell by cell, to the
    8-neighbour with the smallest distance-to-target. Because blocked (inflated)
    cells hold +inf they are never chosen, so the whole route stays in free
    space — no straight-line 'jumps' that cut through obstacles (the bug in the
    earlier heading-descent tracer). Returns cell-centre world points and whether
    the target was reached."""
    i, j = nf._idx(start[0]), nf._idx(start[1])
    n = nf.n
    if not np.isfinite(nf.dist[i, j]):
        return np.empty((0, 2)), False
    pts = []
    for _ in range(max_steps):
        pts.append((i * nf.res - nf.half + nf.res / 2,
                    j * nf.res - nf.half + nf.res / 2))
        if nf.dist[i, j] == 0.0:
            return np.array(pts), True
        best, bd = None, nf.dist[i, j]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n and nf.dist[ni, nj] < bd:
                    bd = nf.dist[ni, nj]; best = (ni, nj)
        if best is None:
            return np.array(pts), False
        i, j = best
    return np.array(pts), False


def _geodesic_path(obstacles, waypoints, start, goal, res=0.1):
    """Collision-free route start → waypoints → goal, extracted by grid descent
    on the NavField distance field for each leg. Returns the path points and
    whether every leg reached its target."""
    obs_xywh = [(cx, cy, h, h) for (cx, cy, h) in obstacles]
    targets = list(waypoints) + [goal]
    path = [np.array(start, dtype=float)]
    feasible = True
    for tgt in targets:
        nf = NavField(half_extent=15.0, res=res, obstacles_xywh=obs_xywh,
                      target_xy=(float(tgt[0]), float(tgt[1])), inflate=F)
        seg, reached = _grid_descent(nf, path[-1])
        if not reached:
            feasible = False
        if len(seg):
            path.extend(seg)
        path.append(np.array(tgt, dtype=float))
    return np.array(path), feasible


def _rollout(model, env, seed):
    o, _ = env.reset(seed=seed)
    xs, ys, touched, hit_xy = [], [], False, None
    for _ in range(EK["max_steps"]):
        a, _ = model.predict(o, deterministic=True)
        o, _, tm, tr, info = env.step(a)
        xs.append(info["pos_xy"][0]); ys.append(info["pos_xy"][1])
        if bool(info.get("collision")) and hit_xy is None:
            hit_xy = (info["pos_xy"][0], info["pos_xy"][1])  # first contact point
        touched = touched or bool(info.get("collision"))
        if tm or tr:
            break
    return bool(info["is_success"]), touched, xs, ys, hit_xy


def _min_clearance(path, obstacles, dstep=0.03):
    """Smallest distance from the path to any obstacle BOX edge (0 if inside),
    sampled DENSELY along every segment — not just at vertices, so a long
    straight segment that clips an obstacle between its endpoints is caught.
    >= footprint F ⇒ the route is collision-free for the disc model."""
    if len(path) < 1 or not obstacles:
        return float("inf")
    # Densify: interpolate points along each segment.
    dense = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        d = float(np.hypot(*(np.asarray(b) - np.asarray(a))))
        k = max(1, int(d / dstep))
        for t in np.linspace(0, 1, k + 1)[1:]:
            dense.append(a + t * (np.asarray(b) - np.asarray(a)))
    best = float("inf")
    for (px, py) in dense:
        for (cx, cy, h) in obstacles:
            dx = max(abs(px - cx) - h, 0.0); dy = max(abs(py - cy) - h, 0.0)
            best = min(best, float(np.hypot(dx, dy)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="results/_obstacle_nav/slalom_field_hard_best.zip")
    ap.add_argument("--seeds", default="8100-8105")
    ap.add_argument("--n-obstacles", type=int, default=5)
    ap.add_argument("--n-waypoints", type=int, default=2)
    ap.add_argument("--path-lo", type=float, default=9.0)
    ap.add_argument("--path-hi", type=float, default=14.0)
    ap.add_argument("--out", default="results/_obstacle_nav/slalom_hard_maps.png")
    ap.add_argument("--perception", default=None,
                    choices=["privileged", "reactive", "slam"],
                    help="Visualise the policy with this perception config "
                         "(must match how it was trained).")
    args = ap.parse_args()

    if args.perception:
        EK.update(PERCEPTION_EK[args.perception])
        print(f"perception: {args.perception} -> {PERCEPTION_EK[args.perception]}")

    if "-" in args.seeds:
        a, b = args.seeds.split("-"); seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    from stable_baselines3 import PPO
    model = PPO.load(args.policy, device="cpu")

    ncol = 3
    nrow = int(np.ceil(len(seeds) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow))
    axs = np.atleast_1d(axs).ravel()

    for ax, seed in zip(axs, seeds):
        env = RoverNavEnv(
            terrain=_make_terrain(seed, args.n_obstacles, args.n_waypoints,
                                  args.path_lo, args.path_hi), **EK)
        success, touched, xs, ys, hit_xy = _rollout(model, env, seed)
        obstacles = _live_obstacles(env)
        start, goal = env.terrain.start_pos, env.terrain.goal_pos
        wps = env.terrain.waypoints

        for (cx, cy, h) in obstacles:
            # C-space obstacle: box grown by footprint F, rounded corners r=F.
            # Outline (no blob) so overlapping halos still reveal the gap.
            ax.add_patch(FancyBboxPatch(
                (cx - h, cy - h), 2 * h, 2 * h,
                boxstyle=f"round,pad={F},rounding_size={F}",
                facecolor="orange", alpha=0.12, edgecolor="darkorange",
                linewidth=1.2, hatch="//", zorder=1.5))
            # true obstacle body.
            ax.add_patch(Rectangle((cx - h, cy - h), 2 * h, 2 * h,
                                   facecolor="saddlebrown", edgecolor="black",
                                   linewidth=0.8, zorder=2))

        # Provably-feasible geodesic path (fine grid so it hugs the true gaps).
        gpath, feasible = _geodesic_path(obstacles, wps, start, goal)
        clr = _min_clearance(gpath, obstacles)
        feasible = feasible and clr >= F  # honest: only "feasible" if disc fits
        if len(gpath) > 1:
            ax.plot(gpath[:, 0], gpath[:, 1], "--", color="green", lw=1.6,
                    alpha=0.9, zorder=4, label="feasible route (exists)")
            # Rover footprint discs sliding along the feasible route — visual
            # proof the 1.8 m rover clears the obstacles on this route.
            step = max(1, len(gpath) // 7)
            for q in gpath[::step]:
                ax.add_patch(Circle((q[0], q[1]), F, facecolor="green",
                                    alpha=0.07, edgecolor="green",
                                    linewidth=0.5, zorder=3))

        # Rover footprint to scale at the start.
        ax.add_patch(Circle(start, F, facecolor="green", alpha=0.18,
                            edgecolor="green", linewidth=1.2, zorder=3))
        ax.plot(*start, "go", ms=9, zorder=6)
        ax.plot(*goal, "r*", ms=20, zorder=6)
        for wp in wps:
            ax.plot(wp[0], wp[1], "bs", ms=10, alpha=0.8, zorder=6)

        # The policy's actual attempt.
        ax.plot(xs, ys, "-", color="navy", lw=2.0, zorder=5,
                label="rover (this run)")
        if hit_xy is not None:
            ax.plot(hit_xy[0], hit_xy[1], "x", color="red", ms=13, mew=3,
                    zorder=7, label="rover collision")

        verdict = "SUCCESS" if success else ("FAIL — collided" if touched
                                             else "FAIL — timeout/stuck")
        feas = f"route exists, min clr {clr:.2f} m" if feasible \
            else "NO clean route"
        ax.set_title(f"seed {seed}: {verdict}\n(map: {feas})", fontsize=9)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=6.5)

    for ax in axs[len(seeds):]:
        ax.axis("off")

    fig.suptitle(
        "Slalom obstacle maps — hatched halo = collision region for the rover "
        f"centre (footprint r={F} m); grey dashed = provably feasible path",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=90)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
