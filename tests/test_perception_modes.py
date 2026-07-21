"""Perception-mode axis for RoverNavEnv: privileged / reactive(none) / slam.

These modes let the CL comparison be run as (CL method × perception mode):
- privileged: ground-truth obstacle AABBs in the obs (teacher-level).
- none:       no ground-truth obstacle info (honest mapless; lidar only).
- slam:       obstacles discovered online into an OccupancyMap; geo_heading is
              planned on the DISCOVERED map, not ground truth.
"""
import numpy as np
import pytest

from rover_cl.envs.nav import RoverNavEnv, OccupancyMap, K_OBSTACLES
from rover_cl.envs.terrains import _flat_template
from rover_cl.envs.randomization import (
    TerrainRoll, sample_start_goal_pair, sample_obstacles_slalom,
)


def _slalom_terrain(n_obs=5):
    spec = _flat_template("percep", max_obstacles=8)

    def _roll(rng):
        st, yaw, goal = sample_start_goal_pair(
            rng, arena_half=12.0, min_separation=9.0, max_separation=14.0,
            margin=2.0, relative_bearing="front")
        pos, sz, _ = sample_obstacles_slalom(
            rng, st, goal, n_obs, 8, size_range=(0.4, 0.6), return_gaps=True)
        return TerrainRoll(start_pos=st, start_yaw=yaw, goal_pos=goal,
                           waypoints=(), obstacle_positions=pos, obstacle_sizes=sz)
    spec.randomize_on_reset = _roll
    return spec


def _obstacle_slots(obs):
    return obs[8:8 + K_OBSTACLES * 4].reshape(K_OBSTACLES, 4)


def test_privileged_mode_exposes_real_obstacles():
    env = RoverNavEnv(terrain=_slalom_terrain(), use_lidar=True, control_mode="vw",
                      progress_reward_mode="geodesic", obstacle_obs_mode="privileged")
    obs, _ = env.reset(seed=8100)
    slots = _obstacle_slots(obs)
    uniq = {tuple(np.round(s, 4)) for s in slots}
    assert len(uniq) > 1, "privileged mode should show distinct real obstacles"


def test_none_mode_hides_obstacles_but_keeps_reward_distance():
    env = RoverNavEnv(terrain=_slalom_terrain(), use_lidar=True, control_mode="vw",
                      progress_reward_mode="geodesic", obstacle_obs_mode="none")
    obs, _ = env.reset(seed=8100)
    slots = _obstacle_slots(obs)
    uniq = {tuple(np.round(s, 4)) for s in slots}
    assert len(uniq) == 1, "none mode must expose no obstacle info (all sentinel)"
    # obs dimension is unchanged (drop-in swap)
    assert obs.shape[0] == env.observation_space.shape[0]
    # true nearest-obstacle distance still available for the proximity reward
    env.step(np.zeros(2, dtype=np.float32))
    assert np.isfinite(env._min_obstacle_dist)


def test_none_and_privileged_same_obs_dim():
    a = RoverNavEnv(terrain=_slalom_terrain(), use_lidar=True, control_mode="vw",
                    obstacle_obs_mode="privileged").observation_space.shape[0]
    b = RoverNavEnv(terrain=_slalom_terrain(), use_lidar=True, control_mode="vw",
                    obstacle_obs_mode="none").observation_space.shape[0]
    assert a == b


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        RoverNavEnv(terrain=_slalom_terrain(), obstacle_obs_mode="bogus")
    with pytest.raises(ValueError):
        RoverNavEnv(terrain=_slalom_terrain(), geo_heading_source="bogus")


def test_slam_mode_discovers_obstacles_online():
    env = RoverNavEnv(terrain=_slalom_terrain(), use_lidar=True, control_mode="vw",
                      progress_reward_mode="geodesic", obstacle_obs_mode="none",
                      geo_heading_obs=True, geo_heading_source="slam")
    obs, _ = env.reset(seed=8100)
    # empty map + no route field at reset
    assert env._occ_map is not None and int((env._occ_map.occ > 1.0).sum()) == 0
    assert env._slam_nav_field is None
    for _ in range(30):
        obs, _, term, trunc, _ = env.step(np.array([1.0, 0.0], dtype=np.float32))
        if term or trunc:
            break
    assert int((env._occ_map.occ > 1.0).sum()) > 0, "lidar should discover obstacles"
    assert env._slam_nav_field is not None, "discovered-map geodesic should build"
    assert obs.shape[0] == env.observation_space.shape[0]


def test_occupancy_map_ray_marking():
    m = OccupancyMap(half_extent=15.0, res=0.3)
    # a ray from origin to (3, 0) that HITS marks the hit cell occupied and the
    # cells before it free.
    m.update(0.0, 0.0, [(3.0, 0.0, True)])
    hit_i, hit_j = m._idx(3.0), m._idx(0.0)
    assert m.occ[hit_i, hit_j] > 0.0, "hit cell should accumulate occupied evidence"
    mid_i, mid_j = m._idx(1.5), m._idx(0.0)
    assert m.occ[mid_i, mid_j] < 0.0, "cells along the ray should be free"
    # dilation grows the blocked region
    b0 = m.blocked(0).sum()
    b2 = m.blocked(2).sum()
    assert b2 >= b0
