"""Tests for the four new CL methods (hybrid, l2, mas, distill) +
the new scenarios (12 joint training, 13 integrated curriculum).

Smoke-level coverage — each method can be constructed, trained on a tiny
2-task budget, saved + loaded, and produces a sensible CL state. Heavy
correctness checks live in test_cl.py for the existing methods.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from rover_cl.cl import (
    DistillCL,
    HybridEwcReplayCL,
    L2CL,
    MasCL,
    make_cl,
)
from rover_cl.envs.nav import RoverNavEnv
from rover_cl.envs.terrains import TERRAIN_CATALOG
from rover_cl.missions.scenarios import get_scenario


# ---------------------------------------------------------------------------- factory
@pytest.mark.parametrize("name,cls", [
    ("hybrid", HybridEwcReplayCL),
    ("l2", L2CL),
    ("mas", MasCL),
    ("distill", DistillCL),
])
def test_make_cl_builds_method(name, cls):
    cl = make_cl(name)
    assert isinstance(cl, cls)
    assert cl.name == name


# ---------------------------------------------------------------------------- tiny training cycle
def _tiny_env(seed: int = 0) -> gym.Env:
    return RoverNavEnv(terrain="T1_flat", max_steps=40, seed=seed)


@pytest.mark.parametrize("name", ["hybrid", "l2", "mas", "distill"])
def test_cl_method_runs_two_tasks(tmp_path, name):
    """Train each method on two tiny tasks. Verify it doesn't crash and
    the CL state grows as expected."""
    cl = make_cl(name)
    env_a = _tiny_env(seed=0)
    cl.train_on(env=env_a, total_timesteps=400, task_id="A", log_dir=None)
    env_a.close()

    env_b = _tiny_env(seed=1)
    cl.train_on(env=env_b, total_timesteps=400, task_id="B", log_dir=None)
    env_b.close()

    assert cl.seen_task_ids == ["A", "B"]


# ---------------------------------------------------------------------------- save / load
@pytest.mark.parametrize("name,cls", [
    ("hybrid", HybridEwcReplayCL),
    ("l2", L2CL),
    ("mas", MasCL),
    ("distill", DistillCL),
])
def test_cl_method_save_load_roundtrip(tmp_path, name, cls):
    cl = make_cl(name)
    env = _tiny_env(seed=42)
    cl.train_on(env=env, total_timesteps=400, task_id="A", log_dir=None)
    env.close()

    path = tmp_path / f"ckpt_{name}.zip"
    cl.save(path)
    assert path.exists()

    cl2 = cls.load(path)
    assert "A" in cl2.seen_task_ids


# ---------------------------------------------------------------------------- specific mechanism checks
def test_hybrid_runs_both_rehearsal_and_penalty(tmp_path):
    """After two tasks, hybrid should have BOTH a non-empty buffer
    AND non-empty Fisher snapshot. The unit-test for "is this really
    a hybrid"."""
    cl = HybridEwcReplayCL(buffer_size_per_task=50, fisher_sample_size=20,
                          rehearsal_steps=5, penalty_steps=5)
    env_a = _tiny_env(seed=0)
    cl.train_on(env=env_a, total_timesteps=400, task_id="A", log_dir=None)
    env_a.close()
    assert "A" in cl.buffers and len(cl.buffers["A"]) > 0
    assert "A" in cl.fisher
    assert "A" in cl.theta_star

    env_b = _tiny_env(seed=1)
    cl.train_on(env=env_b, total_timesteps=400, task_id="B", log_dir=None)
    env_b.close()
    # Second task: rehearsal AND penalty should have actually fired.
    assert cl.last_rehearsal_steps_run > 0
    assert cl.last_penalty_steps_run > 0


def test_l2_uses_uniform_importance():
    """L2CL.fisher should be all-ones tensors (uniform per-param weight)."""
    cl = L2CL(fisher_sample_size=4)
    env = _tiny_env(seed=0)
    cl.train_on(env=env, total_timesteps=200, task_id="A", log_dir=None)
    env.close()
    assert "A" in cl.fisher
    for tensor in cl.fisher["A"].values():
        # All entries are 1.0 (uniform importance — the whole point of L2).
        assert float(tensor.min().item()) == 1.0
        assert float(tensor.max().item()) == 1.0


def test_distill_stores_teacher_and_obs_buffer():
    """DistillCL.teachers should hold a parameter snapshot AND
    obs_buffers should hold collected observations."""
    cl = DistillCL(buffer_size_per_task=30, distill_steps=4)
    env = _tiny_env(seed=0)
    cl.train_on(env=env, total_timesteps=200, task_id="A", log_dir=None)
    env.close()
    assert "A" in cl.teachers
    assert len(cl.teachers["A"]) > 0   # has parameters
    assert "A" in cl.obs_buffers
    assert len(cl.obs_buffers["A"]) > 0


# ---------------------------------------------------------------------------- scenarios 12 and 13
def test_scenario_12_joint_training_builds():
    m = get_scenario("scenario_12_joint_training", train_timesteps=10_000, seed=0)
    assert len(m.tasks) == 1
    assert m.tasks[0].task_id == "RC_full_random"
    assert m.cl_method == "naive"


def test_scenario_13_integrated_curriculum_builds():
    m = get_scenario(
        "scenario_13_integrated_curriculum",
        train_timesteps=10_000, seed=0,
    )
    assert len(m.tasks) == 4
    assert [t.task_id for t in m.tasks] == [
        "RC_foundation", "RC_navigation",
        "RC_navigation_terrain", "RC_full_random",
    ]
    # First three phases have gate thresholds; last does not.
    assert m.tasks[0].min_success_to_advance is not None
    assert m.tasks[-1].min_success_to_advance is None


# ---------------------------------------------------------------------------- new terrains compile
@pytest.mark.parametrize("name", [
    "RC_foundation", "RC_navigation", "RC_navigation_terrain",
])
def test_new_rc_terrains_compile(name):
    from rover_cl.envs.terrains import compile_scene

    spec = TERRAIN_CATALOG[name](seed=0)
    rng = np.random.default_rng(0)
    roll = spec.randomize_on_reset(rng)
    spec.start_pos = tuple(roll.start_pos)
    spec.start_yaw = float(roll.start_yaw)
    spec.goal_pos = tuple(roll.goal_pos)
    spec.waypoints = tuple(roll.waypoints)
    model, _ = compile_scene(spec)
    assert model.ngeom > 0


def test_rc_navigation_has_both_obstacles_and_waypoints():
    """The headline design point of RC_navigation: every episode
    has obstacles AND most have waypoints. Verify the distribution."""
    from rover_cl.envs.randomization import HIDE_Z

    spec = TERRAIN_CATALOG["RC_navigation"](seed=0)
    rng = np.random.default_rng(0)
    has_obs = 0
    has_wp = 0
    for _ in range(200):
        r = spec.randomize_on_reset(rng)
        if any(p[2] > HIDE_Z + 1.0 for p in r.obstacle_positions):
            has_obs += 1
        if r.waypoints:
            has_wp += 1
    # Every episode should have at least 1 obstacle.
    assert has_obs == 200, f"expected all episodes with obstacles, got {has_obs}/200"
    # Most episodes should have waypoints.
    assert has_wp >= 140, f"expected >=70% with waypoints, got {has_wp}/200"
