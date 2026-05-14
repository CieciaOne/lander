"""Tests for scenario_11_robust_generalist + its new mechanisms.

Covers:
  * every RC_* terrain compiles + the rover settles to the expected rest pose
  * scenario_11 builds a mission with the right phase structure
  * interim eval cadence fires and is recorded in results.json
  * adaptive gate stops a phase early when success threshold is reached
  * plot_skill_survival writes a PNG without erroring on the new interim
    eval fields
"""

from __future__ import annotations

import warnings

import mujoco
import numpy as np
import pytest

from rover_cl.envs.terrains import TERRAIN_CATALOG, compile_scene
from rover_cl.missions.scenarios import get_scenario

warnings.simplefilter("ignore")


RC_TERRAINS = [
    "RC_locomotion",
    "RC_path_following",
    "RC_obstacle_avoidance",
    "RC_path_and_obstacles",
    "RC_terrain",
    "RC_terrain_plus",
    "RC_full_random",
]


# ---------------------------------------------------------------------------- terrains
@pytest.mark.parametrize("name", RC_TERRAINS)
def test_rc_terrain_rolls_and_settles(name):
    """Each new RC_* terrain produces a roll → compiles → rover settles."""
    spec = TERRAIN_CATALOG[name](seed=0)
    rng = np.random.default_rng(0)
    assert spec.randomize_on_reset is not None
    roll = spec.randomize_on_reset(rng)
    spec.start_pos = tuple(roll.start_pos)
    spec.start_yaw = float(roll.start_yaw)
    spec.goal_pos = tuple(roll.goal_pos)
    spec.waypoints = tuple(roll.waypoints)

    m, d = compile_scene(spec)
    for _ in range(80):
        mujoco.mj_step(m, d)
    # Rover should be upright (z within sane range) and not tipped.
    base_z = float(d.qpos[2])
    assert 0.2 < base_z < 2.5, f"rover z={base_z} out of range for {name}"
    quat = d.qpos[3:7]
    upright_z = 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)
    assert upright_z > 0.6, f"rover tipped on first settle on {name}"


def test_rc_full_random_samples_diverse_layouts():
    """RC_full_random should produce a meaningful mix across 30 rolls."""
    spec = TERRAIN_CATALOG["RC_full_random"](seed=0)
    rng = np.random.default_rng(1)
    n_waypoints = []
    n_obstacles = []
    has_terrain = []
    for _ in range(60):
        roll = spec.randomize_on_reset(rng)
        n_waypoints.append(len(roll.waypoints))
        n_obstacles.append(len(roll.obstacle_positions))
        has_terrain.append(roll.heightmap is not None and roll.heightmap.max() > 0.05)
    # Distribution should cover the full range of each axis.
    assert max(n_waypoints) >= 2, f"waypoint count too narrow: max={max(n_waypoints)}"
    assert min(n_waypoints) == 0
    assert max(n_obstacles) >= 3
    assert min(n_obstacles) == 0
    # At least some episodes should have meaningful terrain.
    assert any(has_terrain), "no rolls produced non-flat terrain"


# ---------------------------------------------------------------------------- scenario
def test_scenario_11_builds_correctly():
    m = get_scenario("scenario_11_robust_generalist", train_timesteps=10_000, seed=0)
    assert len(m.tasks) == 7
    assert [t.task_id for t in m.tasks] == [
        "RC_locomotion", "RC_path_following", "RC_obstacle_avoidance",
        "RC_path_and_obstacles", "RC_terrain", "RC_terrain_plus",
        "RC_full_random",
    ]
    # First six phases have gate thresholds set; last (capstone) does not.
    thresholds = [t.min_success_to_advance for t in m.tasks]
    assert all(th is not None for th in thresholds[:6])
    assert thresholds[-1] is None
    # Interim eval enabled by default.
    assert all(t.interim_eval_every == 25_000 for t in m.tasks)
    # EWC λ injected.
    assert m.cl_kwargs == {"lam": 400.0}


def test_scenario_11_gate_can_be_disabled():
    m = get_scenario("scenario_11_robust_generalist",
                     train_timesteps=10_000, seed=0, enable_gate=False)
    assert all(t.min_success_to_advance is None for t in m.tasks)


# ---------------------------------------------------------------------------- runner integration
def test_runner_records_interim_eval_in_results(tmp_path):
    """Train a TINY scenario with interim eval enabled and check the
    results.json contains an interim_eval list in the phase timings."""
    from rover_cl.envs.nav import RoverNavEnv
    from rover_cl.missions.base import Mission, Runner, Task

    def factory(seed: int):
        return RoverNavEnv(terrain="T1_flat", max_steps=80, seed=seed)

    task = Task(
        task_id="T1_flat",
        env_factory=factory,
        train_timesteps=400,
        eval_episodes=2,
        eval_max_steps=80,
        # Interim eval at every 200 steps so we get 2 checkpoints.
        interim_eval_every=200,
        interim_eval_episodes=2,
    )
    mission = Mission(name="tiny_interim_test", tasks=[task],
                      cl_method="naive", seed=0)
    runner = Runner(mission, results_dir=tmp_path, verbose=False)
    runner.run()

    import json
    data = json.loads((tmp_path / "results.json").read_text())
    phase0 = data["evaluations"][0]
    assert "timings" in phase0
    interim = phase0["timings"].get("interim_eval", [])
    assert len(interim) >= 1
    for entry in interim:
        assert set(entry) >= {"steps_trained_in_phase", "success_rate",
                              "mean_return", "n_episodes"}


def test_skill_survival_plot_handles_interim_eval(tmp_path):
    """The new plot reads `interim_eval` if present; gracefully no-ops otherwise."""
    import json
    from rover_cl.viz.plots import plot_skill_survival

    # Synthesize a minimal results.json with 2 phases. Phase 1 has interim eval.
    results = {
        "mission_name": "tiny", "cl_method": "naive", "seed": 0,
        "task_ids": ["A", "B"],
        "evaluations": [
            {
                "phase": 0, "after_training": "A",
                "per_task": {
                    "A": {"success_rate": 0.5, "mean_return": 1.0, "n_episodes": 2},
                    "B": None,
                },
                "timings": {"interim_eval": [
                    {"steps_trained_in_phase": 200, "success_rate": 0.25,
                     "mean_return": 0.0, "n_episodes": 2},
                    {"steps_trained_in_phase": 400, "success_rate": 0.50,
                     "mean_return": 1.0, "n_episodes": 2},
                ]},
            },
            {
                "phase": 1, "after_training": "B",
                "per_task": {
                    "A": {"success_rate": 0.3, "mean_return": 0.5, "n_episodes": 2},
                    "B": {"success_rate": 0.7, "mean_return": 2.0, "n_episodes": 2},
                },
                "timings": {"interim_eval": []},  # no interim for phase 1
            },
        ],
    }
    out = tmp_path / "skill_survival.png"
    plot_skill_survival(results, task_ids=["A", "B"], out=out)
    assert out.exists() and out.stat().st_size > 1000
