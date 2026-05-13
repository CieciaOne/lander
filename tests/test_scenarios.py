"""End-to-end scenario tests.

These run a TINY version of the real research scenarios (a few hundred timesteps,
2 eval episodes per phase). They verify that the full mission pipeline works:
each CL method trains, evaluates, dumps JSON, and produces a correct-shape
retention matrix. They take a few minutes total on an M3 Air, not seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rover_cl.eval import compute_retention_matrix, load_results
from rover_cl.missions import Runner, get_scenario

pytestmark = pytest.mark.slow


TINY_TRAIN = 1024     # one PPO rollout per task
TINY_EVAL = 2
TINY_MAX_STEPS = 80


@pytest.mark.parametrize("cl_method", ["naive", "replay", "ewc"])
def test_scenario_01_runs_end_to_end(tmp_path: Path, cl_method: str) -> None:
    """Scenario 1 (T1 → T2) with both CL methods produces a valid retention matrix."""
    mission = get_scenario(
        "scenario_01_sequential_terrains",
        cl_method=cl_method,
        train_timesteps=TINY_TRAIN,
        eval_episodes=TINY_EVAL,
        max_steps=TINY_MAX_STEPS,
        seed=0,
    )
    runner = Runner(mission, results_dir=tmp_path, verbose=False)
    result = runner.run()

    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "ckpt_phase_0_after_T1_flat.zip").exists()
    assert (tmp_path / "ckpt_phase_1_after_T2_corridor.zip").exists()
    # Top-down report per (phase, eval-task) — phase 0 sees only T1; phase 1 both.
    assert (tmp_path / "report_phase_0_after_T1_flat_on_T1_flat.png").exists()
    assert (tmp_path / "report_phase_1_after_T2_corridor_on_T1_flat.png").exists()
    assert (tmp_path / "report_phase_1_after_T2_corridor_on_T2_corridor.png").exists()

    data = load_results(tmp_path / "results.json")
    assert data["cl_method"] == cl_method
    assert data["task_ids"] == ["T1_flat", "T2_corridor"]
    assert len(data["evaluations"]) == 2

    # Phase 0: only T1 evaluated; phase 1: both. Per-task entries are dicts or None.
    p0 = data["evaluations"][0]["per_task"]
    assert p0["T1_flat"] is not None and "success_rate" in p0["T1_flat"]
    assert p0["T2_corridor"] is None
    p1 = data["evaluations"][1]["per_task"]
    assert p1["T1_flat"] is not None
    assert p1["T2_corridor"] is not None

    mat = compute_retention_matrix(data)
    assert mat.shape == (2, 2)
    assert np.isnan(mat[0, 1])         # T2 not yet seen at phase 0
    assert not np.isnan(mat[0, 0])
    assert not np.isnan(mat[1, 0])
    assert not np.isnan(mat[1, 1])


def test_scenario_writes_plots(tmp_path: Path) -> None:
    """Plot helpers can render a real Runner output without errors."""
    from rover_cl.viz import plot_retention_curves, plot_retention_matrix

    mission = get_scenario(
        "scenario_01_sequential_terrains", cl_method="naive",
        train_timesteps=TINY_TRAIN, eval_episodes=TINY_EVAL,
        max_steps=TINY_MAX_STEPS, seed=1,
    )
    out = tmp_path / "naive"
    Runner(mission, results_dir=out, verbose=False).run()

    data = load_results(out / "results.json")
    mat = compute_retention_matrix(data)
    p1 = out / "matrix.png"
    p2 = out / "curves.png"
    plot_retention_matrix(mat, data["task_ids"], "test", p1)
    plot_retention_curves(data, data["task_ids"], p2)
    assert p1.stat().st_size > 0
    assert p2.stat().st_size > 0
