"""Tests for rover_cl.eval.metrics and rover_cl.viz.plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytestmark = pytest.mark.fast

from rover_cl.eval import (  # noqa: E402
    EpisodeStats,
    compute_avg_retention,
    compute_forgetting,
    compute_retention_matrix,
    evaluate_policy,
)
from rover_cl.viz import (  # noqa: E402
    plot_method_comparison,
    plot_retention_curves,
    plot_retention_matrix,
)


# ---------------------------------------------------------------------------
# Toy env for evaluate_policy
# ---------------------------------------------------------------------------


class _AlwaysSuccessEnv(gym.Env):
    """A 1-state / 1-action env that terminates with success on the first step."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(1)
        self._steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._steps = 0
        return 0, {}

    def step(self, action):
        self._steps += 1
        reward = 1.0
        terminated = True
        truncated = False
        info = {"is_success": True}
        return 0, reward, terminated, truncated, info


def test_evaluate_policy_with_toy_env() -> None:
    env = _AlwaysSuccessEnv()
    stats = evaluate_policy(
        policy_predict_fn=lambda _obs: 0,
        env=env,
        n_episodes=5,
        max_steps=10,
    )
    assert isinstance(stats, EpisodeStats)
    assert stats.n_episodes == 5
    assert stats.success_rate == pytest.approx(1.0)
    assert stats.mean_return == pytest.approx(1.0)
    assert stats.mean_steps_to_goal == pytest.approx(1.0)

    # to_dict / from_dict roundtrip.
    restored = EpisodeStats.from_dict(stats.to_dict())
    assert restored == stats


# ---------------------------------------------------------------------------
# Retention matrix
# ---------------------------------------------------------------------------


def _synthetic_results(success_grid: list[list[float | None]], method: str = "naive") -> dict[str, Any]:
    task_ids = [f"T{i + 1}" for i in range(len(success_grid))]
    evaluations = []
    for phase, row in enumerate(success_grid):
        per_task: dict[str, dict[str, Any] | None] = {}
        for j, value in enumerate(row):
            if value is None:
                per_task[task_ids[j]] = None
            else:
                per_task[task_ids[j]] = {
                    "success_rate": float(value),
                    "mean_return": 10.0 * float(value),
                    "mean_steps_to_goal": 200.0,
                    "n_episodes": 20,
                }
        evaluations.append(
            {"phase": phase, "after_training": task_ids[phase], "per_task": per_task}
        )
    return {
        "mission_name": "sequential_terrains",
        "cl_method": method,
        "seed": 0,
        "task_ids": task_ids,
        "evaluations": evaluations,
    }


def test_retention_matrix_shape_and_nan() -> None:
    grid: list[list[float | None]] = [
        [0.65, None, None],
        [0.22, 0.71, None],
        [0.18, 0.50, 0.80],
    ]
    results = _synthetic_results(grid)
    matrix = compute_retention_matrix(results)

    assert matrix.shape == (3, 3)
    # Upper triangle (strictly above diagonal) should be NaN.
    assert np.isnan(matrix[0, 1])
    assert np.isnan(matrix[0, 2])
    assert np.isnan(matrix[1, 2])
    # Diagonal and lower triangle should match the grid values.
    np.testing.assert_allclose(np.diag(matrix), [0.65, 0.71, 0.80])
    assert matrix[1, 0] == pytest.approx(0.22)
    assert matrix[2, 0] == pytest.approx(0.18)
    assert matrix[2, 1] == pytest.approx(0.50)


def test_forgetting_computation() -> None:
    # Hand-built 3x3 retention matrix with known forgetting per task.
    #   task 0: max = 0.9 (phase 0), final = 0.4 -> forgetting 0.5
    #   task 1: max = 0.8 (phase 1), final = 0.5 -> forgetting 0.3
    #   task 2: max = 0.7 (phase 2), final = 0.7 -> forgetting 0.0
    retention = np.array(
        [
            [0.9, np.nan, np.nan],
            [0.6, 0.8, np.nan],
            [0.4, 0.5, 0.7],
        ]
    )
    forgetting = compute_forgetting(retention)
    np.testing.assert_allclose(forgetting, [0.5, 0.3, 0.0])
    assert compute_avg_retention(retention) == pytest.approx((0.4 + 0.5 + 0.7) / 3)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def test_plots_produce_pngs(tmp_path: Path) -> None:
    grid: list[list[float | None]] = [
        [0.70, None, None, None],
        [0.55, 0.80, None, None],
        [0.40, 0.65, 0.82, None],
        [0.30, 0.55, 0.70, 0.85],
    ]
    naive = _synthetic_results(grid, method="naive")
    replay_grid: list[list[float | None]] = [
        [0.70, None, None, None],
        [0.66, 0.78, None, None],
        [0.60, 0.70, 0.80, None],
        [0.55, 0.68, 0.74, 0.83],
    ]
    replay = _synthetic_results(replay_grid, method="replay")
    task_ids = naive["task_ids"]

    retention = compute_retention_matrix(naive)

    p1 = tmp_path / "retention_matrix.png"
    fig1 = plot_retention_matrix(retention, task_ids, "Retention (naive)", p1)
    plt.close(fig1)

    p2 = tmp_path / "retention_curves.png"
    fig2 = plot_retention_curves(naive, task_ids, p2)
    plt.close(fig2)

    p3 = tmp_path / "method_comparison_avg.png"
    fig3 = plot_method_comparison({"naive": naive, "replay": replay}, task_ids, p3, metric="avg_retention")
    plt.close(fig3)

    p4 = tmp_path / "method_comparison_forget.png"
    fig4 = plot_method_comparison({"naive": naive, "replay": replay}, task_ids, p4, metric="forgetting")
    plt.close(fig4)

    for p in (p1, p2, p3, p4):
        assert p.exists(), f"missing plot {p}"
        assert p.stat().st_size > 0, f"empty plot {p}"
