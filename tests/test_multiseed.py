"""Tests for multi-seed sweeps and cross-seed aggregation.

All tests use synthetic results dicts that mirror ``MissionResult.to_dict()``
from ``src/rover_cl/missions/base.py``. No PPO training, no environment
stepping, no I/O beyond ``tmp_path``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rover_cl.eval import (  # noqa: E402
    aggregate_retention_matrices,
    collect_seed_results,
)
from rover_cl.viz import plot_method_comparison_with_variance  # noqa: E402

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Synthetic results helpers
# ---------------------------------------------------------------------------


def _results_from_matrix(
    retention: list[list[float | None]],
    *,
    method: str = "naive",
    seed: int = 0,
    mission_name: str = "scenario_test",
) -> dict[str, Any]:
    """Build a results dict whose ``compute_retention_matrix`` equals ``retention``.

    NaN cells (where j > i in a typical CL run) are encoded as ``None`` in
    ``per_task`` — the same shape that ``MissionResult.to_dict`` produces.
    """
    n = len(retention)
    task_ids = [f"T{i + 1}" for i in range(n)]
    evaluations = []
    for phase, row in enumerate(retention):
        per_task: dict[str, dict[str, Any] | None] = {}
        for j, value in enumerate(row):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                per_task[task_ids[j]] = None
            else:
                per_task[task_ids[j]] = {
                    "success_rate": float(value),
                    "mean_return": 10.0 * float(value),
                    "mean_steps_to_goal": 200.0,
                    "n_episodes": 10,
                }
        evaluations.append(
            {"phase": phase, "after_training": task_ids[phase], "per_task": per_task}
        )
    return {
        "mission_name": mission_name,
        "cl_method": method,
        "seed": seed,
        "task_ids": task_ids,
        "evaluations": evaluations,
    }


# ---------------------------------------------------------------------------
# aggregate_retention_matrices
# ---------------------------------------------------------------------------


def test_aggregate_retention_matrices_mean_and_std() -> None:
    r1 = _results_from_matrix([[0.5, None], [0.3, 0.8]], seed=0)
    r2 = _results_from_matrix([[0.7, None], [0.4, 0.6]], seed=1)
    r3 = _results_from_matrix([[0.6, None], [0.5, 0.7]], seed=2)

    mean, std = aggregate_retention_matrices([r1, r2, r3])

    # Stack the matrices ourselves to verify against numpy's reference impl.
    stacked = np.array(
        [
            [[0.5, np.nan], [0.3, 0.8]],
            [[0.7, np.nan], [0.4, 0.6]],
            [[0.6, np.nan], [0.5, 0.7]],
        ]
    )
    expected_mean = np.nanmean(stacked, axis=0)
    expected_std = np.nanstd(stacked, axis=0, ddof=0)

    # NaN cells must propagate.
    assert np.isnan(mean[0, 1])
    assert np.isnan(std[0, 1])

    # Non-NaN cells must match nanmean/nanstd exactly.
    finite_mask = ~np.isnan(expected_mean)
    np.testing.assert_allclose(mean[finite_mask], expected_mean[finite_mask])
    np.testing.assert_allclose(std[finite_mask], expected_std[finite_mask])


def test_aggregate_raises_on_shape_mismatch() -> None:
    r_2x2 = _results_from_matrix([[0.5, None], [0.3, 0.8]])
    r_3x3 = _results_from_matrix(
        [[0.5, None, None], [0.3, 0.8, None], [0.2, 0.7, 0.9]]
    )
    with pytest.raises(ValueError, match=r"task_ids mismatch|shape"):
        aggregate_retention_matrices([r_2x2, r_3x3])


# ---------------------------------------------------------------------------
# collect_seed_results
# ---------------------------------------------------------------------------


def test_collect_seed_results_orders_by_seed(tmp_path: Path) -> None:
    method_dir = tmp_path / "naive"
    method_dir.mkdir()
    # Write seeds out of natural order on disk.
    for seed_value in (2, 0, 1):
        d = method_dir / f"seed_{seed_value}"
        d.mkdir()
        results = _results_from_matrix(
            [[0.5, None], [0.3, 0.8]], seed=seed_value
        )
        (d / "results.json").write_text(json.dumps(results))

    loaded = collect_seed_results(method_dir)
    assert [r["seed"] for r in loaded] == [0, 1, 2]


def test_collect_seed_results_empty_dir(tmp_path: Path) -> None:
    method_dir = tmp_path / "naive"
    method_dir.mkdir()
    assert collect_seed_results(method_dir) == []
    # Non-existent dir returns []
    assert collect_seed_results(tmp_path / "does_not_exist") == []


# ---------------------------------------------------------------------------
# plot_method_comparison_with_variance
# ---------------------------------------------------------------------------


def test_plot_method_comparison_with_variance_writes_png(tmp_path: Path) -> None:
    naive_seeds = [
        _results_from_matrix([[0.7, None], [0.4, 0.8]], method="naive", seed=s)
        for s in range(3)
    ]
    replay_seeds = [
        _results_from_matrix([[0.7, None], [0.6, 0.78]], method="replay", seed=s)
        for s in range(3)
    ]
    out = tmp_path / "comparison.png"
    fig = plot_method_comparison_with_variance(
        {"naive": naive_seeds, "replay": replay_seeds},
        task_ids=["T1", "T2"],
        out=out,
        metric="avg_retention",
    )
    try:
        assert out.exists()
        assert out.stat().st_size > 0
        ax = fig.axes[0]
        # bar() with yerr= produces error-bar caps/lines as Line2D objects.
        assert len(ax.lines) > 0
    finally:
        plt.close(fig)


def test_plot_handles_single_seed_method(tmp_path: Path) -> None:
    one_seed = [
        _results_from_matrix([[0.7, None], [0.4, 0.8]], method="naive", seed=0)
    ]
    out = tmp_path / "single_seed.png"
    fig = plot_method_comparison_with_variance(
        {"naive": one_seed},
        task_ids=["T1", "T2"],
        out=out,
        metric="avg_retention",
    )
    try:
        assert out.exists()
        assert out.stat().st_size > 0
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# summary.csv via the compare aggregator
# ---------------------------------------------------------------------------


def _seed_dirs_for(method_dir: Path, matrices: list[list[list[float | None]]],
                   *, method: str) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    for seed_value, mat in enumerate(matrices):
        d = method_dir / f"seed_{seed_value}"
        d.mkdir()
        (d / "results.json").write_text(
            json.dumps(_results_from_matrix(mat, method=method, seed=seed_value))
        )


@pytest.mark.parametrize("metric", ["avg_retention", "forgetting"])
def test_summary_csv_has_expected_columns(tmp_path: Path, metric: str) -> None:
    # Import here so the script's PROJECT_ROOT-relative sys.path adjustment
    # has been applied via the module-level sys.path.insert above.
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_scenario.py"
    spec = importlib.util.spec_from_file_location("run_scenario", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    scenario = "scenario_test"
    results_dir = tmp_path / "results"
    scen_dir = results_dir / scenario
    naive_matrices: list[list[list[float | None]]] = [
        [[0.7, None], [0.4, 0.8]],
        [[0.6, None], [0.3, 0.7]],
    ]
    replay_matrices: list[list[list[float | None]]] = [
        [[0.75, None], [0.6, 0.8]],
        [[0.65, None], [0.55, 0.75]],
    ]
    _seed_dirs_for(scen_dir / "naive", naive_matrices, method="naive")
    _seed_dirs_for(scen_dir / "replay", replay_matrices, method="replay")

    # Run the compare aggregator in-process.
    plot_path, csv_path = mod.compare(scenario, results_dir)
    assert plot_path.exists() and plot_path.stat().st_size > 0
    assert csv_path.exists() and csv_path.stat().st_size > 0

    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    columns = set(rows[0].keys())
    expected_cols = {
        "method",
        "n_seeds",
        "avg_retention_mean",
        "avg_retention_std",
        "forgetting_mean",
        "forgetting_std",
    }
    assert expected_cols.issubset(columns)

    rows_by_method = {row["method"]: row for row in rows}
    assert set(rows_by_method) == {"naive", "replay"}

    # Verify the means by direct math (independent of the implementation).
    naive_ar = np.array([0.4 + 0.8, 0.3 + 0.7]) / 2  # avg of final row
    replay_ar = np.array([0.6 + 0.8, 0.55 + 0.75]) / 2
    assert float(rows_by_method["naive"]["avg_retention_mean"]) == pytest.approx(
        float(np.mean(naive_ar))
    )
    assert float(rows_by_method["replay"]["avg_retention_mean"]) == pytest.approx(
        float(np.mean(replay_ar))
    )
    # forgetting per task = max retention column - last value.
    # naive seed 0: col0 max = 0.7, last = 0.4 -> 0.3. col1: max=0.8 last=0.8 -> 0.0. mean=0.15
    # naive seed 1: col0 max = 0.6, last = 0.3 -> 0.3. col1: max=0.7 last=0.7 -> 0.0. mean=0.15
    naive_fg = np.array([0.15, 0.15])
    # replay seed 0: col0 0.75->0.6 = 0.15, col1 0.8->0.8 = 0.0. mean=0.075
    # replay seed 1: col0 0.65->0.55 = 0.10, col1 0.75->0.75 = 0.0. mean=0.05
    replay_fg = np.array([0.075, 0.05])
    assert float(rows_by_method["naive"]["forgetting_mean"]) == pytest.approx(
        float(np.mean(naive_fg))
    )
    assert float(rows_by_method["replay"]["forgetting_mean"]) == pytest.approx(
        float(np.mean(replay_fg))
    )

    # n_seeds column is an integer.
    assert int(rows_by_method["naive"]["n_seeds"]) == 2
    assert int(rows_by_method["replay"]["n_seeds"]) == 2

    # The parametrized metric variable is consumed implicitly: the columns
    # checked above cover both metrics, so we just sanity-check it's known.
    assert metric in {"avg_retention", "forgetting"}
