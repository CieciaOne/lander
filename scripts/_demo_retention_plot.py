"""Generate a sample retention-matrix plot at /tmp/retention_demo.png.

Quick sanity check of the plotting stack:
    source .venv/bin/activate
    python scripts/_demo_retention_plot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from rover_cl.eval import compute_retention_matrix  # noqa: E402
from rover_cl.viz import (  # noqa: E402
    plot_method_comparison,
    plot_retention_curves,
    plot_retention_matrix,
)


def _synth(success_grid: list[list[float | None]], method: str) -> dict:
    task_ids = [f"T{i + 1}" for i in range(len(success_grid))]
    evaluations = []
    for phase, row in enumerate(success_grid):
        per_task: dict[str, dict | None] = {}
        for j, val in enumerate(row):
            per_task[task_ids[j]] = (
                None
                if val is None
                else {
                    "success_rate": float(val),
                    "mean_return": 10.0 * float(val),
                    "mean_steps_to_goal": 250.0 - 100.0 * float(val),
                    "n_episodes": 20,
                }
            )
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


def main() -> None:
    # Realistic-looking forgetting pattern: naive loses old tasks, replay holds on.
    naive_grid: list[list[float | None]] = [
        [0.72, None, None, None],
        [0.48, 0.81, None, None],
        [0.31, 0.55, 0.84, None],
        [0.22, 0.39, 0.66, 0.88],
    ]
    replay_grid: list[list[float | None]] = [
        [0.72, None, None, None],
        [0.68, 0.79, None, None],
        [0.62, 0.71, 0.82, None],
        [0.58, 0.65, 0.74, 0.85],
    ]
    naive = _synth(naive_grid, "naive")
    replay = _synth(replay_grid, "replay")
    task_ids = naive["task_ids"]

    retention = compute_retention_matrix(naive)
    plot_retention_matrix(
        retention,
        task_ids,
        title="sequential_terrains — naive (seed 0)",
        out=Path("/tmp/retention_demo.png"),
    )
    plot_retention_curves(naive, task_ids, out=Path("/tmp/retention_curves_demo.png"))
    plot_method_comparison(
        {"naive": naive, "replay": replay},
        task_ids,
        out=Path("/tmp/method_comparison_demo.png"),
        metric="avg_retention",
    )
    plot_method_comparison(
        {"naive": naive, "replay": replay},
        task_ids,
        out=Path("/tmp/method_comparison_forgetting_demo.png"),
        metric="forgetting",
    )
    print("wrote /tmp/retention_demo.png and three companion plots.")


if __name__ == "__main__":
    main()
