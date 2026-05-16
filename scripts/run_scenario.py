"""Run a continual-learning scenario end to end.

Two invocation styles are supported:

1. YAML config (preferred for reproducible experiments):
       source .venv/bin/activate
       python scripts/run_scenario.py --config configs/scenario_01_ewc.yaml
       python scripts/run_scenario.py --config configs/scenario_01_replay.yaml

   YAMLs may set either ``seed: 0`` or ``seeds: [0, 1, 2]``; the latter trains
   one mission per seed and aggregates at the end.

2. Legacy positional + flags:
       python scripts/run_scenario.py scenario_01_sequential_terrains \\
           --cl-method naive --train-steps 100000 --seed 0
       python scripts/run_scenario.py scenario_01_sequential_terrains \\
           --cl-method ewc  --train-steps 100000 --seeds 0,1,2

After multiple methods have results on the same scenario, compare with:
    python scripts/run_scenario.py --compare scenario_01_sequential_terrains

The comparison reads every ``seed_*`` directory it finds and writes both a
``comparison.png`` (mean +/- std bar chart) and a ``summary.csv`` of per-method
means and stds for downstream LaTeX / thesis processing.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from rover_cl.configs import load_missions_config  # noqa: E402
from rover_cl.eval import (  # noqa: E402
    collect_seed_results,
    compute_avg_retention,
    compute_forgetting,
    compute_retention_matrix,
    load_results,
)
from rover_cl.missions import Mission, Runner, get_scenario  # noqa: E402
from rover_cl.viz import (  # noqa: E402
    plot_method_comparison_with_variance,
    plot_retention_curves,
    plot_skill_survival,
    plot_retention_matrix,
)


def _scenario_label(mission: Mission) -> str:
    # mission.name is "<scenario>_<cl_method>"; strip the cl_method suffix so
    # results land under results/<scenario>/<cl_method>/seed_<N>/ regardless of
    # whether the mission came from a YAML or from CLI flags.
    suffix = f"_{mission.cl_method}"
    return mission.name[: -len(suffix)] if mission.name.endswith(suffix) else mission.name


def run_mission(mission: Mission, results_dir: Path, n_envs: int = 1,
                backend: str = "cpu", mjx_impl: str = "jax") -> Path:
    scenario_label = _scenario_label(mission)
    out_dir = results_dir / scenario_label / mission.cl_method / f"seed_{mission.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner(
        mission, results_dir=out_dir, verbose=True, n_envs=n_envs,
        backend=backend, mjx_impl=mjx_impl,
    )
    runner.run()

    results = load_results(out_dir / "results.json")
    retention = compute_retention_matrix(results)
    task_ids = results["task_ids"]

    plot_retention_matrix(
        retention, task_ids,
        title=f"{scenario_label} — {mission.cl_method} (seed {mission.seed})",
        out=out_dir / "retention_matrix.png",
    )
    plot_retention_curves(results, task_ids, out=out_dir / "retention_curves.png")
    plot_skill_survival(results, task_ids, out=out_dir / "skill_survival.png")

    print(f"\n=== Summary ({mission.cl_method}, seed {mission.seed}) ===")
    print(f"  avg retention: {compute_avg_retention(retention):.3f}")
    print(f"  forgetting per task: "
          f"{dict(zip(task_ids, compute_forgetting(retention).round(3).tolist()))}")
    print(f"  results: {out_dir / 'results.json'}")
    return out_dir


def _parse_seeds(spec: str) -> list[int]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(
            "--seeds must be a non-empty comma-separated list of integers, e.g. '0,1,2'"
        )
    try:
        return [int(p) for p in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--seeds must contain only integers, got {spec!r}"
        ) from e


def _build_missions_from_cli(
    scenario: str, cl_method: str, train_steps: int, seeds: list[int],
) -> list[Mission]:
    return [
        get_scenario(scenario, cl_method=cl_method, train_timesteps=train_steps, seed=s)
        for s in seeds
    ]


def _print_aggregated_summary(
    label: str, results_by_method: dict[str, list[dict]]
) -> None:
    print(f"\n=== Aggregated summary: {label} ===")
    print(f"{'method':<12} {'n_seeds':>7} "
          f"{'avg_ret mean':>13} {'avg_ret std':>12} "
          f"{'forget mean':>12} {'forget std':>11}")
    for method, seed_results in results_by_method.items():
        ar_vals: list[float] = []
        fg_vals: list[float] = []
        for res in seed_results:
            mat = compute_retention_matrix(res)
            ar_vals.append(compute_avg_retention(mat))
            fg = compute_forgetting(mat)
            fg_vals.append(float(np.nanmean(fg)) if fg.size else float("nan"))
        ar_arr = np.asarray(ar_vals, dtype=float)
        fg_arr = np.asarray(fg_vals, dtype=float)
        ar_mean = float(np.nanmean(ar_arr)) if ar_arr.size else float("nan")
        ar_std = float(np.nanstd(ar_arr, ddof=0)) if ar_arr.size else float("nan")
        fg_mean = float(np.nanmean(fg_arr)) if fg_arr.size else float("nan")
        fg_std = float(np.nanstd(fg_arr, ddof=0)) if fg_arr.size else float("nan")
        print(f"{method:<12} {len(seed_results):>7d} "
              f"{ar_mean:>13.3f} {ar_std:>12.3f} "
              f"{fg_mean:>12.3f} {fg_std:>11.3f}")


def run_missions(missions: list[Mission], results_dir: Path, n_envs: int = 1,
                 backend: str = "cpu", mjx_impl: str = "jax") -> None:
    """Train each mission and print an aggregated mean +/- std summary."""
    if not missions:
        raise SystemExit("run_missions: no missions to run")

    # All missions in one invocation share a scenario + cl_method (we never
    # build mixed batches here), so we can group them under one label.
    for m in missions:
        run_mission(m, results_dir, n_envs=n_envs, backend=backend, mjx_impl=mjx_impl)

    scenario_label = _scenario_label(missions[0])
    method = missions[0].cl_method
    method_dir = results_dir / scenario_label / method
    seed_results = collect_seed_results(method_dir)
    _print_aggregated_summary(
        f"{scenario_label} / {method}",
        {method: seed_results},
    )


def compare(scenario: str, results_dir: Path) -> tuple[Path, Path]:
    """Aggregate across methods and seeds. Returns (plot_path, csv_path)."""
    methods_dir = results_dir / scenario
    if not methods_dir.exists():
        raise SystemExit(f"No results for {scenario} in {results_dir}")

    results_by_method: dict[str, list[dict]] = {}
    task_ids: list[str] | None = None
    for method_dir in sorted(methods_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        seed_results = collect_seed_results(method_dir)
        if not seed_results:
            continue
        results_by_method[method_dir.name] = seed_results
        if task_ids is None:
            task_ids = list(seed_results[0]["task_ids"])

    if not results_by_method or task_ids is None:
        raise SystemExit(f"No results found under {methods_dir}")

    plot_out = methods_dir / "comparison.png"
    plot_method_comparison_with_variance(
        results_by_method, task_ids, out=plot_out, metric="avg_retention"
    )

    csv_out = methods_dir / "summary.csv"
    write_summary_csv(results_by_method, csv_out)

    _print_aggregated_summary(scenario, results_by_method)
    print(f"\ncomparison plot: {plot_out}")
    print(f"summary csv:     {csv_out}")
    return plot_out, csv_out


def write_summary_csv(
    results_by_method: dict[str, list[dict]], out_path: Path
) -> Path:
    """Write per-method mean/std of avg_retention and forgetting to CSV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, seed_results in results_by_method.items():
        ar_vals: list[float] = []
        fg_vals: list[float] = []
        for res in seed_results:
            mat = compute_retention_matrix(res)
            ar_vals.append(compute_avg_retention(mat))
            fg = compute_forgetting(mat)
            fg_vals.append(float(np.nanmean(fg)) if fg.size else float("nan"))
        ar_arr = np.asarray(ar_vals, dtype=float)
        fg_arr = np.asarray(fg_vals, dtype=float)
        rows.append({
            "method": method,
            "n_seeds": len(seed_results),
            "avg_retention_mean": (
                float(np.nanmean(ar_arr)) if ar_arr.size else float("nan")
            ),
            "avg_retention_std": (
                float(np.nanstd(ar_arr, ddof=0)) if ar_arr.size else float("nan")
            ),
            "forgetting_mean": (
                float(np.nanmean(fg_arr)) if fg_arr.size else float("nan")
            ),
            "forgetting_std": (
                float(np.nanstd(fg_arr, ddof=0)) if fg_arr.size else float("nan")
            ),
        })

    fieldnames = [
        "method",
        "n_seeds",
        "avg_retention_mean",
        "avg_retention_std",
        "forgetting_mean",
        "forgetting_std",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenario", nargs="?", default=None,
                    help="scenario name (see rover_cl.missions.scenarios); "
                         "ignored when --config is given")
    ap.add_argument("--config", type=Path, default=None,
                    help="Path to a YAML mission config. When set, --cl-method, "
                         "--train-steps, --seed, --seeds, and the positional "
                         "scenario are ignored.")
    ap.add_argument("--cl-method", default="naive",
                    choices=["naive", "replay", "ewc", "hybrid",
                             "l2", "mas", "distill"])
    ap.add_argument("--train-steps", type=int, default=100_000,
                    help="PPO timesteps per task. 30k learns ~nothing on this env, "
                         "100k gives a usable signal, 300k–1M is research-quality.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Single seed (ignored if --seeds is given).")
    ap.add_argument("--seeds", type=_parse_seeds, default=None,
                    help="Comma-separated list of seeds (e.g. '0,1,2'). "
                         "Overrides --seed when set; each seed trains and "
                         "writes to results/<scenario>/<method>/seed_<N>/, "
                         "then prints an aggregated mean +/- std summary.")
    ap.add_argument("--results-dir", type=Path,
                    default=PROJECT_ROOT / "results")
    ap.add_argument("--n-envs", type=int, default=1,
                    help="Number of parallel MuJoCo instances for PPO rollout "
                         "collection (SubprocVecEnv). 1 = current single-env "
                         "behavior. On Mac M3 (8 cores), 4 is a good default "
                         "and roughly halves wall-clock for the same number of "
                         "PPO updates. EWC / replay post-training collection "
                         "still uses a single env.")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "mjx"],
                    help="Rollout backend. 'cpu' (default) uses SubprocVecEnv "
                         "with native MuJoCo, one process per env. 'mjx' uses "
                         "MuJoCo-XLA — a single JAX-jitted batched env that "
                         "vectorises `--n-envs` rovers under one process. MJX "
                         "wins on GPU/TPU; on Mac CPU JAX it is slower than the "
                         "subproc path. Default: cpu.")
    ap.add_argument("--mjx-impl", default="jax", choices=["jax", "warp"],
                    help="MJX collision-pipeline backend. 'jax' (default) is "
                         "portable. 'warp' enables Nvidia's MuJoCo-Warp path — "
                         "requires Linux + CUDA + `pip install nvidia-warp` and "
                         "the mujoco-warp extras. Adds support for cylinder-mesh, "
                         "ellipsoid-cylinder, and ellipsoid-hfield collisions on "
                         "supported hardware. Ignored when --backend=cpu.")
    ap.add_argument("--compare", action="store_true",
                    help="Aggregate all CL methods + seeds for this scenario "
                         "into one plot (comparison.png) and one summary.csv.")
    args = ap.parse_args()

    if args.compare:
        if args.scenario is None:
            ap.error("--compare requires the positional scenario name")
        compare(args.scenario, args.results_dir)
        return

    if args.config is not None:
        missions = load_missions_config(args.config)
        run_missions(missions, args.results_dir, n_envs=args.n_envs,
                 backend=args.backend, mjx_impl=args.mjx_impl)
        return

    if args.scenario is None:
        ap.error("either --config or a positional scenario name is required")

    seeds = args.seeds if args.seeds is not None else [args.seed]
    missions = _build_missions_from_cli(
        args.scenario, args.cl_method, args.train_steps, seeds,
    )
    run_missions(missions, args.results_dir, n_envs=args.n_envs,
                 backend=args.backend, mjx_impl=args.mjx_impl)


if __name__ == "__main__":
    main()
