"""Re-evaluate saved checkpoints to regenerate the per-task report PNGs.

Use this when you want to apply newer plot styling / trajectory-capture
without retraining. Loads each checkpoint in a results directory, evaluates
it on every task in the mission with the current `evaluate_with_trajectories`
+ `plot_run_report` pipeline, and overwrites the PNGs in place.

The retention numbers (results.json) come from the original training run and
are not modified — only the plots.

Example:
    python scripts/regenerate_reports.py \
        results/scenario_10_robust_curriculum/ewc/seed_0/

If `--scenario` is omitted, the script reads it from results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from rover_cl.cl import make_cl  # noqa: E402
from rover_cl.eval.metrics import evaluate_with_trajectories  # noqa: E402
from rover_cl.missions import get_scenario  # noqa: E402
from rover_cl.viz.plots import plot_run_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("seed_dir", type=Path,
                    help="Path to a results/<scenario>/<method>/seed_<N>/ dir")
    ap.add_argument("--scenario", default=None,
                    help="Scenario name (default: read from results.json mission_name)")
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=1500)
    args = ap.parse_args()

    seed_dir: Path = args.seed_dir
    if not seed_dir.is_dir():
        ap.error(f"{seed_dir} is not a directory")

    results_path = seed_dir / "results.json"
    if not results_path.exists():
        ap.error(f"missing {results_path}")
    results = json.loads(results_path.read_text())

    # Mission name is "<scenario>_<cl_method>"; strip the cl_method suffix to
    # recover the scenario factory key.
    scenario_name = args.scenario
    cl_method = results["cl_method"]
    if scenario_name is None:
        mission_name = results["mission_name"]
        suffix = f"_{cl_method}"
        scenario_name = (mission_name[: -len(suffix)]
                         if mission_name.endswith(suffix) else mission_name)

    seed = int(results["seed"])
    task_ids: list[str] = list(results["task_ids"])
    print(f"Loaded results from {results_path}")
    print(f"  scenario={scenario_name}, cl_method={cl_method}, seed={seed}")
    print(f"  tasks: {task_ids}")

    mission = get_scenario(scenario_name, cl_method=cl_method, seed=seed,
                           eval_episodes=args.eval_episodes,
                           max_steps=args.max_steps)

    cl_cls = make_cl(cl_method).__class__
    for phase, task in enumerate(mission.tasks):
        ckpt = seed_dir / f"ckpt_phase_{phase}_after_{task.task_id}.zip"
        if not ckpt.exists():
            print(f"[skip] phase {phase}: no checkpoint at {ckpt}")
            continue
        print(f"\n[phase {phase}] loading {ckpt.name}")
        cl = cl_cls.load(ckpt)

        for j, other in enumerate(mission.tasks):
            if j > phase:
                continue  # task wasn't seen at this phase; matches Runner logic
            eval_env = other.env_factory(seed + 10000 * (phase + 1) + 100 * j)
            try:
                stats, trajectories = evaluate_with_trajectories(
                    policy_predict_fn=lambda o: cl.predict(o, deterministic=True)[0],
                    env=eval_env,
                    n_episodes=args.eval_episodes,
                    max_steps=args.max_steps,
                    seed_base=seed + 10000 * (phase + 1) + 100 * j,
                )
                terrain_spec = getattr(eval_env.unwrapped, "terrain", None)
                if terrain_spec is None:
                    continue
                out_path = (seed_dir
                            / f"report_phase_{phase}_after_{task.task_id}_on_{other.task_id}.png")
                plot_run_report(
                    terrain=terrain_spec,
                    trajectories=trajectories,
                    out=out_path,
                    title=f"{mission.name} | phase {phase} (after {task.task_id}) "
                          f"| eval on {other.task_id}",
                )
                plt.close("all")
                print(f"  wrote {out_path.name}  (success={stats.success_rate:.2f})")
            finally:
                try:
                    eval_env.close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
