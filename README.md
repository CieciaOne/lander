# Master's thesis — Continual Learning for Mars rover navigation

Empirical analysis of continual learning techniques applied to PPO-trained
Mars rover navigation in MuJoCo. The rover is a rocker-bogie 6-wheel drive
on the Curiosity / Perseverance template; the policy outputs 2-D Ackermann
(throttle, steer). The CL question: when a single policy is trained
sequentially across phases (e.g. locomotion → path-following → obstacle
avoidance → terrain), how well do the various CL methods preserve earlier
skills?

The thesis compares **7 CL methods** against a **joint-training baseline**
on a **4-phase integrated curriculum**.

For day-to-day work guidance see [`CLAUDE.md`](CLAUDE.md). For the
authoritative experiment plan see [`docs/plan.md`](docs/plan.md). For design
rationale on individual components see [`docs/design/`](docs/design/).

## What's in this repo

- A configurable MuJoCo rover env (`src/rover_cl/envs/`)
- 7 continual-learning methods (`src/rover_cl/cl/`)
- A scenario / mission orchestrator that runs them end-to-end
  (`src/rover_cl/missions/`)
- Evaluation metrics + thesis-style plotting (`src/rover_cl/eval/`,
  `src/rover_cl/viz/`)
- An optional GPU-accelerated rollout backend via MuJoCo XLA
  (`src/rover_cl/envs/nav_mjx.py`, `mjx_vec_env.py`)
- Tests (~150 tests, ~45 s on Mac CPU)

## CL methods

| Method | Mechanism |
|---|---|
| **naive** | No CL protection; fine-tune baseline (the control). |
| **replay** | Per-task buffer of (obs, action). BC rehearsal before each new task. |
| **ewc** | Diagonal Fisher per task. L2 penalty toward `θ*` weighted by Fisher. |
| **l2** | Uniform-weight L2 (no Fisher). The dumb-baseline proving EWC's Fisher matters. |
| **mas** | Memory Aware Synapses (Aljundi 2018). Importance = `E[|∂out/∂param|]`. |
| **distill** | Frozen teacher per task. Student matches teacher via KL on stored obs. |
| **hybrid** | EWC + Replay together. Strongest practical combo. |

## Scenarios

The interesting ones for the thesis:

- **`scenario_13_integrated_curriculum`** — 4 phases, no-single-skill-isolation
  design. Locomotion → path+obstacles → +terrain → full random. This is the
  curriculum design that actually works.
- **`scenario_12_joint_training`** — single-phase training on `RC_full_random`
  for many timesteps. The joint-training upper bound that CL methods are
  measured against, AND the deployable >90% candidate.
- **`scenario_11_robust_generalist`** — 7-phase mixed-distribution curriculum.
  Earlier design, still useful as a comparison.
- `scenario_10_robust_curriculum` — 13-phase rich curriculum. Demonstrates
  catastrophic forgetting in pure single-skill phases (and is the reason
  scenario_13 exists).
- `scenario_01`-`scenario_09` — smaller / earlier studies on the original
  T1-T6 terrains. Kept for the comparison set.

See [`docs/design/scenarios.md`](docs/design/scenarios.md) for each scenario's
research question and what it measures.

## Quickstart

Requires Python 3.13+. Inside the repo:

```bash
source .venv/bin/activate     # already provisioned via Poetry
poetry install                # if you need to refresh deps
```

Run the integrated curriculum with hybrid CL across 3 seeds:

```bash
python scripts/run_scenario.py scenario_13_integrated_curriculum \
    --cl-method hybrid --seeds 0,1,2 --train-steps 600000 --n-envs 6
```

Joint-training baseline (long single run, no CL):

```bash
python scripts/run_scenario.py scenario_12_joint_training \
    --seeds 0 --train-steps 5000000 --n-envs 6
```

After running multiple `--cl-method` choices on the same scenario,
generate a comparison bar chart:

```bash
python scripts/run_scenario.py scenario_13_integrated_curriculum --compare
```

Interactive 3D viewer for the rover:

```bash
mjpython scripts/visualize_rover.py        # macOS — MuJoCo viewer needs main thread
python   scripts/visualize_rover.py        # Linux
```

Visualize policy checkpoints from a completed run:

```bash
mjpython scripts/visualize_all_phases.py \
    --scenario scenario_13_integrated_curriculum --seed 0
```

Run tests:

```bash
pytest
```

## Mac vs Linux + GPU

Default `--backend cpu` (SubprocVecEnv with native MuJoCo) works on both.

On a CUDA Linux box (e.g. RTX 3060 Ti), `--backend mjx --n-envs 256` runs
many rovers in parallel under one JAX process for a roughly 5-30× wall-clock
speedup on full multi-seed runs. First-run JIT compile is slow (5-15 minutes);
set `JAX_COMPILATION_CACHE_DIR=~/.cache/jax-compile` so subsequent runs reuse
the compiled artifacts. The MJX path requires `pip install -e ".[mjx]"`
plus `pip install -U "jax[cuda12]"`.

On Mac the MJX path works (via CPU JAX) but is slower than SubprocVecEnv —
use it for testing the GPU path locally, not for thesis-scale runs.

## Project structure

```
praca-magisterska/
├── src/rover_cl/         # main package
│   ├── envs/             # RoverNavEnv + terrain catalog + MJX backend
│   ├── cl/               # 7 CL methods, all subclassing BaseCLMethod
│   ├── missions/         # Task / Mission / Runner / scenario registry
│   ├── viz/              # matplotlib plots (retention, skill survival, reports)
│   └── eval/             # success_rate, retention matrix, forgetting
├── assets/
│   └── rover.xml         # authoritative rocker-bogie MJCF
├── scripts/              # CLI entry points
│   ├── run_scenario.py       # main training / eval / comparison runner
│   ├── visualize_rover.py    # rover feature demo + policy replay viewer
│   ├── visualize_all_phases.py # cycle through phase checkpoints
│   └── regenerate_reports.py # re-render report PNGs from saved results
├── tests/                # pytest suite (~150 tests, ~45 s)
├── docs/
│   ├── plan.md           # research plan + RQs
│   ├── roadmap.md        # gaps and planned work
│   ├── research_overview.md  # stats, sensors, train→eval loop walkthrough
│   ├── ergonomics_review.md  # ranked CLI / script friction points
│   └── design/           # per-component design notes
├── configs/              # YAML scenario configurations (optional)
├── stage01/              # CL scenario index (maps to plan phases)
├── results/              # experiment outputs (gitignored)
├── pyproject.toml
├── CLAUDE.md             # Claude Code guidance
└── README.md
```

## Documents

| File | What it covers |
|------|----------------|
| [`docs/plan.md`](docs/plan.md) | Authoritative experiment plan (Phases 0-7) |
| [`docs/roadmap.md`](docs/roadmap.md) | Gaps, planned terrain, scenarios, multi-seed |
| [`docs/research_overview.md`](docs/research_overview.md) | Sensors, env stats, train→eval loop |
| [`docs/ergonomics_review.md`](docs/ergonomics_review.md) | CLI improvement ranking |
| [`docs/design/scenarios.md`](docs/design/scenarios.md) | Per-scenario design rationale |
| [`docs/design/environment.md`](docs/design/environment.md) | Rover geometry, suspension, friction |
| [`docs/design/rewards.md`](docs/design/rewards.md) | Reward-term history |
| [`docs/design/training.md`](docs/design/training.md) | PPO + CL training pipeline |
| [`docs/design/evaluation.md`](docs/design/evaluation.md) | Eval reports + metrics |
| [`stage01/README.md`](stage01/README.md) | Scenario index |
| [`CLAUDE.md`](CLAUDE.md) | Code architecture + working conventions |

## License

Private — academic thesis project.
