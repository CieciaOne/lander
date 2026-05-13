# Master's thesis — Continual Learning for rover autonomy and security

Empirical analysis of continual learning techniques in models that learn new information on the fly: **(1)** RL navigation of a Mars rover (rocker-bogie suspension, 6-wheel drive, 4-corner steering) on a sequence of terrains in MuJoCo, **(2)** supervised threat classification on a sequence of threat classes, **(3)** a multi-task fusion phase with a shared encoder. CL methods currently implemented: `NaiveCL` (fine-tune baseline) and `ReplayCL` (rollout buffer + BC rehearsal). EWC and the supervised threat-classification track are on the roadmap — see `docs/roadmap.md`.

The authoritative experiment plan lives in [`docs/plan.md`](docs/plan.md). Guidance for working in this repository with Claude Code lives in [`CLAUDE.md`](CLAUDE.md).

## Goal

Answer the three research questions (`RQ1`–`RQ3`) defined in `docs/plan.md`:

- **RQ1** — do EWC, replay, and their hybrid limit catastrophic forgetting in (a) RL navigation across terrains and (b) supervised threat classification across classes?
- **RQ2** — is a shared encoder across both tasks worthwhile compared to two separate models?
- **RQ3** — under realistic on-board memory budgets, which CL configuration is preferable?

## Rover model

The simulated rover (`assets/rover.xml`) approximates the Curiosity / Perseverance class of Mars rovers:

- **6-wheel drive**, velocity-controlled (skid-steer compatible).
- **4-corner steering** (front-left, front-right, rear-left, rear-right); the 2 middle wheels are fixed-direction, matching the real rover.
- **Passive Λ-shaped rocker-bogie suspension** with a transverse **differential** coupling the two rockers (chassis pitch = average of rocker pitches).
- **4-DOF actuated arm** (yaw / shoulder / elbow / wrist), position-controlled.
- **5-ray lidar fan** sweeping ±60° around the rover's forward direction (returns −1 on miss).
- **Two cameras**: forward-facing mast camera (`navcam`) and third-person `chase` camera; the viewer adds a tracking camera by default.

See [`CLAUDE.md`](CLAUDE.md) for the full actuator / sensor map and the forward-direction sign convention.

## Requirements

- Python 3.13+
- [Poetry](https://python-poetry.org/) for dependency management (installed inside the project venv, not globally)
- MuJoCo (installed via the `gymnasium[mujoco]` and `mujoco` packages declared in `pyproject.toml`)

## Install

The virtualenv is checked out at `.venv/` inside the repository. Activate it before running anything:

```bash
cd praca-magisterska
source .venv/bin/activate

# install / refresh dependencies
poetry install
```

If you prefer not to activate the venv, prefix commands with `.venv/bin/`:

```bash
.venv/bin/poetry install
.venv/bin/python scripts/run_scenario.py --help
```

The Python dependencies (Gymnasium + MuJoCo, Stable-Baselines3, PyTorch, NumPy, Matplotlib, PyYAML, pytest) are declared in [`pyproject.toml`](pyproject.toml).

## Run

CLI entry points live under `scripts/`:

```bash
# always activate the venv first
source .venv/bin/activate

# run a CL scenario end-to-end (training + retention plots)
python scripts/run_scenario.py scenario_01_sequential_terrains \
    --cl-method naive  --train-steps 30000 --seed 0
python scripts/run_scenario.py scenario_01_sequential_terrains \
    --cl-method replay --train-steps 30000 --seed 0

# bar chart comparing CL methods within a scenario
python scripts/run_scenario.py scenario_01_sequential_terrains --compare

# interactive viewer (macOS requires mjpython instead of python)
mjpython scripts/visualize_rover.py   # macOS
python   scripts/visualize_rover.py   # Linux
```

Tests:

```bash
pytest
```

## Project structure

```
praca-magisterska/
├── src/rover_cl/         # main Python package
│   ├── envs/             # Gymnasium + MuJoCo rover environments (terrains)
│   ├── cl/               # continual-learning primitives (EWC, replay, hybrid)
│   ├── missions/         # task / scenario orchestration (terrain & class sequences)
│   ├── viz/              # plotting, env rendering helpers
│   └── eval/             # metrics: retention, forgetting, SPL, F1, ...
├── assets/               # rover MJCF + meshes
│   ├── rover.xml         # rocker-bogie rover MJCF
│   └── meshes/           # STL / OBJ meshes referenced by the MJCF
├── scripts/              # CLI entry points (run_scenario.py, visualize.py)
├── configs/              # YAML scenario configurations
├── tests/                # pytest test suite
├── docs/                 # research plan and supporting notes
├── stage01/              # CL scenario definitions (mapped to plan phases)
├── results/              # experiment outputs (gitignored)
├── pyproject.toml
├── CLAUDE.md             # Claude Code guidance for this repo
└── README.md
```

## Documents

| File | Description |
|------|-------------|
| [`docs/plan.md`](docs/plan.md) | Authoritative experiment plan (Phases 0–7), deliverables |
| [`docs/continual_learning_one_pager.md`](docs/continual_learning_one_pager.md) | Short CL one-pager (research context) |
| [`docs/alternatives-stack.md`](docs/alternatives-stack.md) | Technology-alternative analysis (Python vs Rust, Isaac Lab, MJX) |
| [`docs/isaac-lab-start.md`](docs/isaac-lab-start.md) | Getting-started notes for Isaac Lab |
| [`docs/resources-used.md`](docs/resources-used.md) | External resources (assets, libraries, papers) |
| [`stage01/README.md`](stage01/README.md) | Scenario index and mapping onto plan phases |
| [`CLAUDE.md`](CLAUDE.md) | Claude Code guidance for working in this repository |

## License

Private — academic thesis project.
