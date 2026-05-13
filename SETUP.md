# Setup

How to bring this project up from scratch on a new machine. Tested on macOS
M-series (Apple Silicon) and Linux x86_64; Windows users should use WSL2.

## Prerequisites

- **Python ≥ 3.13** (`pyproject.toml` pins `requires-python = ">=3.13,<4.0"`).
- **Git**.
- **macOS only**: Xcode command-line tools (for `clang`, needed by some
  scientific deps): `xcode-select --install`.
- **Linux only**: a working OpenGL stack for the MuJoCo viewer if you want
  to render. Headless training works without it.

## 1. Clone the repository

```bash
git clone git@github.com:CieciaOne/lander.git
cd lander
```

## 2. Create the virtual environment

The project uses Python's built-in `venv`. We pin Python 3.13.

```bash
# macOS / Linux
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## 3. Install dependencies

We have a `pyproject.toml` declaring all runtime + dev deps. The simplest
install path is:

```bash
pip install -e ".[dev]"
```

This installs the project in editable mode plus `pytest`. Key deps it
pulls (versions pinned in `pyproject.toml`):

- `gymnasium[mujoco] >= 0.29`
- `mujoco >= 3.5, < 4.0`
- `stable-baselines3 >= 2.0`
- `torch >= 2.0`
- `matplotlib >= 3.10`
- `numpy >= 1.24`
- `pyyaml`, `opensimplex`

## 4. Verify the install

```bash
pytest tests/ -q
```

Should print `87 passed` (or close to it — number grows as the project
evolves) in ~15 s. If anything fails, the error usually points at a
missing system lib or a MuJoCo install problem.

## 5. Run your first training scenario

The smallest scenario (`scenario_01_sequential_terrains`) trains a
two-task curriculum (T1_flat → T2_corridor) and writes results +
plots to `results/`.

```bash
python scripts/run_scenario.py scenario_01_sequential_terrains \
    --cl-method naive --train-steps 100000 --seed 0
```

Wall-clock: ~10 minutes on an M3 Air. Outputs land in
`results/scenario_01_sequential_terrains/naive/seed_0/`:
- `results.json` — retention matrix + timings.
- `ckpt_phase_<k>_after_<task>.zip` — SB3 PPO checkpoints.
- `report_phase_<k>_after_<...>_on_<...>.png` — top-down trajectory
  reports.
- `matrix.png` / `curves.png` — retention plots.

To use parallel rollout collection (4–8 MuJoCo workers in parallel):

```bash
python scripts/run_scenario.py scenario_01_sequential_terrains \
    --cl-method ewc --train-steps 500000 --n-envs 6 --seed 0
```

## 6. Visualize a trained policy

**macOS requires `mjpython`** (it ships with the `mujoco` pip package).
MuJoCo's passive viewer must own the main event loop, which only
`mjpython` provides on macOS.

```bash
# macOS:
mjpython scripts/visualize_rover.py \
    --policy results/scenario_01_sequential_terrains/naive/seed_0/ckpt_phase_1_after_T2_corridor.zip \
    --terrain-name T2_corridor

# Linux:
python scripts/visualize_rover.py \
    --policy results/scenario_01_sequential_terrains/naive/seed_0/ckpt_phase_1_after_T2_corridor.zip \
    --terrain-name T2_corridor
```

TAB cycles through cameras (tracking → chase → navcam → free).
Drop `--policy` for the scripted feature demo.

## Common gotchas

- **macOS viewer crash without `mjpython`**: the script will detect this
  and print a hint. Always use `mjpython` for the viewer on macOS;
  plain `python` is fine for training.
- **`xcode-select` not installed**: pip install of `torch` or `mujoco`
  may fail with a compiler error. Install Xcode CLI tools and retry.
- **Python 3.13 not available**: use [`pyenv`](https://github.com/pyenv/pyenv)
  to install it, or get it from Homebrew (`brew install python@3.13`).
- **`results/` is gitignored**: every run writes there, but those
  outputs don't get committed. Re-run training to regenerate.
- **Original rover source assets** (URDF, blender files, full mesh
  library) live in `data/` and are also gitignored — they're not
  loaded at runtime. The active rover MJCF is `assets/rover.xml`.

## Where to read about the project

- `README.md` — top-level summary.
- `CLAUDE.md` — operating doc with full architecture / rover-model
  details. Designed for an LLM agent but readable by humans.
- `docs/design/` — focused knowledge base: env design, observations,
  rewards, training pipeline, scenarios, evaluation, changelog.
- `docs/plan.md` — thesis roadmap and phase plan.
