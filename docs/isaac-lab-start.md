# Getting started with Isaac (Isaac Lab) and how to think about it

> **NOTE — informational only, not the chosen stack.** This project runs on **MuJoCo + CPU** and targets a Mac M3 Air. Isaac Lab requires ≥16 GB NVIDIA VRAM, 32 GB RAM, and Ubuntu/Windows 11 — outside the resource budget. This doc is kept for reference in case the hardware situation changes. The actual implementation plan is in `docs/roadmap.md` and `docs/ergonomics_review.md`.

## What it is and why

- **Isaac Sim** — NVIDIA's 3D simulator (Omniverse): PhysX physics, USD scenes, rendering. This is the "engine".
- **Isaac Lab** — a framework for **training robots** (RL, imitation) **on the GPU**: thousands of copies of an environment run in parallel on a single card. It is a "layer" on top of Isaac Sim.

**Difference relative to a plain Gym + PyBullet setup:** instead of 1 env on the CPU you have **vectorisation**: e.g. 4096 envs on the GPU; one simulation step = one step across all envs. RL training is much faster (data collected in parallel).

**Isaac Gym** is deprecated; **use Isaac Lab**.

---

## Mental model

1. **Application** — everything starts via the **Isaac Sim** application (an Omniverse app). Training scripts first launch `AppLauncher` and only afterwards import the rest of Isaac Lab. Without the "application" running there is no simulation.

2. **Configuration, not just code** — environments are described by **configuration classes** (`@configclass`): number of envs, dt, robot, rewards, limits. Changing behaviour = changing the config (often from the CLI: `--num_envs`, `--task`).

3. **Two env modes**
   - **Direct** — everything in a single env definition (quick start, custom task).
   - **Manager-based** — scenes, robots, sensors, rewards as separate "managers" (modularity, swappable robots / sensors).

4. **Robots** — these are **configurations** (e.g. `ArticulationCfg`): a USD file + initial state + actuator definitions. A custom robot = a new configuration file (and optionally a USD model).

5. **Gymnasium** — envs are registered in Gymnasium (`gym.register(...)`). Training is run through the Isaac Lab scripts (`train.py` for SKRL, RL-Games, SB3, etc.) selecting `--task=Isaac-Name-v0`.

---

## Requirements (Isaac Lab + Isaac Sim 5.x)

- **OS:** Ubuntu 22.04 or Windows 11
- **Python:** 3.11 (must match the Isaac Sim version)
- **GPU:** NVIDIA, **≥16 GB VRAM**, current driver (580+ recommended)
- **RAM:** ≥32 GB

Without a capable NVIDIA card Isaac Lab does not make sense (it is GPU-optimised).

---

## Installation (quick start)

### 1. Python 3.11 environment

```bash
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
```

(Or `uv venv --python 3.11` / `python3.11 -m venv ...`.)

### 2. PyTorch with CUDA

```bash
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

(Match `cu128` to your CUDA version, e.g. `cu124`.)

### 3. Isaac Sim (pip)

```bash
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

### 4. Isaac Lab (from source)

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install    # Linux
# or: isaaclab.bat --install   # Windows
```

### 5. Test

From the `IsaacLab` directory:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task=Isaac-Cartpole-Direct-v0 --num_envs=1024 --headless
```

`--headless` = no window; drop `--headless` to see one.

---

## How to use it: typical workflow

1. **List envs**
   ```bash
   ./isaaclab.sh -p scripts/environments/list_envs.py
   ```
   You will see e.g. `Isaac-Cartpole-Direct-v0`, `Isaac-Ant-v0`, etc. with their `EntryPoint` and `Config`.

2. **Training**
   - **SKRL:** `./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task=Isaac-Ant-v0 --num_envs=4096`
   - **RL-Games:** `scripts/reinforcement_learning/rl_games/train.py --task=...`
   - **SB3:** via a script in `scripts/reinforcement_learning/stable_baselines3/` (if present) or a custom runner that, after `AppLauncher`, creates an env through `gym.make("Isaac-...-v0")`.

3. **Replay / evaluation**
   ```bash
   ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task=Isaac-Ant-v0 --load_run=...
   ```

4. **Custom environment**
   - **Quick skeleton:** project generator:
     ```bash
     ./isaaclab.sh --new
     ```
     Choose: External, Direct (or Manager), framework (e.g. SKRL). A template with an env and Gymnasium registration will be created.
   - In the template: env class (e.g. `*Env`) + configuration class (`*EnvCfg`). The robot in the env is a configuration (USD + `ArticulationCfg`).
   - After changes: `pip install -e source/<project_name>` and train with `--task=YourEnv-v0`.

---

## Where to find things

| You want to… | Where |
|--------------|-------|
| Install step by step | [Isaac Lab – Local Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), [Quickstart](https://isaac-sim.github.io/IsaacLab/main/source/setup/quickstart.html) |
| Understand configurations and vectorisation | Quickstart → "Configurations", "Robots" sections |
| Add a custom robot | [Add new robot](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/01_assets/add_new_robot.html) (USD + `ArticulationCfg`) |
| Create a custom env (Direct) | [Create Direct RL env](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html) |
| Train without your own machine | [Isaac Launchable](https://github.com/isaac-sim/isaac-launchable) (NVIDIA Brev, browser + VS Code) |

---

## Relation to this project (rover, CL)

- **Current stack:** MuJoCo + Gymnasium + Stable-Baselines3 PPO on the CPU. Env in `src/rover_cl/envs/nav.py`; rover MJCF in `assets/rover.xml`. No Isaac, no GPU.
- **Hypothetical move to Isaac Lab** (only relevant if hardware constraints change):
  - The env would be re-implemented as `DirectRLEnv` or Manager-based; observations / actions / rewards rewritten in the Isaac Lab format.
  - Rover: URDF imported into USD + `ArticulationCfg`. The current MJCF would not be reused directly.
  - Training would still use PPO, but via `AppLauncher` + Isaac Lab envs (`gym.make("Isaac-Rover-...")`).
  - CL (EWC, replay, hybrid) logic is portable — only the env layer changes; vectorized `num_envs` on GPU could speed up training significantly.

**In practice:** stay on MuJoCo for the thesis. Adding Isaac Lab would only make sense if (a) training time becomes a hard blocker AND (b) a suitable GPU becomes available.
