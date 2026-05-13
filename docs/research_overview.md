# Mars-Rover Continual-Learning — Research Overview

This document describes what the project measures, how the rover perceives its
environment, and the end-to-end training/evaluation loop. It is grounded in the
actual code (not `CLAUDE.md`, which may drift). Every claim cites the file +
identifier so the reader can confirm in source.

Sources cross-referenced:
`assets/rover.xml`, `src/rover_cl/envs/nav.py`, `src/rover_cl/envs/terrains.py`,
`src/rover_cl/cl/base.py`, `src/rover_cl/cl/naive.py`, `src/rover_cl/cl/replay.py`,
`src/rover_cl/missions/base.py`, `src/rover_cl/missions/scenarios.py`,
`src/rover_cl/eval/metrics.py`, `src/rover_cl/viz/plots.py`,
`scripts/run_scenario.py`.

---

## 1. Stats we collect

### 1.1 Per-step `info` dict (from `RoverNavEnv.step`, `nav.py:280-285`)

Returned on every call to `env.step(...)`:

| Key                 | Type   | Meaning                                                  |
| ------------------- | ------ | -------------------------------------------------------- |
| `distance_to_goal`  | float  | L2 distance (m) from rover XY to terrain goal XY         |
| `is_success`        | bool   | True once `_goal_hold >= GOAL_HOLD_STEPS` (5 steps inside `goal_radius`) |
| `collision`         | bool   | True when any rover body contacts a geom whose name starts with `obs_` |
| `tipped`            | bool   | True when body-Z world component `< TIP_OVER_COS = 0.5`  |

On the terminal/truncated step, an extra key is added (`nav.py:286-292`):

| Key       | Type | Content                                                   |
| --------- | ---- | --------------------------------------------------------- |
| `episode` | dict | Serialized `EpisodeOutcome`: `success`, `steps`, `distance_to_goal`, `cumulative_reward` |

`EpisodeOutcome` is defined in `nav.py:36-41`. `cumulative_reward` is the sum
over the episode of the reward returned by `step` (see §2.3).

### 1.2 Per-evaluation `EpisodeStats` (from `evaluate_policy`, `metrics.py:56-143`)

`evaluate_policy` rolls out `n_episodes` deterministic episodes (capped at
`max_steps` per episode) and returns an `EpisodeStats` (defined `metrics.py:22-53`):

| Field                | Type        | How it's computed                                         |
| -------------------- | ----------- | --------------------------------------------------------- |
| `success_rate`       | float       | `mean(episode_success)` across `n_episodes`               |
| `mean_return`        | float       | `mean(sum(reward))` across episodes                       |
| `mean_steps_to_goal` | float\|None | mean step index at which `info["is_success"]` first fired; `None` if no episode succeeded |
| `n_episodes`         | int         | number of evaluation rollouts                             |

The success detector reads `info["is_success"]`; if the env never emits that
key it falls back to `last_reward > 0` (`metrics.py:113-125`).

### 1.3 Per-phase results (`PhaseResult`, `missions/base.py:46-59`)

After each training phase `i` the runner evaluates on **every task seen so far**
(`base.py:128-148`). Each phase produces one `PhaseResult`:

```python
PhaseResult(
    phase=i,                                      # 0-indexed
    after_training=task_ids[i],                   # which task was just trained
    per_task={task_id: EpisodeStats | None, ...}  # None where j > i
)
```

The aggregate `MissionResult` (`base.py:63-82`) contains:

| Field           | Type                 |
| --------------- | -------------------- |
| `mission_name`  | str                  |
| `cl_method`     | str (e.g. `"naive"`) |
| `seed`          | int                  |
| `task_ids`      | list[str]            |
| `evaluations`   | list[PhaseResult]    |

### 1.4 Derived metrics (`metrics.py:146-203`)

| Function                       | Output                       | Definition                                                            |
| ------------------------------ | ---------------------------- | --------------------------------------------------------------------- |
| `compute_retention_matrix`     | `np.ndarray[N,N]`            | `R[i,j] = success_rate` on task `j` evaluated after phase `i`; NaN where unseen |
| `compute_forgetting`           | `np.ndarray[N]`              | `forgetting[j] = max_k R[k,j] - R[last,j]` (over non-NaN cells)        |
| `compute_avg_retention`        | `float`                      | `nanmean(R[last, :])` — average final-row success rate                 |

The CLI prints the avg retention and per-task forgetting after each run
(`run_scenario.py:61-65`).

### 1.5 Artifacts on disk

Layout under `results/<scenario>/<method>/seed_<N>/`
(`run_scenario.py:45`, `missions/base.py:150-162`):

| Path                                         | Producer                                          |
| -------------------------------------------- | ------------------------------------------------- |
| `results.json`                               | `MissionResult.save` (`base.py:79-82`)            |
| `ckpt_phase_<i>_after_<task_id>.zip`         | `cl.save(...)` after each phase (`base.py:150-152`) |
| `<ckpt>.replay.npz` (replay only)            | `ReplayCL.save` sidecar (`replay.py:190-217`)     |
| `retention_matrix.png`                       | `plot_retention_matrix` (`plots.py:28-85`)        |
| `retention_curves.png`                       | `plot_retention_curves` (`plots.py:88-130`)       |
| `tb/` (only if `tensorboard` import succeeds) | SB3 PPO tensorboard logs (`missions/base.py:98-105`) |

At the scenario level (one up from method dirs):

| Path                                         | Producer                                          |
| -------------------------------------------- | ------------------------------------------------- |
| `results/<scenario>/comparison.png`          | `plot_method_comparison` via `--compare` (`run_scenario.py:69-88`) |

---

## 2. Sensors and how they interact with the world

There are **two distinct layers** of perception. The MJCF sensor block exists
for completeness/visualisation but the RL policy never sees it directly — the
env builds its own 14-D observation.

### 2.1 MJCF sensors (`assets/rover.xml`, `<sensor>` block, lines 272-288)

All 12 sensors are declared on the chassis body. Lidar sites sit on the chassis
front face at `(0, 0.95, 0.30)` with `xyaxes` rotated so each site's local +Z
casts in the horizontal plane (see `rover.xml:82-90`).

| Name           | MJCF type        | Dim | Site / object | Physical quantity                              | Notes |
| -------------- | ---------------- | --- | ------------- | ---------------------------------------------- | ----- |
| `imu_gyro`     | `gyro`           | 3   | `imu`         | Body angular velocity (rad/s)                  |       |
| `imu_accel`    | `accelerometer`  | 3   | `imu`         | Body proper acceleration (m/s²)                |       |
| `base_pos`     | `framepos`       | 3   | site `imu`    | World position of chassis (m)                  | Used by env |
| `base_quat`    | `framequat`      | 4   | site `imu`    | Orientation `(w, x, y, z)`                     | Used by env |
| `base_linvel`  | `framelinvel`    | 3   | site `imu`    | World-frame linear velocity (m/s)              | Used by env |
| `base_angvel`  | `frameangvel`    | 3   | site `imu`    | World-frame angular velocity (rad/s)           | Used by env |
| `lidar_m60`    | `rangefinder`    | 1   | `lidar_ray_m60` | Ray hit distance at −60° (m)                | **Returns −1 if no hit** |
| `lidar_m30`    | `rangefinder`    | 1   | `lidar_ray_m30` | Ray hit distance at −30° (m)                | −1 sentinel as above |
| `lidar_0`      | `rangefinder`    | 1   | `lidar_ray_0`   | Ray hit distance at 0° (forward) (m)        | −1 sentinel |
| `lidar_p30`    | `rangefinder`    | 1   | `lidar_ray_p30` | Ray hit distance at +30° (m)                | −1 sentinel |
| `lidar_p60`    | `rangefinder`    | 1   | `lidar_ray_p60` | Ray hit distance at +60° (m)                | −1 sentinel |
| `tool_pos`     | `framepos`       | 3   | site `tool_tip` | World position of arm tool tip (m)         | Not used by policy |

Cameras declared in the MJCF (lines 67-80):
- `chase` — third-person follow, `targetbody=base_link`, offset `(0, -4.5, 2.0)`.
- `navcam` — mast-mounted forward (+Y) camera, fovy 70°.

### 2.2 Env observation (`RoverNavEnv._build_obs`, `nav.py:154-174`)

The Gymnasium observation is **NOT** the raw MJCF sensor array. The env
synthesises a 14-D vector from `base_pos`/`base_quat`/`base_linvel`/`base_angvel`
plus an **independent 8-ray lidar fan** computed via `mj_ray` (the MJCF
rangefinders are not read at all by the policy).

Observation layout (`obs_dim = 6 + NUM_LIDAR_RAYS = 14`, `nav.py:91-94`):

| Index | Name              | Units | Sign / range                                  |
| ----- | ----------------- | ----- | --------------------------------------------- |
| 0     | `rel_fwd`         | m     | Signed forward distance to goal (+ = ahead)   |
| 1     | `rel_right`       | m     | Signed lateral offset (+ = to rover's right)  |
| 2     | `heading_to_goal` | rad   | `atan2(rel_right, rel_fwd)` ∈ (−π, π]         |
| 3     | `linvel_x`        | m/s   | World-frame x linear velocity                 |
| 4     | `linvel_y`        | m/s   | World-frame y linear velocity                 |
| 5     | `angvel_z`        | rad/s | Yaw rate                                      |
| 6..13 | `lidar[0..7]`     | —     | Normalised ray distance ∈ [0, 1] (= dist / `LIDAR_MAX_RANGE`) |

Lidar fan parameters (`nav.py:27-29`):
- `NUM_LIDAR_RAYS = 8`
- `LIDAR_MAX_RANGE = 8.0` m
- `LIDAR_FAN_DEG = 180.0` → rays are linearly spaced over ±90° around rover
  forward, **not the ±60° fan of the MJCF rangefinders**.
- A miss or out-of-range hit is clamped to `LIDAR_MAX_RANGE` then normalised,
  so a "saturated" reading is 1.0 — no −1 sentinel reaches the policy.

Action space (`nav.py:90`): `Box(-1, 1, (2,))` — skid-steer wheel commands.
`action[0]` drives the right-side wheels, `action[1]` drives the left-side
wheels; both are scaled by `MAX_WHEEL_VEL = 3.0` rad/s and **negated**
internally in `step` (`nav.py:235-243`) so that user-facing `action > 0` is
forward (+Y).

### 2.3 Reward (`RoverNavEnv.step`, `nav.py:270-277`)

```
reward =   progress_reward_scale * (prev_dist - dist)   # +ve on progress
         - step_cost                                    # constant per step
         - (collision_penalty   if collision else 0)
         - (tipped_penalty      if tipped    else 0)
         + (goal_bonus          if success   else 0)
```

Default coefficients (`nav.py:48-58`):

| Term                       | Symbol                   | Default | Trigger                              |
| -------------------------- | ------------------------ | ------- | ------------------------------------ |
| Progress shaping           | `progress_reward_scale`  | 1.0     | Every step (signed delta of dist)    |
| Step cost                  | `step_cost`              | 0.01    | Every step                           |
| Goal bonus                 | `goal_bonus`             | 50.0    | Once, when `_goal_hold >= 5`         |
| Collision penalty          | `collision_penalty`      | 1.0     | Step where an `obs_*` geom is touched|
| Tipped penalty             | `tipped_penalty`         | 20.0    | Step where body upright cos < 0.5    |

Termination is `success or tipped`; truncation is `step_count >= max_steps`
(`nav.py:266-268`).

---

## 3. The learning process, step by step

Concrete trace for:

```bash
source .venv/bin/activate
python scripts/run_scenario.py scenario_01_sequential_terrains \
    --cl-method replay --train-steps 30000 --seed 0
```

### Step 1 — scenario lookup

`get_scenario("scenario_01_sequential_terrains", cl_method="replay",
train_timesteps=30000, seed=0)` (`run_scenario.py:39-44`, dispatched via
`SCENARIO_REGISTRY` in `missions/scenarios.py:62-71`) builds a
`Mission("scenario_01_replay", tasks=[Task("T1_flat", ...),
Task("T2_corridor", ...)], cl_method="replay", seed=0)`. Each `Task` carries
its own `env_factory` (a closure returning a fresh `RoverNavEnv` for the
given terrain) and the budget `train_timesteps=30000, eval_episodes=10,
eval_max_steps=600` (`scenarios.py:25-38`).

### Step 2 — Runner setup

`Runner(mission, results_dir=results/scenario_01_sequential_terrains/replay/seed_0,
verbose=True)` (`run_scenario.py:45-47`).  The CL method is **not** built at
`Runner.__init__`; it is instantiated lazily inside `Runner.run` via
`make_cl(mission.cl_method, **mission.cl_kwargs)` (`missions/base.py:108`).

### Step 3 — Per-phase loop (`missions/base.py:113-152`)

For each phase `i = 0..N-1` (here N = 2):

1. **Fresh env**: `train_env = task.env_factory(seed + phase)` so terrain RNG
   varies across phases (`base.py:116`).
2. **Train**: `cl.train_on(train_env, total_timesteps=30000, task_id,
   log_dir=tb_dir)` (`base.py:117-122`).
   - **Phase 0**: `_ensure_model` (`cl/base.py:63-71`) constructs PPO:
     - policy = `"MlpPolicy"` with `net_arch=[64, 64]` (`base.py:13-21`)
     - `learning_rate=3e-4`, `n_steps=512`, `batch_size=64`,
       `gamma=0.99`, `gae_lambda=0.95`, `verbose=0`.
   - **Phase ≥ 1**: same PPO instance, `model.set_env(env)` swaps the env
     but keeps the policy / value weights (`cl/base.py:70-71`).
   - PPO runs with `reset_num_timesteps=False` so the global step counter
     is monotonic across tasks (TensorBoard continuity)
     (`cl/naive.py:26-30`, `cl/replay.py:78-82`).
3. **Replay-specific work** (`cl/replay.py:68-99`):
   - **After `learn(...)`**: if any *prior*-task buffer is non-empty,
     `_rehearse(past_ids)` does up to `rehearsal_steps=100` PPO grad steps of
     behavioural cloning: sample `rehearsal_batch_size=64` `(obs, action)`
     pairs uniformly from prior-task buffers, compute
     `loss = -log_prob(action | obs).mean()` via `policy.evaluate_actions`,
     backprop through the PPO optimizer (with LR temporarily scaled by
     `rehearsal_lr_scale=0.5`) (`replay.py:102-135`).
   - **Then collect**: roll out the policy on `train_env` deterministically
     for up to `buffer_size_per_task=1000` transitions and add them to a
     per-task `_TaskBuffer`. The buffer uses **reservoir sampling** once at
     capacity (`replay.py:35-46`) so it stays an unbiased sample of the
     stream (`_collect_into_buffer`, `replay.py:150-186`).
   - For phase 0 there is nothing to rehearse against, so only collection
     happens; `last_rehearsal_steps_run = 0`.
4. **Evaluate on every task seen so far** (`base.py:128-148`):
   - For each `j <= i`: build a fresh eval env with seed `seed + 1000 + j`,
     call `evaluate_policy(cl.predict(o, deterministic=True)[0], env,
     n_episodes=10, max_steps=600)`, store the returned `EpisodeStats` in
     `per_task[task_id]`.
   - For `j > i`: store `None`.
5. **Checkpoint**: `cl.save(results_dir / f"ckpt_phase_{i}_after_{task_id}.zip")`
   (`base.py:150-152`). For `ReplayCL`, the SB3 zip is accompanied by a
   `<...>.replay.npz` sidecar holding flattened transition tensors
   (`replay.py:190-217`).
6. Append a `PhaseResult` to `evaluations`.

### Step 4 — Final write & plots (`base.py:154-163`, `run_scenario.py:50-65`)

After the last phase, `MissionResult.save(results_dir/"results.json")` writes
the JSON. Then `run_scenario.run`:

- Reloads JSON via `load_results` (`metrics.py:206-210`).
- Builds `R = compute_retention_matrix(results)` — an N×N matrix where
  `R[i, j] = success_rate` on task `j` after phase `i` (NaN where unseen).
- Calls `plot_retention_matrix(R, task_ids, title=..., out=.../retention_matrix.png)`
  — a viridis heatmap, NaN cells masked to light grey, every cell annotated
  with its value (`plots.py:28-85`).
- Calls `plot_retention_curves(results, task_ids, out=.../retention_curves.png)`
  — one line per task showing how its success_rate evolves across phases
  (`plots.py:88-130`).
- Prints `avg retention` (`compute_avg_retention(R)`) and the per-task
  `forgetting` dict (`compute_forgetting(R)`).

### Step 5 — Cross-method comparison (`--compare`)

`python scripts/run_scenario.py scenario_01_sequential_terrains --compare`
runs `compare(scenario, results_dir)` (`run_scenario.py:69-88`):

1. Walk `results/<scenario>/<method>/seed_*/results.json` for each method dir.
2. Load the first seed found per method into `results_by_method`.
3. Call `plot_method_comparison(results_by_method, task_ids,
   out=results/<scenario>/comparison.png, metric="avg_retention")`
   (`plots.py:133-193`) — a bar chart of `compute_avg_retention(R)` per
   method, y-axis ∈ [0, 1.1].

### Tests

`pytest tests/ -q` exercises the full path (env, scenarios, CL methods,
metrics, plots) with tiny budgets.
