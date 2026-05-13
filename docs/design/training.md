# Training

> Code: `src/rover_cl/cl/base.py`, `src/rover_cl/cl/naive.py`,
> `src/rover_cl/cl/replay.py`, `src/rover_cl/cl/ewc.py`,
> `src/rover_cl/missions/base.py`.

## PPO defaults

`cl/base.py::DEFAULT_PPO_KWARGS`:

```python
{
    "n_steps": 2048,                       # per-env rollout length per PPO update
    "batch_size": 128,
    "learning_rate": 3e-4,
    "gamma": 0.995,                        # effective horizon ≈ 200 steps
    "gae_lambda": 0.95,
    "policy_kwargs": {"net_arch": [128, 128]},
    "ent_coef": 0.01,                      # keeps policy stochastic; avoids "drive forward and freeze"
    "verbose": 0,
}
```

### Why these values

- **`gamma=0.995`** (up from SB3's `0.99` default). Episodes here run up
  to 1 500 steps; with `0.99`, the +50 goal bonus at step 500 backprops
  as ≈ +0.3 at step 0 — invisible signal. At `0.995` it's ≈ +4.
- **`net_arch=[128, 128]`** (up from `[64, 64]`). The 38-D obs with 8
  obstacle slots needs enough capacity to reason about gaps and
  combine with pose + velocity. Still tiny by RL standards (~30 K
  params).
- **`ent_coef=0.01`** (up from `0.0`). Without an exploration bonus PPO
  collapses into the first decent policy it finds, which on this env
  was "drive forward and stop". 0.01 is small enough to converge
  eventually but big enough that the policy stays stochastic during the
  early "discover the arc maneuver" phase.
- **`n_steps=2048`** (up from `512`). Better advantage estimates for the
  long episodes, especially when goal bonus arrives only at the end.

## Parallel rollout collection (`--n-envs`)

`missions/base.py::Runner.__init__(n_envs=N)` switches between two paths:

**Single-env (`n_envs=1`, the default and original behavior)**:
- Build one `RoverNavEnv` via `task.env_factory(seed)`.
- Pass to `cl.train_on(env, ...)`; the CL method calls
  `model.learn(...)` then (for EWC/Replay) does its post-training
  collection on the same env.

**Multi-env (`n_envs > 1`)**:
- Build N worker envs via `SubprocVecEnv([Monitor(env_factory(seed+i)) for i in range(N)])`.
  Each worker has a distinct seed so they collect independent rollouts.
  The `Monitor` wrap is required because SB3 reads `ep_info["r"]` from
  it; without it `model.learn()` crashes with `KeyError`.
- Call `cl.train_on(vec_env, ..., skip_post_train=True)`.
- Close the VecEnv.
- Build a fresh single env and call `cl.post_train(post_env, task_id)`.

On a Mac M3 Air (8 logical CPUs) the sweet spot is **4–6 workers**. With 4
you typically see ~50% CPU; the remaining workers are idling on the
main process between PPO updates. More workers (6–8) edges that up.

### Why CPU, not MPS / Neural Engine

The policy is a tiny [128, 128] MLP (~30 K params). PyTorch MPS has
non-trivial kernel-launch + host↔device transfer overhead per forward /
backward; for networks under ~1 M params this overhead exceeds the
saving. Empirical benchmarks (and SB3's own docs) recommend CPU for
small policies. NumPy here is already using Apple Accelerate BLAS (uses
NEON + AMX), so matrix ops are already native. The actual bottleneck is
**MuJoCo physics per env**, and that scales with parallel envs not GPU
compute.

## CL method hook structure

Three CL methods, all inheriting from `BaseCLMethod`:

| Method | `train_on` does | `post_train` does |
|---|---|---|
| `NaiveCL` | `model.learn(...)` only | (noop) |
| `ReplayCL` | `learn(...)` then BC-rehearse on past-task buffers (if any) | Collect `buffer_size_per_task` (obs, action) pairs into `self.buffers[task_id]`. |
| `EwcCL` | `learn(...)` then run `_apply_ewc_penalty(past_ids)` (SGD on `λ × Σ F × (θ − θ★)²` for each prior task) | Compute diagonal Fisher for the just-trained task; snapshot `θ★`. |

The split exists because the post-training collection (Fisher, buffer)
needs a **single** env with a deterministic step loop. When `--n-envs >
1` the Runner builds a SubprocVecEnv for `train_on` and a fresh single
env for `post_train` — the CL method itself doesn't need to know
whether its `train_on` env is vec or not.

`train_on(skip_post_train=True)` is the flag the Runner uses to tell
the CL method "don't call `self.post_train(env, …)` at the end; I'll
call it myself with a different env."

### Why `EwcCL` uses fresh SGD for the penalty pass

The original implementation reused PPO's Adam optimizer for the penalty
pass. Adam's running `v_t` is calibrated for tiny PPO gradients; when a
much-larger penalty gradient lands on top, Adam's per-parameter scaling
overshoots and pushes weights *away* from `θ★` instead of toward it.
Verified empirically: `drift_on / drift_off` went from 1.28 (broken,
pushing away) to 0.21 (working, pulling toward) after switching to a
fresh `SGD(lr=penalty_lr)` optimizer with gradient clipping. See
`cl/ewc.py::_apply_ewc_penalty`.

## CLI

`scripts/run_scenario.py` thin wrapper:

```bash
# basic
python scripts/run_scenario.py <scenario_name> --cl-method {naive|replay|ewc} \
    --train-steps 500000 --seeds 0,1,2

# parallel rollouts
python scripts/run_scenario.py <scenario_name> --n-envs 4 ...

# from a YAML config (overrides scenario / cl_method / seed)
python scripts/run_scenario.py --config configs/<file>.yaml --n-envs 4

# after running multiple methods, aggregate
python scripts/run_scenario.py <scenario_name> --compare
```

Per-seed results land in `results/<scenario>/<method>/seed_<N>/`:
`results.json`, `ckpt_phase_*.zip`, `report_phase_*.png`. The cross-seed
aggregation (mean ± std plus a comparison bar chart) writes to
`results/<scenario>/comparison.png` / `summary.csv` via the `--compare`
flag.
