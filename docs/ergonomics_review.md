# Ergonomics review — rover-cl

Scope: a researcher-perspective audit of the CLI, configuration, scenario authoring,
visualization, logging, and result layout as of 2026-05-13. Source files reviewed:
`scripts/run_scenario.py`, `scripts/visualize_rover.py`, `src/rover_cl/missions/{base.py,scenarios.py}`,
`src/rover_cl/cl/{base.py,naive.py,replay.py,__init__.py}`, `src/rover_cl/envs/{terrains.py,nav.py}`,
`src/rover_cl/viz/plots.py`, `src/rover_cl/eval/metrics.py`, `configs/README.md`,
`pyproject.toml`, `README.md`, `docs/plan.md`, `stage01/scenarios/*`.

This review is intentionally implementation-light. Each finding cites file paths and
sketches a fix; nothing here is a write-the-code task.

---

## 1. Current happy path (concrete)

### 1.1 Train one scenario / one CL method / one seed

```bash
source /Users/jakubciecka/praca-magisterska/.venv/bin/activate
python scripts/run_scenario.py scenario_01_sequential_terrains \
    --cl-method naive --train-steps 30000 --seed 0
```

Files touched: none — uses the registered scenario in
`src/rover_cl/missions/scenarios.py:62` (`SCENARIO_REGISTRY`).
Output: `results/scenario_01_sequential_terrains/naive/seed_0/{results.json, retention_matrix.png, retention_curves.png, ckpt_phase_*.zip}`.

### 1.2 Compare two CL methods (single seed each)

```bash
python scripts/run_scenario.py scenario_01_sequential_terrains --cl-method naive  --seed 0
python scripts/run_scenario.py scenario_01_sequential_terrains --cl-method replay --seed 0
python scripts/run_scenario.py scenario_01_sequential_terrains --compare
```

Note: `--compare` (`scripts/run_scenario.py:69-88`) silently picks
**`seed_dirs[0]`** per method — i.e. one seed only. There is no aggregation.

### 1.3 Visualize a trained policy

- **Does not work today.** `scripts/visualize_rover.py` is a hard-coded scripted
  demo (phases: settle / forward / Ackermann / spin / arm sweep — see
  `make_schedule()` at line 152). It never loads a checkpoint and has no
  `--policy` flag.
- Closest thing: `RoverNavEnv.render()` exists (`src/rover_cl/envs/nav.py:295`)
  with `render_mode="rgb_array"` but is never wired to the viewer or to
  `run_scenario.py`.

### 1.4 Add a new terrain

1. Edit `src/rover_cl/envs/terrains.py`: add a `terrain_T4_<name>(seed)` function
   returning a `TerrainSpec` (template: `terrain_T3_obstacle_field` at line 158).
2. Register in `TERRAIN_CATALOG` (`terrains.py:179`).
3. Reference the new terrain ID in a scenario (next section). No CLI / YAML / test
   changes required, but `tests/test_env.py` iterates `TERRAIN_CATALOG`, so the
   new terrain is automatically smoke-tested.

### 1.5 Add a new scenario

1. Edit `src/rover_cl/missions/scenarios.py`: add a `scenario_NN_<name>(...)`
   function returning a `Mission` (template: `scenario_02_three_terrains` at
   line 41).
2. Register it in `SCENARIO_REGISTRY` (`scenarios.py:62`).
3. Optionally write a markdown spec under `stage01/scenarios/`.

There is **no YAML path** — the README example
`python scripts/run_scenario.py --config configs/<scenario>.yaml` in
`README.md:64` is aspirational; that flag does not exist in
`scripts/run_scenario.py:91-101`. README is misleading on this point.

---

## 2. Friction points

Ordered roughly by how often a researcher will hit them.

### F1. Hyperparameters live in code, not config

- **Where:** `src/rover_cl/cl/base.py:13-21` (`DEFAULT_PPO_KWARGS`:
  `n_steps=512`, `batch_size=64`, `lr=3e-4`, `net_arch=[64, 64]`),
  `src/rover_cl/cl/replay.py:53-58` (`buffer_size_per_task=1000`,
  `rehearsal_batch_size=64`, `rehearsal_steps=100`, `rehearsal_lr_scale=0.5`),
  `src/rover_cl/missions/scenarios.py:18-24` (`train_timesteps`,
  `eval_episodes`, `max_steps`), `src/rover_cl/envs/nav.py:48-60`
  (reward shape, `progress_reward_scale`, `goal_bonus`,
  `collision_penalty`, `step_cost`, `tipped_penalty`, `tip_over_cos`).
- **Why it hurts:** every λ / buffer-size / lr sweep requires either a code edit
  or a custom call site. The plan calls for "EWC λ ∈ {1e2, 1e3, 1e4}, replay
  buffer ∈ {5%, 10%, 20%}" (`docs/plan.md:101`) — that is the central
  experiment of the thesis and there is no way to drive it from the CLI.
- **Fix:** add a `configs/scenario_01.yaml` schema covering `scenario`,
  `cl_method`, `cl_kwargs`, `ppo_kwargs`, `env_kwargs`, `train_steps`,
  `eval_episodes`, `seeds`. Load it in `run_scenario.py` (PyYAML is already a
  dependency, `pyproject.toml:15`); `Mission.cl_kwargs` already exists
  (`missions/base.py:41`) but is unused from the CLI. Allow `--override
  ppo_kwargs.learning_rate=1e-4` for one-off tweaks. Persist the resolved
  config alongside `results.json`.

- [ ] `configs/scenario_01.yaml` + `configs/_base.yaml` defaults
- [ ] `run_scenario.py` learns `--config FILE` and `--override KEY=VAL`
- [ ] dump resolved config to `results/.../config.yaml`

### F2. No multi-seed sweep, no aggregation

- **Where:** `scripts/run_scenario.py:91-107` has `--seed` (singular, int).
  `--compare` (`run_scenario.py:75-88`) explicitly picks `seed_dirs[0]` and
  drops the rest.
- **Why it hurts:** the thesis cannot defend a single-seed result.
  `stage01/scenarios/01_autonomy_sequential_terrains.md` asks for retention
  curves across methods — those need mean ± std over ≥3 seeds.
- **Fix:** `--seeds 0,1,2` flag in `run_scenario.py`, iterate sequentially
  (cheap to parallelize later). Extend `plot_method_comparison`
  (`viz/plots.py:133`) to accept `dict[method, list[results]]` and draw a
  bar + std error bar; add a new
  `plot_retention_curves_with_band(results_list)` that draws mean line + shaded
  ±1σ band. Aggregator should read all `seed_*/results.json` under
  `results/<scenario>/<method>/`.

- [ ] `--seeds` flag (comma list) + loop in `run_scenario.py`
- [ ] `plot_method_comparison` accepts multi-seed input, draws error bars
- [ ] new `plot_retention_curves_band` for per-task mean±std curves

### F3. Seeds are not fully plumbed

- **Where:** `src/rover_cl/envs/nav.py:72` stores `self._np_random_seed = seed`
  but never feeds it to `np.random` / `gym.spaces` (only
  `super().reset(seed=...)` is called and only if the caller passes it).
  `src/rover_cl/cl/base.py:67-69` constructs PPO without `seed=`. `random`
  is used directly in `replay.py:41,145` with no seed. No global
  `torch.manual_seed` / `np.random.seed` anywhere.
- **Why it hurts:** "seed 0" and "seed 1" are partially-different runs but
  not fully reproducible. Two researchers cannot get the same number.
- **Fix:** in `run_scenario.py` / `Runner.run`, derive sub-seeds from the
  mission seed and apply: `np.random.seed(s)`, `random.seed(s)`,
  `torch.manual_seed(s)`, pass `seed=s` to `PPO(...)` in
  `cl/base.py:67`, and call `env.reset(seed=s)` on first reset. Document
  the convention in `CLAUDE.md`.

- [ ] central `set_global_seed(seed)` helper in `rover_cl/__init__.py`
- [ ] `PPO(..., seed=...)` in `cl/base.py:_ensure_model`
- [ ] doc note: env_seed = mission_seed + phase (already done) but is the
      only source of variation per phase

### F4. Visualize-trained-policy gap

- **Where:** `scripts/visualize_rover.py` — entirely scripted demo, no policy
  loading. Checkpoints are saved as `ckpt_phase_*_after_*.zip`
  (`missions/base.py:151`) but there is no consumer.
- **Why it hurts:** this is the single most common debugging request once a
  policy starts misbehaving — "let me watch it drive". Currently impossible
  without writing fresh code.
- **Fix:** add a sibling script `scripts/run_policy.py` (or a `--policy
  PATH --terrain T1_flat` flag on the existing viewer). It should:
  load PPO via `PPO.load`, build `RoverNavEnv` with the requested terrain,
  step through episodes inside `mujoco.viewer.launch_passive`. The
  `_guard_macos_mjpython()` pattern is already there
  (`visualize_rover.py:208`).

- [ ] `scripts/run_policy.py --policy <ckpt> --terrain T1_flat
      --episodes 5 --deterministic`
- [ ] reuse `visualize_rover.py`'s viewer/tracking camera setup

### F5. Logging is effectively silent

- **Where:** PPO is constructed with `verbose: 0` (`cl/base.py:20`).
  TensorBoard is opt-in and silently skipped if the package isn't installed
  (`missions/base.py:100-104`). The only stdout during training is the
  `_log()` calls in `Runner.run()` (one line at phase start, one per
  eval task — `missions/base.py:113-144`).
- **Why it hurts:** A 30 000-step PPO run gives zero feedback for minutes.
  Researchers can't tell if anything is happening or stuck.
- **Fix:** (a) bump `verbose=1` by default in `DEFAULT_PPO_KWARGS`
  (`cl/base.py:20`), or expose `--quiet`. (b) Print a one-line summary at
  the end of `train_on` (mean episode return / mean episode length / SB3's
  `ep_rew_mean`). (c) Add `tensorboard` and `tqdm` to the default deps in
  `pyproject.toml:8-16`, not optional, so the user always gets a progress
  bar.

- [ ] `verbose=1` default
- [ ] per-phase summary print in `Runner.run` after `cl.train_on`
- [ ] move `tensorboard` from optional to required (or document loudly)

### F6. No resume / no continue-from-phase

- **Where:** `Runner.run` (`missions/base.py:107`) always starts from
  phase 0 with a fresh `make_cl(...)`. Checkpoints are written
  (`missions/base.py:151`) but never loaded.
- **Why it hurts:** any crash in phase 3 of 4 = full retrain. Sweep
  experiments cannot reuse the phase-1 PPO.
- **Fix:** `--resume-from <results-dir>` flag (or
  `--start-phase N --init-from <ckpt>`). On resume, `Runner.run` should
  load the latest `ckpt_phase_N_after_*.zip` via `cl.load(...)`, skip
  trained phases, and re-evaluate only what's missing. `BaseCLMethod.load`
  exists (`cl/base.py:96-100`); `ReplayCL.load` exists
  (`cl/replay.py:219-249`).

- [ ] `--resume-from` plumbing in `run_scenario.py` and `Runner.run`
- [ ] `Runner.run(start_phase=N, init_method=cl)` overload

### F7. `results/` directory is barely self-describing

- **Where:** `results/<scenario>/<method>/seed_<N>/` written by
  `run_scenario.py:45`. Contents: `results.json`, two PNGs, N `ckpt_phase_*.zip`,
  optional `tb/`. No `config.yaml`, no `env_versions.txt`, no git SHA, no
  start/end timestamps, no `stdout.log`.
- **Why it hurts:** opening `results/scenario_01_sequential_terrains/replay/seed_0/`
  in three weeks gives you a PNG and a JSON — you have no idea which
  hyperparameters produced it.
- **Fix:** on every run write `meta.json` next to `results.json` containing:
  resolved config (after F1), git SHA (`git rev-parse HEAD`), wall-clock
  start/end, Python/MuJoCo/SB3 versions, hostname. Two small files, one
  small helper in `missions/base.py` or `run_scenario.py`. Also tee training
  stdout into `train.log`.

- [ ] `meta.json` with config + versions + git SHA + timestamps
- [ ] tee Runner stdout to `train.log` in `results_dir`

### F8. `--compare` plot is single-seed and undocumented

- **Where:** `scripts/run_scenario.py:69-88` — only `seed_dirs[0]` is used.
  No legend caveat. Bar chart shows just a single number per method
  (`viz/plots.py:155-190`).
- **Why it hurts:** the plot looks like a result, but it isn't statistically
  meaningful. Easy to mis-read.
- **Fix:** wire up F2 multi-seed aggregation here; until then add a stderr
  warning when only one seed is found ("comparison is single-seed; results
  are anecdotal"). Title the plot with the seed count.

- [ ] aggregate across `seed_*` dirs, draw error bars
- [ ] include N (seeds) in plot title

### F9. Plot defaults — mostly OK, small gaps

- `plot_retention_matrix` (`viz/plots.py:28`) — annotated heatmap, axis
  labels, title, colorbar with label. Good.
- `plot_retention_curves` (`viz/plots.py:88`) — labeled per-task, has
  legend, y-axis label, x-tick rotation. Good. Could use a horizontal
  reference line at first-phase value per task to make forgetting visible at
  a glance.
- `plot_method_comparison` (`viz/plots.py:133`) — bar values annotated,
  axis labels present. No CI / std bars (see F2). No "joint" upper bound
  reference line (see G1).

- [ ] `plot_retention_curves`: per-task horizontal dashed line at the
      first-trained-on value (forgetting baseline reference)

### F10. README documents a CLI that doesn't exist

- **Where:** `README.md:64` says
  `python scripts/run_scenario.py --config configs/<scenario>.yaml` and
  `README.md:67` says `python scripts/visualize.py` (script is actually
  `visualize_rover.py`).
- **Why it hurts:** first impression for a new reader is broken commands.
- **Fix:** rewrite the "Run" block to match the real CLI today, and
  re-update once F1 (configs) and F4 (policy viewer) land.

- [ ] sync `README.md` "Run" section with current CLI
- [ ] rename the file `scripts/visualize.py` → `scripts/visualize_rover.py`
      in README (CLAUDE.md already uses the correct name)

### F11. `configs/` is a placeholder

- `configs/README.md` is a one-screen TODO; no YAML files exist. Fix
  coincides with F1.

### F12. No "joint" / "from-scratch" baselines

- The scenario registry only encodes sequential CL runs. The plan demands
  per-terrain and joint baselines (`docs/plan.md:74-87`,
  `stage01/scenarios/01_*.md` "Joint (upper bound)"). There is no
  `scenario_*_joint` registered, and joint training requires a different
  env (random-terrain reset at each episode) — `RoverNavEnv` is wedded
  to a single terrain at construction.
- **Fix:** add `MultiTerrainNavEnv` (small wrapper around `RoverNavEnv`
  that picks a terrain in `reset()`) and register
  `scenario_01_joint_terrains` and `scenario_01_per_terrain` (K independent
  runs).

- [ ] `envs/multi_nav.py` random-terrain wrapper
- [ ] `scenarios.py`: `scenario_01_joint` and a helper to run per-terrain baselines

### F13. Replay buffer doesn't scale with task count

- **Where:** `cl/replay.py:54` — `buffer_size_per_task=1000` is fixed; total
  memory grows with K tasks. The plan calls for a memory-budget sweep
  (`docs/plan.md:165`, scenario 04). The current API can't express
  "10% of training transitions" or "global cap M".
- **Fix:** support a `total_capacity` mode plus per-task reservoir; or
  parameterize via config (F1).

- [ ] `ReplayCL(total_capacity=...)` alternative to per-task capacity

### F14. CL "registry" has only naive + replay; EWC is missing

See section 4 (it's a thesis blocker, not just ergonomics).

---

## 3. Proposed minimal improvements (priority order)

1. **F1 — YAML configs + `--config` / `--override`.** Unlocks every sweep.
   `~150 LOC, 1 day`. (`scripts/run_scenario.py`, new `rover_cl/config.py`,
   `configs/scenario_01.yaml`, dump resolved config in `Runner`.)

2. **F2 — Multi-seed sweep + aggregation.** Required for any defendable
   plot. `~120 LOC, half a day`. (`--seeds` in `run_scenario.py`,
   `viz/plots.py` accepts list-of-results, new mean±std band helper.)

3. **F14 — EWC implementation.** Thesis blocker (see §4). `~250 LOC, 1–2
   days`. (`cl/ewc.py`: Fisher accumulation after `train_on`, penalty
   injected into PPO's policy update via SB3 callback or by subclassing
   `PPO`. Register in `cl/__init__.py:_REGISTRY`.)

4. **F4 — Policy viewer (`run_policy.py --policy <ckpt>`).** Cheap and
   high-value debugging. `~120 LOC, half a day`. Mostly reuses
   `visualize_rover.py`'s viewer scaffolding.

5. **F3 — Seed plumbing through env / np / torch / PPO.** Reproducibility
   floor. `~40 LOC, 1 hour`. Touches `cl/base.py`, `envs/nav.py`,
   `run_scenario.py`.

6. **F5 — Verbose training + progress bar.** Quality of life. `~10 LOC,
   15 minutes`. Flip `verbose=1`, print phase summary, move `tensorboard`
   out of optional.

7. **F7 — `meta.json` next to `results.json`.** Long-tail reproducibility
   insurance. `~50 LOC, 1 hour`. Self-describing run dirs.

8. **F6 — Resume from checkpoint.** Saves time once runs grow. `~80
   LOC, half a day`. Needs F7 to know what was configured.

9. **F12 — Joint and per-terrain baselines.** Required by the plan.
   `~120 LOC, 1 day`. `MultiTerrainNavEnv` wrapper + two new scenarios.

10. **F13 — Replay total-capacity mode.** Needed for scenario 04
    memory-vs-retention sweep. `~30 LOC, 1 hour`.

11. **F10/F11 — README and configs README fixes.** `~30 LOC, 30 minutes`.
    Do alongside F1.

12. **F8/F9 — Comparison-plot polish.** `~40 LOC, 1 hour`. Do alongside F2.

13. (Nice to have) **threat-classification track stub.** Bigger, see §4.

14. (Nice to have) **LaTeX/markdown figure export pipeline.** Probably
    overkill — matplotlib already writes PNGs at 150 dpi
    (`viz/plots.py:19`). A 20-line helper that re-exports the final
    comparison plot as PDF + `.tex` caption stub would be enough; don't
    build a framework.

---

## 4. What's missing for the thesis

Cross-reference: `docs/plan.md` Phases 0–7,
`stage01/scenarios/{01..05}.md`.

### G1. EWC implementation — **must-have**

- Plan phase 3.1, scenario 01 expects four variants: fine-tune, **EWC**,
  replay, EWC+replay. Today: only `naive` and `replay` exist
  (`cl/__init__.py:12-15`). Without EWC there is no answer to RQ1.
- Effort: ~250 LOC. Fisher diagonal estimated via the policy log-prob
  gradients on a buffer of recent transitions; penalty term added to the
  PPO objective. The replay buffer pattern is reusable.

### G2. EWC+replay hybrid — **must-have**

- Plan phase 3.5 names it. After G1 lands, this is a 30 LOC composition.

### G3. Multi-seed runs with confidence intervals — **must-have**

- See F2 / improvement #2. Without seeds, no figure in the thesis is
  defendable.

### G4. Joint and per-terrain baselines — **must-have**

- See F12 / improvement #9. The retention story needs an upper bound
  (joint) and a per-task gold standard.

### G5. Memory-vs-retention sweep — **must-have** (it's RQ3)

- `stage01/scenarios/04_memory_retention_tradeoff.md`. Needs F1 + F13 +
  a new aggregator that plots retention vs `total_capacity`. The plot
  itself is a new `viz/plots.py` helper.

### G6. Order-sensitivity sweep — **should-have**

- `stage01/scenarios/03_order_sensitivity.md`. Once F1 (configs) lands,
  this is mostly a list of YAML files with permuted task orders.

### G7. Threat-classification track (security) — **must-have for full RQ1**

- Plan phases 4, 5; scenario 02. Today nothing exists for this — no
  classifier, no dataset generator, no supervised CL methods. This is the
  largest single chunk of remaining work.
- Decision the user should make: if time is tight, restrict the thesis
  scope to navigation-only and demote phases 4–6 to "future work". The
  current code base is well-aligned with that descope; the plan still
  promises both.
- Effort: ~800–1500 LOC if attempted (dataset stubs, MLP/1D-CNN,
  supervised-CL `EWC` reusing G1 code, supervised replay buffer, scenario
  2 spec).

### G8. Fusion / shared-encoder track — **nice-to-have**

- Plan phase 6, scenario 05. Only relevant if G7 lands. Otherwise
  drop from scope.

### G9. Cross-method plots with shaded variance bands — **must-have**

- Falls out of F2 (improvement #2). The thesis needs these as standard.

### G10. LaTeX export pipeline — **nice-to-have / probably skip**

- A `make figures` target that copies the final PNGs/PDFs into
  `thesis/figures/` would be sufficient. Don't build a framework.

### G11. Phase-0 deliverable (`docs/analiza_technik_cl.md`) — **must-have, prose-only**

- Plan deliverable §0.3. Not a code change but on the checklist
  (`docs/plan.md:231`). Mention here so it isn't forgotten.

### G12. Stale comment in env code — **trivial**

- `src/rover_cl/envs/nav.py:133` "rover forward is -Y in body frame" is
  stale (per `CLAUDE.md` "Things that look broken but aren't"). Fix while
  in the area; no behavioural impact.

---

## Summary checklist (do this in order)

- [ ] F1 + F11 + F10 — configs + README fix (improvement #1, #11)
- [ ] F3 — seed plumbing (improvement #5)
- [ ] F2 + F8 + F9 — multi-seed + plot polish (#2, #12)
- [ ] F5 + F7 — verbose + `meta.json` (#6, #7)
- [ ] G1 + G2 — EWC + hybrid (#3)
- [ ] F4 — policy viewer (#4)
- [ ] F12 — joint / per-terrain baselines (#9)
- [ ] F6 + F13 — resume + replay capacity (#8, #10)
- [ ] G5 + G6 — memory + order sweeps
- [ ] Decide G7/G8 scope (security + fusion) or descope and document

Estimated cumulative effort to get to a defendable nav-only thesis
result (everything above except G7/G8): roughly **2.5–3.5 weeks of focused
work**.
