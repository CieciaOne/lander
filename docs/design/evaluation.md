# Evaluation and visualization

> Code: `src/rover_cl/eval/metrics.py`, `src/rover_cl/viz/plots.py`,
> `src/rover_cl/missions/base.py::Runner.run`.

## Eval flow

After PPO training on each task phase, `Runner.run` iterates over every
task seen so far and calls `evaluate_with_trajectories` on a freshly-built
eval env. It then writes:

- `results.json` (accumulated retention matrix across phases).
- `ckpt_phase_<k>_after_<task>.zip` (PPO checkpoint).
- `report_phase_<k>_after_<train_task>_on_<eval_task>.png` (top-down
  trajectory report — see below).

## `EpisodeTrajectory` (per-episode record)

Defined in `eval/metrics.py`. Built incrementally by
`rollout_with_trajectory`:

| Field | Source |
|---|---|
| `positions: np.ndarray (T, 2)` | `info["pos_xy"]` per step. |
| `yaws: np.ndarray (T,)` | `info["yaw"]` per step. |
| `contact_positions: np.ndarray (n_contact, 2)` | `info["pos_xy"]` filtered to steps where `info["collision"]`. |
| `waypoint_hit_steps: list[int]` | Steps at which `info["waypoint_index"]` advanced. |
| `success / tipped / truncated: bool` | From episode termination cause. |
| `steps: int` | Number of recorded positions (≤ episode length). |
| `final_distance_to_goal: float` | Last `info["distance_to_goal"]`. |
| `cumulative_reward: float` | Sum of per-step rewards. |
| `outcome` (property) | `"success" | "tipped" | "timeout"`. |

## Eval seed variation

`evaluate_with_trajectories(seed_base=X)` calls
`rollout_with_trajectory(env, ..., seed=X+i)` for `i ∈ [0, n_episodes)`.
Each call invokes `env.reset(seed=X+i)`, which reseeds the env's
`np_random`. The env's start-pose jitter (`±0.5 m, ±0.2 rad`) is sampled
from `self.np_random`, so different `seed_base + i` values produce
**different starting states** and therefore different trajectories under
the same deterministic policy.

Runner uses `seed_base = mission.seed + 10000 * (phase + 1) + 100 * j`
where `j` is the eval-task index. The +10 000 offset keeps eval seeds
disjoint from training seeds (training uses `seed + phase * 1000 + worker_idx`).

## Top-down run report (`plot_run_report`)

The most prominent artifact of each eval phase. Single PNG, two-panel
layout:

**Left panel — the arena**:

- Arena boundary as a dashed rectangle (or implicit edge when there's a
  heightmap).
- **Heightmap underlay** when terrain has one: `imshow` of
  `heightmap * heightmap_extent[2]` with `gist_earth` colormap, alpha
  0.42 so trajectories still pop on top. A second colorbar reports
  height in metres.
- Obstacles as gray boxes with a subtle drop-shadow underneath.
- Start (slate blue dot + halo), waypoints (translucent cyan disks with
  `wpN` labels), goal (green star + radius disk).
- **All eval trajectories** drawn as `LineCollection` segments colored
  by **rover speed** with the `magma` colormap (capped at 1.5 m/s so
  one outlier doesn't wash out the others). A faint white underlay
  gives each path a subtle glow over the heightmap.
- Endpoint dot per trajectory colored by outcome (success = sage green,
  tipped = wine, timeout = burnt sienna).
- **Contact positions** as red X markers.

**Right panel — info sidebar**:

- Big headline `success_rate %` (color-coded — green ≥ 70%, ochre 30–70%,
  wine < 30%).
- Three outcome tiles: success / tipped / timeout counts.
- Averages table: steps (success and overall), final `d_goal`, mean
  return, mean contact steps, any-hit count.
- `any-hit` row is color-flagged red when the rover is hitting things.

The sidebar is rendered inside a `FancyBboxPatch` panel. Because that
patch confuses `tight_layout`, the figure uses explicit
`fig.subplots_adjust(...)` instead.

## Thesis plot style

Applied at module import (`_apply_thesis_style()` in `viz/plots.py`).
Settings:

- **Typography**: serif body (Charter → Source Serif Pro → DejaVu Serif
  fallback chain), 10 pt base. Sans-serif reserved for fine
  annotations and stat values.
- **Spines**: top + right dropped. Subtle warm-grey grid at 0.65 alpha,
  axisbelow=True.
- **Lines**: 1.8 px default, 5.5 marker. Patches: 0.6 px edge.
- **Colors**: curated 8-colour palette in `_THESIS_COLORS`. Distinct in
  both colour and luminance (works in greyscale print), colourblind-safe
  except for a slight blue-green overlap.
- **Saves**: 200 DPI, white facecolor, 0.10 pad.

The palette and helper colours `_FG`, `_FG_MUTED`, `_GRID`, `_PANEL_BG`
are module-level constants — change in one place to retheme everything.

## Other plots

| Function | What it shows |
|---|---|
| `plot_retention_matrix` | Heatmap of success_rate across (training phase × eval task). NaN cells (task not yet seen) render in warm beige. Cell values annotated; text color flips for contrast. |
| `plot_retention_curves` | One line per eval task across training phases. Each line gets a faint halo stroke + an annotated final-value label at its right end. |
| `plot_method_comparison_with_variance` | Bar chart per CL method (mean ± std across seeds). Value labels above bars; seed count `n=N` inside each bar in white. |
| `plot_method_comparison` | Same as variance version but without error bars; legacy single-seed path. |

## Where reports land

```
results/
└── <scenario>/
    └── <method>/
        └── seed_<N>/
            ├── ckpt_phase_0_after_T1_flat.zip
            ├── ckpt_phase_1_after_T2_corridor.zip
            ├── report_phase_0_after_T1_flat_on_T1_flat.png
            ├── report_phase_1_after_T2_corridor_on_T1_flat.png
            ├── report_phase_1_after_T2_corridor_on_T2_corridor.png
            ├── results.json
            ├── matrix.png            (after run_mission)
            └── curves.png            (after run_mission)
        └── comparison.png            (after `--compare`)
```

`results/` is gitignored.
