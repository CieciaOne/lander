# Research Roadmap — Mars Rover Continual Learning

Planning document. Turns the current prototype (two CL methods, three
flat box-and-plane terrains, two trivial scenarios) into a thesis-grade
setup covering all five scenarios in `stage01/scenarios/`.

Cross-references: `src/rover_cl/envs/terrains.py`, `src/rover_cl/envs/nav.py`,
`src/rover_cl/missions/{base,scenarios}.py`, `src/rover_cl/cl/{naive,replay}.py`,
`scripts/run_scenario.py`, `assets/rover.xml`, `docs/plan.md`,
`stage01/scenarios/0{1..5}_*.md`.

Always activate the venv: `source /Users/jakubciecka/praca-magisterska/.venv/bin/activate`.

---

## 1. Organic / procedural terrain (heightfields)

Today every terrain in `TERRAIN_CATALOG` is a `<geom type="plane">` plus
axis-aligned boxes — fine for T1/T2/T3 but cannot express dunes, slopes,
or craters.

### 1.1 Why heightfields

MuJoCo's first-class mechanism for non-planar ground is **HField**: a 2-D
elevation grid declared in `<asset>` as `<hfield>` and instantiated as
`<geom type="hfield" hfield="..."/>`. HFields collide correctly with rigid
bodies, accept standard `friction` / `material` attributes, and work
transparently with `mj_ray` (so `RoverNavEnv._cast_lidar` needs **no**
change). Data can be inline (XML grid), loaded from a PNG, or written
directly to `MjModel.hfield_data` after compilation.

### 1.2 Concrete MJCF snippet to splice into `compose_scene`

`size` on the hfield asset is `(radius_x, radius_y, elev_z, base_z)`:
radii in metres, `elev_z` max elevation, `base_z` skirt depth below z=0.

```xml
<asset>
  <hfield name="terrain_hf" nrow="128" ncol="128" size="15 15 1.5 0.5"/>
</asset>
<worldbody>
  <geom name="ground" type="hfield" hfield="terrain_hf"
        material="ground_mat" friction="1.0 0.05 0.001"
        contype="1" conaffinity="1"/>
</worldbody>
```

For PNG maps use `file="path/to/map.png"` on the hfield asset. For
Python-generated maps, write the float grid into `MjModel.hfield_data`
after compilation — standard MuJoCo idiom because a 128×128 XML literal
is unwieldy.

### 1.3 Procedural generation in Python

| Method | When to use | Dependency |
|--------|-------------|------------|
| OpenSimplex noise | smooth organic relief (dunes, hills) | `opensimplex` (pure-Python; pick this over `noise`, which is a C ext and breaks on Py3.13) |
| PNG heightmap | curated maps (hand-drawn crater, real DEM tile) | Pillow (already a transitive dep of `gymnasium[mujoco]`) |

Sketch (planning only):

```python
def perlin_heightmap(res=128, scale=8.0, octaves=4, seed=0) -> np.ndarray:
    """Sum N octaves of OpenSimplex noise, normalise to [0,1]."""
def png_heightmap(path: str, res=128) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((res, res))) / 255.0
```

Both return the same shape so the downstream consumer is identical. Add
`opensimplex>=0.4` to `pyproject.toml` dependencies.

### 1.4 Wiring into `TerrainSpec`

Add two optional fields (non-breaking; flat terrains stay untouched):

```python
# (existing fields …)
heightmap: np.ndarray | None = None                    # (nrow, ncol) in [0,1]
heightmap_size: tuple[float, float, float] = (15.0, 15.0, 1.5)  # (rx, ry, max_elev)
```

`compose_scene` branches on `terrain.heightmap is None`:
1. Emit `<hfield>` asset with matching `nrow`/`ncol`/`size`.
2. Emit `<geom type="hfield">` instead of `<geom type="plane">`.
3. After `MjModel.from_xml_string`, copy
   `terrain.heightmap.ravel().astype(np.float32)` into
   `self._model.hfield_data` from `RoverNavEnv.__init__`.

### 1.5 New terrains for the catalog

| ID | Generator | Key params | Purpose |
|----|-----------|-----------|---------|
| `T4_dunes` | OpenSimplex | `scale=12, octaves=4, max_elev=0.6 m` | organic relief; exercises suspension |
| `T5_rocky` | OpenSimplex hi-freq + 18 boulders | `scale=4, max_elev=0.35 m` | hfield + box obstacles coexisting |
| `T6_slope` | analytic ramp `h = grade·y` | `grade=0.12` (≈7°) | constant-grade traction/tipping |
| `T7_crater_rim` | analytic radial rim `h = a·(1 − exp(−r²/σ²))` | rim 0.8 m, σ=4 m | rover skirts rim, must not enter bowl |

### 1.6 Stability and lidar implications

- **Suspension.** Rocker ±25° (`range="-0.44 0.44"`), bogie ±20°
  (`range="-0.35 0.35"`); per-wheel bumps up to ~0.25 m are handled
  (chassis lifts by half thanks to the differential equality). T4/T5
  stay under that; T6's 7° is gentle; T7 is deliberately steep — expect
  occasional tip-overs, which is the desired stress signal.
- **Friction.** Wheel friction (`1.8 0.1 0.01` in `default class="wheel"`)
  is tuned for the flat plane. For T6/T7 bump the *terrain* `friction`
  via the existing `TerrainSpec.friction` field (don't edit `rover.xml`).
- **Lidar.** `mj_ray(..., flg_static=1, ...)` casts against all static
  geoms including `mjGEOM_HFIELD`. **No code change** beyond adding the
  geom; add a test that the central ray returns < max-range on a ridge.

---

## 2. Mission scenarios — one Mission per spec

Each of the five concept docs in `stage01/scenarios/` gets one concrete
`Mission` factory in `missions/scenarios.py`. Default cohort: seeds 0,1,2
(n=3); promote to 5 for trend confirmation; 10 only for final thesis tables.

### 2.1 Scenario 1 — `scenario_01_sequential_terrains_full`

Maps to `01_autonomy_sequential_terrains.md`. Replaces the 2-task toy
with the full T1→T2→T3→T4 sequence; T4 is the hfield terrain so the
scenario tests CL across terrain *families*, not just obstacle counts.

- Tasks: `T1_flat → T2_corridor → T3_obstacle_field → T4_dunes`.
- Steps/task: 80k (current 30k is too short for the dune terrain). Eval: 20 episodes.
- Methods: `naive`, `replay`, `ewc`, `ewc+replay`, `joint` (random
  terrain at reset, upper-bound baseline).
- Output: 4×4 retention matrix, retention-on-T1 curve, per-method bar.

### 2.2 Scenario 3 — `scenario_03_order_sensitivity`

Maps to `03_order_sensitivity.md`. Two reversed orders, identical
otherwise — the analysis is about *order*, not *method*.

- Variants: `easy_to_hard` (`T1_flat → T6_slope → T2_corridor → T3_obstacle_field`)
  and `hard_to_easy` (reverse).
- Steps: 60k/task. Eval: 15. Methods: best CL method from §2.1
  (probably `ewc+replay`) plus `naive` as a sanity baseline.
- Output: two retention matrices side-by-side, ordering-vs-mean-retention
  bar; optional random-permutation run characterises variance.

### 2.3 Scenario 4 — `scenario_04_replay_sweep`

Maps to `04_memory_retention_tradeoff.md`. Sweep the buffer to draw the
trade-off curve.

- Sequence: `T1_flat → T2_corridor → T3_obstacle_field` (3 tasks is
  enough to see the shape).
- Steps: 50k/task. Eval: 15. Method: `replay` with
  `buffer_size_per_task ∈ {100, 500, 1000, 2500, 5000}`; anchor with
  `naive` (buffer=0) and `joint` (upper bound).
- Output: `mean_retention vs buffer_size` curve with seed error bars;
  secondary `wall-clock vs buffer` plot; annotate the knee for the
  Phase 3.4 recommendation in `docs/plan.md`.

### 2.4 Scenario 5 — `scenario_05_fusion_multi_task` *(stub, blocks on §4.2)*

Maps to `05_fusion_multi_task.md`. Until the threat track exists the
factory raises `NotImplementedError`.

- Mixed sequence: `[T1, C1, T2, C2, T3, C3, T4, C4]`. Nav tasks call
  `train_on(nav_env, ...)`; classifier tasks call `train_on(loader, ...)`
  on the same shared encoder.
- Budget: 40k PPO steps per nav task, 10 epochs per class.
- Methods: `naive` and `ewc+replay` shared across both task types.
- Output: combined retention vector (4 nav + 4 cls); param-count and
  inference-time vs the "two separate models" baseline.

### 2.5 Scenario 2 — `scenario_02_security_sequential_classes` *(stub, in §4.2)*

Supervised CL; lives in a parallel `rover_cl.security` package. Class
sequence C1..C4; methods `naive_cls`, `replay_cls`, `ewc_cls`,
`ewc+replay_cls`, `joint_cls`; output confusion matrix + macro-F1 per class.

### 2.6 HField bonus mission — `scenario_06_dune_to_crater`

End-to-end test of the hfield work on terrain only hfields can express.

- Sequence: `T4_dunes → T7_crater_rim → T6_slope`. Steps: 80k/task. Eval: 20.
- Methods: `naive` vs `ewc+replay`. Hypothesis: crater-rim "edge
  avoidance" forgets fastest because slope and dune tasks never need it.
- Output: retention matrix + qualitative trajectory plots showing
  whether the rover still respects the rim after the slope task.

### 2.7 Summary table

| Scenario factory | Tasks | Steps/task | Methods | Seeds | New code? |
|---|---|---|---|---|---|
| `scenario_01_sequential_terrains_full` | 4 (incl. T4 hfield) | 80k | 5 | 3 | extends existing |
| `scenario_03_order_sensitivity(direction)` | 4 | 60k | 2 | 3 | new factory |
| `scenario_04_replay_sweep(buffer_size)` | 3 | 50k | replay only | 3 | new factory |
| `scenario_05_fusion_multi_task` | 8 mixed | 40k+10ep | 2 | 3 | new, blocked |
| `scenario_06_dune_to_crater` | 3 hfields | 80k | 2 | 3 | new factory |

---

## 3. Concrete code interfaces (sketches only)

### 3.1 `envs/terrains.py::add_hfield`

```python
def add_hfield(spec: TerrainSpec, heightmap: np.ndarray,
               size: tuple[float, float, float] = (15.0, 15.0, 1.5)
               ) -> TerrainSpec:
    """Attach a procedurally generated heightmap to an existing TerrainSpec.

    Returns a new TerrainSpec with `heightmap` set; the original is unmodified
    (dataclass replace semantics). `heightmap` must be a 2-D float32 array
    with values in [0, 1]; `size` is (radius_x, radius_y, max_elev_z) in metres.
    The ground plane in compose_scene is auto-replaced by an <hfield> geom.
    """
```

### 3.2 `missions/scenarios.py::scenario_03_order_sensitivity`

```python
def scenario_03_order_sensitivity(
    direction: Literal["easy_to_hard", "hard_to_easy"],
    cl_method: str = "replay",
    train_timesteps: int = 60_000,
    eval_episodes: int = 15,
    max_steps: int = 600,
    seed: int = 0,
) -> Mission:
    """Scenario 3: probes whether task ordering changes final retention.

    Builds the same 4-terrain sequence in two opposite orderings; the
    caller runs both variants with the same seeds and compares retention
    matrices. The chosen CL method should be the best one identified in
    scenario_01; the comparison is about ordering, not method.
    """
```

### 3.3 `missions/scenarios.py::scenario_04_replay_sweep`

```python
def scenario_04_replay_sweep(
    buffer_size: int,
    train_timesteps: int = 50_000,
    eval_episodes: int = 15,
    max_steps: int = 600,
    seed: int = 0,
) -> Mission:
    """Scenario 4: replay buffer-size sweep across T1→T2→T3.

    Fixes cl_method='replay' and exposes buffer_size_per_task as the swept
    knob via cl_kwargs. The driver script (scripts/run_sweep.py) loops
    over a list of sizes, runs each as a separate Mission with its own
    results dir, and aggregates a memory-vs-retention curve.
    """
```

### 3.4 `scripts/run_sweep.py`

```python
def main() -> None:
    """Run a parameter grid (cl_method × seed × scenario-specific knob).

    CLI:
      python scripts/run_sweep.py --scenario scenario_04_replay_sweep \\
        --buffer-sizes 100,500,1000,2500,5000 \\
        --seeds 0,1,2 \\
        --train-steps 50000

    Writes per-cell results under results/<scenario>/<knob_value>/seed_<N>/,
    then aggregates a CSV (one row per (knob, seed, task_id, phase)) plus
    a summary plot. Re-running with the same args skips finished cells.
    """
```

### 3.5 `Runner.multi_seed`

```python
class Runner:
    def __init__(self, mission: Mission, results_dir: Path | None = None,
                 verbose: bool = True, multi_seed: bool | int = False):
        """If `multi_seed` is a positive int N, run() loops over N seeds
        (mission.seed, mission.seed+1, ..., mission.seed+N-1), writes
        each seed's results into a `seed_<i>/` subdir, and returns a list
        of MissionResult. Aggregation (mean/std across seeds for retention
        matrices and curves) is delegated to rover_cl.eval.
        Keeps the single-seed path unchanged when multi_seed is False/0."""
```

---

## 4. Open issues / not-yet-built

### 4.1 EWC not implemented

Only `NaiveCL` and `ReplayCL` exist; Phase 3 of `docs/plan.md` requires
EWC and an EWC+replay hybrid. Plan (~150 LOC):

- New `src/rover_cl/cl/ewc.py` with `class EWC(BaseCLMethod)`.
- After each `train_on(task_k)`: collect 5–10 rollouts, compute the
  Fisher diagonal as mean squared gradient of log-prob w.r.t. policy
  params. Store `(theta_star_k, fisher_k)`.
- Next task's PPO loss += `λ · Σ_i fisher_i · (θ_i − θ*_i)²`. Use
  *online EWC* (accumulate Fishers into a single vector) to keep memory
  O(|θ|).
- Implement by subclassing SB3 `PPO` and overriding `_compute_loss`
  (cleaner than monkey-patching `policy.forward`).
- Hyperparams: λ ∈ {1e2, 1e3, 1e4} (Phase 3.4 sweep);
  `fisher_n_rollouts ∈ {5, 10}`; default λ=1e3.
- EWC+replay = subclass that also keeps `ReplayCL`'s buffers and does
  BC rehearsal after PPO+EWC training.
- Tests: Fisher finite/positive; EWC loss = 0 on the seed task;
  retention(ewc) > retention(naive) on T1→T2.

### 4.2 Threat-classification track not built

Scenarios 2 and 5 need a supervised classifier that doesn't exist.

- New `src/rover_cl/security/` package:
  - `data.py` — synthetic 32-step telemetry windows (IMU + lidar +
    cmd-vel). Four classes: normal, sensor-spoof, drift-anomaly,
    command-injection. Returns `torch.utils.data.Dataset`.
  - `model.py` — small 1-D CNN (3 conv blocks → GAP → MLP), ~10k params.
  - `cl/` mirrors `rover_cl/cl/`: `NaiveCls`, `ReplayCls` (buffer of
    `(x, y)`), `EWCCls` (Fisher of correct-class log-prob).
- Generalise `Task.env_factory` to `Task.factory: Callable` and let the
  CL method dispatch on input type, or fork a `ClsTask` dataclass
  (~10 LOC refactor either way).
- `Runner.run` swaps `evaluate_policy` for `evaluate_classifier` on
  supervised tasks.

### 4.3 No hyperparameter sweep tooling

`scripts/run_scenario.py` runs *one* `(scenario, method, seed)` triple;
sweeps are done by shell loops with no caching/aggregation/parallelism.
Plan: §3.4 `scripts/run_sweep.py` + `configs/sweeps/*.yaml`:

```yaml
scenario: scenario_04_replay_sweep
grid:
  buffer_size: [100, 500, 1000, 2500, 5000]
  seed: [0, 1, 2]
train_steps: 50000
parallel: 4
```

`yaml.safe_load` → `itertools.product` the grid → `ProcessPoolExecutor`
(MuJoCo's Python wrapper is the only GIL-bound part, so processes scale).
Skip cells whose `results.json` already exists.

### 4.4 Heightmap terrain itself

Tracked in §1; prerequisite for §2.1's T4 entry and §2.6.

### 4.5 Other gaps spotted while reading the code

- **Stale comment** in `src/rover_cl/envs/nav.py` line 133 ("rover
  forward is -Y in body frame"); behaviour is correct (`step` negates
  ctrl), comment misleading. Per `CLAUDE.md`. Fix on next touch.
- **Lidar duplication.** `RoverNavEnv._cast_lidar` uses 8 rays via
  `mj_ray`, but the MJCF also declares 5 `<rangefinder>` sensors. The
  two are independent and nothing reads the MJCF sensors today.
  Consolidate before hfield work — hfield ray casts are more expensive
  than plane casts, so saving 8 casts/step matters.
- **`Runner` lacks a deterministic RNG.** Once EWC lands, Fisher rollouts
  need a `np.random.Generator` threaded through `train_on` rather than
  relying on env seeds.
- **`compose_scene` does string templating.** Fine for boxes; fragile
  once we add hfields, per-hfield materials, and per-terrain friction
  overrides. Consider an `xml.etree` / `dm_control.mjcf` builder
  in a follow-up. Not blocking.
- **Empty `configs/`** (only `README.md`). First YAML to land should be
  `configs/best_cl_autonomy.yaml` per `docs/plan.md` Phase 3.4 deliverable.
- **No across-seed aggregation.** `compute_retention_matrix` is per-run;
  add `aggregate_retention(results_list) -> (mean, std)` once multi-seed
  exists.

---

## 5. Suggested execution order

1. EWC (§4.1) — unblocks the actual research comparison in Scenario 1.
2. HField terrains (§1) — adds T4..T7 and enables Scenario 6.
3. `Runner.multi_seed` + `scripts/run_sweep.py` (§3.4, §3.5, §4.3) —
   removes the bash-loop sprawl.
4. Scenarios 1-full, 3, 4, 6 (§2.1, §2.2, §2.3, §2.6) — these are the
   four nav-only scenarios; produce the bulk of the thesis figures.
5. Threat classification track (§4.2) — unlocks Scenario 2 and Scenario 5.
6. Scenarios 2 and 5 — final figures, fusion comparison.

Each item is independently testable; nothing in §2-§5 depends on
anything in §1 *except* the hfield-using scenarios (1-full and 6),
which can be deferred if hfields slip.
