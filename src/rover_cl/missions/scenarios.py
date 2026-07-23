"""Predefined missions (research scenarios).

Adding a new scenario = adding one function that returns a `Mission` and
registering it in `SCENARIO_REGISTRY`. Scenario factories accept (at minimum)
`cl_method`, `train_timesteps`, `eval_episodes`, `max_steps`, `seed`.

## Training budget notes

The `RoverNavEnv` task is **harder than typical gym continuous-control tasks**:
30-D observation (6 pose+vel + 8 nearest-obstacle slots × 3 fields), 2-D action
(Ackermann throttle + steer), sparse-ish reward (+50 goal bonus dominates), and
~25-second episodes. Realistic PPO compute requirements per task:

- **30k timesteps** — practically nothing learned (random-walk-ish policy).
- **100k timesteps** (current default) — policy starts driving forward
  consistently; success rate on T1_flat reaches ~10-30%. Good for development.
- **300k timesteps** — policy learns to steer around obstacles; T1 success
  ~50-70%. Good for one-off plots.
- **500k–1M timesteps** — research-quality numbers; success >80% on solved
  terrains. Use this for thesis-headline runs.

`max_steps=1000` gives a perfect policy ~25 s of sim time per episode (rover
top speed ≈ 0.6 m/s, goal at 10 m → ~17 s ideal). Don't drop below 800.
"""

from __future__ import annotations

from typing import Literal

from .base import Mission, Task
from rover_cl.envs import RoverNavEnv


def _env_factory(terrain_name: str, max_steps: int = 500,
                 env_kwargs: dict | None = None):
    ekw = dict(env_kwargs or {})
    def _make(seed: int):
        return RoverNavEnv(terrain=terrain_name, max_steps=max_steps, seed=seed, **ekw)
    # Stash the build inputs so a post-processor (e.g. apply_perception) can
    # rebuild the factory with merged env_kwargs without the scenario knowing.
    _make._terrain_name = terrain_name
    _make._max_steps = max_steps
    _make._env_kwargs = ekw
    return _make


# PERCEPTION MODE axis (orthogonal to the CL method) — env_kwargs overrides that
# turn any obstacle scenario into a given observation-realism setting, so the
# thesis can compare (CL method × perception mode). See RoverNavEnv docs.
PERCEPTION_ENV_KWARGS: dict[str, dict] = {
    # full ground truth: obstacle AABBs AND a geo_heading route hint planned on
    # the true map — the genuine upper bound. (Paired with slam's discovered
    # route hint so the comparison is apples-to-apples on ROUTE quality, not on
    # "who has a route hint at all".)
    "privileged": dict(obstacle_obs_mode="privileged", use_lidar=True,
                       geo_heading_obs=True, geo_heading_source="truth"),
    # honest mapless: no ground-truth obstacle info, lidar only.
    "reactive": dict(obstacle_obs_mode="none", use_lidar=True,
                     geo_heading_obs=False),
    # discover-and-plan: obstacles sensed into an occupancy map online, and the
    # geo_heading route hint is planned on that DISCOVERED map. The occupancy
    # field is built independently of the reward mode, so we DON'T force the
    # reward here — every perception mode keeps the scenario's own reward, which
    # keeps the comparison clean (only the observation differs).
    "slam": dict(obstacle_obs_mode="none", use_lidar=True, geo_heading_obs=True,
                 geo_heading_source="slam"),
}


def apply_perception(mission: "Mission", perception: str | None) -> "Mission":
    """Rebuild every task's env factory with the perception override merged on
    top of the scenario's own env_kwargs. No-op when perception is None."""
    if perception is None:
        return mission
    if perception not in PERCEPTION_ENV_KWARGS:
        raise KeyError(f"Unknown perception {perception!r}. "
                       f"Known: {sorted(PERCEPTION_ENV_KWARGS)}")
    override = PERCEPTION_ENV_KWARGS[perception]
    for task in mission.tasks:
        fn = task.env_factory
        if not hasattr(fn, "_env_kwargs"):
            # factory not built by _env_factory (custom) — can't safely rebuild
            continue
        base_ekw = dict(fn._env_kwargs)
        base_ekw.update(override)
        task.env_factory = _env_factory(fn._terrain_name, fn._max_steps, base_ekw)
    return mission


def _make_task(
    terrain_name: str,
    train_timesteps: int,
    eval_episodes: int,
    max_steps: int,
    *,
    ent_coef: float | None = None,
    min_success_to_advance: float | None = None,
    max_budget_multiplier: float = 2.0,
    gate_check_interval: int = 50_000,
    gate_eval_episodes: int = 16,
    interim_eval_every: int = 0,
    interim_eval_episodes: int = 5,
    env_kwargs: dict | None = None,
) -> Task:
    return Task(
        terrain_name,
        _env_factory(terrain_name, max_steps, env_kwargs),
        train_timesteps=train_timesteps,
        eval_episodes=eval_episodes,
        eval_max_steps=max_steps,
        ent_coef=ent_coef,
        min_success_to_advance=min_success_to_advance,
        max_budget_multiplier=max_budget_multiplier,
        gate_check_interval=gate_check_interval,
        gate_eval_episodes=gate_eval_episodes,
        interim_eval_every=interim_eval_every,
        interim_eval_episodes=interim_eval_episodes,
    )


def scenario_01_sequential_terrains(
    cl_method: str = "naive",
    train_timesteps: int = 100_000,
    eval_episodes: int = 10,
    max_steps: int = 1000,
    seed: int = 0,
) -> Mission:
    """Scenario 1: T1 → T2 sequential. Probes forgetting on terrain shift."""
    return Mission(
        name=f"scenario_01_{cl_method}",
        tasks=[
            Task("T1_flat", _env_factory("T1_flat", max_steps),
                 train_timesteps=train_timesteps, eval_episodes=eval_episodes,
                 eval_max_steps=max_steps),
            Task("T2_corridor", _env_factory("T2_corridor", max_steps),
                 train_timesteps=train_timesteps, eval_episodes=eval_episodes,
                 eval_max_steps=max_steps),
        ],
        cl_method=cl_method,
        seed=seed,
    )


def scenario_02_three_terrains(
    cl_method: str = "naive",
    train_timesteps: int = 100_000,
    eval_episodes: int = 10,
    max_steps: int = 1000,
    seed: int = 0,
) -> Mission:
    """Scenario 2: T1 → T2 → T3 sequential. Extends scenario 1 with denser obstacles."""
    return Mission(
        name=f"scenario_02_{cl_method}",
        tasks=[
            Task(t, _env_factory(t, max_steps),
                 train_timesteps=train_timesteps, eval_episodes=eval_episodes,
                 eval_max_steps=max_steps)
            for t in ["T1_flat", "T2_corridor", "T3_obstacle_field"]
        ],
        cl_method=cl_method,
        seed=seed,
    )


def scenario_03_order_sensitivity(
    direction: Literal["easy_to_hard", "hard_to_easy"] = "easy_to_hard",
    cl_method: str = "ewc",
    train_timesteps: int = 100_000,
    eval_episodes: int = 10,
    max_steps: int = 1000,
    seed: int = 0,
) -> Mission:
    """Scenario 3: order-sensitivity. Same three terrains, two orderings.

    Run BOTH directions ('easy_to_hard' = T1→T2→T3, 'hard_to_easy' = T3→T2→T1)
    with the same CL method and seed, then compare final retention. If the
    rocker-bogie nav benefits from curriculum order, EWC's `avg_retention` will
    differ noticeably between the two. Maps onto `stage01/scenarios/03_*`.
    """
    if direction == "easy_to_hard":
        order = ["T1_flat", "T2_corridor", "T3_obstacle_field"]
    elif direction == "hard_to_easy":
        order = ["T3_obstacle_field", "T2_corridor", "T1_flat"]
    else:
        raise ValueError(
            f"direction must be 'easy_to_hard' or 'hard_to_easy'; got {direction!r}"
        )
    return Mission(
        name=f"scenario_03_{direction}_{cl_method}",
        tasks=[_make_task(t, train_timesteps, eval_episodes, max_steps) for t in order],
        cl_method=cl_method,
        seed=seed,
    )


def scenario_04_replay_sweep(
    buffer_size: int = 1000,
    cl_method: str = "replay",
    train_timesteps: int = 100_000,
    eval_episodes: int = 10,
    max_steps: int = 1000,
    seed: int = 0,
) -> Mission:
    """Scenario 4: memory–retention tradeoff. Run T1→T2→T3 with a chosen
    replay buffer size; sweep `buffer_size` ∈ {100, 1000, 5000} across multiple
    invocations and plot retention vs buffer. Maps onto `stage01/scenarios/04_*`.
    """
    if cl_method != "replay":
        # The sweep is meaningful only for replay-style methods; warn but allow.
        # (We don't hard-fail because someone may want a control run with naive.)
        pass
    return Mission(
        name=f"scenario_04_replay{buffer_size}_{cl_method}",
        tasks=[
            _make_task(t, train_timesteps, eval_episodes, max_steps)
            for t in ["T1_flat", "T2_corridor", "T3_obstacle_field"]
        ],
        cl_method=cl_method,
        cl_kwargs={"buffer_size_per_task": int(buffer_size)} if cl_method == "replay" else {},
        seed=seed,
    )


def scenario_05_full_terrain_curriculum(
    cl_method: str = "ewc",
    train_timesteps: int = 100_000,
    eval_episodes: int = 10,
    max_steps: int = 1000,
    seed: int = 0,
) -> Mission:
    """Scenario 5: full curriculum across the catalog — flat → corridor → obstacles → dunes.

    Includes the HField-based `T4_dunes` so the policy must adapt to organic
    terrain after only ever seeing flat/box scenes. This is the "headline" run
    for the thesis's continual-learning track.
    """
    return Mission(
        name=f"scenario_05_full_{cl_method}",
        tasks=[
            _make_task(t, train_timesteps, eval_episodes, max_steps)
            for t in ["T1_flat", "T2_corridor", "T3_obstacle_field", "T4_dunes"]
        ],
        cl_method=cl_method,
        seed=seed,
    )


def scenario_07_blocked_arc(
    cl_method: str = "naive",
    train_timesteps: int = 200_000,
    eval_episodes: int = 10,
    max_steps: int = 1500,
    seed: int = 0,
) -> Mission:
    """Scenario 7: single-task arc-around-blocker on T1_blocked_arc.

    One terrain only — the rover must arc-left around a center blocker to
    reach the goal. Uses the env's waypoint mechanic (intermediate point at
    (-2.2, 6.0)) so the natural progress-reward gradient biases CCW motion.
    Useful for: validating that the policy can learn obstacle avoidance at
    all before moving on to multi-task CL settings.
    """
    return Mission(
        name=f"scenario_07_blocked_arc_{cl_method}",
        tasks=[
            _make_task("T1_blocked_arc", train_timesteps, eval_episodes, max_steps),
        ],
        cl_method=cl_method,
        seed=seed,
    )


def scenario_08_blocked_arc_hills(
    cl_method: str = "naive",
    train_timesteps: int = 300_000,
    eval_episodes: int = 10,
    max_steps: int = 1500,
    seed: int = 0,
) -> Mission:
    """Scenario 8: arc-around-blocker on a gently-undulating heightmap.

    Same waypoint geometry as scenario_07 (T1_blocked_arc) but the ground is
    a low-amplitude perlin heightmap (≤ 0.15 m bumps). Lets you measure how
    much the rocker-bogie suspension matters when navigation logic is held
    fixed — a flat-terrain policy should transfer, just slower.
    """
    return Mission(
        name=f"scenario_08_blocked_arc_hills_{cl_method}",
        tasks=[
            _make_task("T1_blocked_arc_hills", train_timesteps, eval_episodes, max_steps),
        ],
        cl_method=cl_method,
        seed=seed,
    )


def scenario_09_curriculum_arc(
    cl_method: str = "naive",
    train_timesteps: int = 200_000,
    eval_episodes: int = 10,
    max_steps: int = 1500,
    seed: int = 0,
) -> Mission:
    """Scenario 9: 3-task curriculum on the arc-around-blocker family.

    T1_flat → T1_blocked_arc → T1_blocked_arc_hills. Each phase warm-starts
    PPO from the previous phase's weights, so the policy first masters basic
    forward-drive + steering on an empty arena, then learns to arc-left around
    a center blocker, then adds suspension dynamics over a gentle heightmap.
    This is the "right way" to bring up a hard task: split it into pieces and
    use the natural curriculum order, instead of asking PPO to discover all
    three skills simultaneously from scratch.

    Default budget = 200k steps × 3 tasks = 600k total. Combine with
    `cl_method='ewc'` or `'replay'` to also preserve earlier-task performance.
    """
    return Mission(
        name=f"scenario_09_curriculum_arc_{cl_method}",
        tasks=[
            _make_task("T1_flat", train_timesteps, eval_episodes, max_steps),
            _make_task("T1_blocked_arc", train_timesteps, eval_episodes, max_steps),
            _make_task("T1_blocked_arc_hills", train_timesteps, eval_episodes, max_steps),
        ],
        cl_method=cl_method,
        seed=seed,
    )


def scenario_10_robust_curriculum(
    cl_method: str = "ewc",
    train_timesteps: int = 200_000,
    eval_episodes: int = 10,
    max_steps: int = 1500,
    seed: int = 0,
    ewc_lam: float = 400.0,
) -> Mission:
    """Scenario 10: 9-phase domain-randomized curriculum.

    Builds a single robust policy by walking from "drive anywhere on a flat
    plane" to "everything at once on a heightmap with obstacles". Each phase
    uses a `RT_` (randomized terrain) factory that re-rolls obstacle
    positions, start, goal, and (where applicable) the heightmap shape on
    every `reset()` — so the policy sees thousands of distinct configurations
    per phase instead of one fixed layout.

    Phase progression (revised after first-pass analysis):
        0. RT_drive_random       — flat, no obstacles, random start + goal
        1. RT_with_waypoint      — + 1 random waypoint (tight lateral jitter)
        2. RT_with_two_waypoints — + 2 random waypoints
        3. RT_one_obstacle       — 1 random obstacle on path
        4. RT_obstacle_field     — 3-6 random obstacles
        5. RT_dense_obstacles    — 8-12 random obstacles
        6. RT_gentle_hills       — random heightmap up to 0.2 m, no obstacles
        7. RT_dunes              — random heightmap up to 0.6 m, no obstacles
        8. RT_mixed              — capstone: obstacles + waypoints + heightmap

    Why this ordering: drive → waypoint variants first while obs/action are
    simplest (no obstacles to dodge); then obstacle phases stack contiguously
    (avoidance skills reinforce each other); then hfield phases; final
    `RT_mixed` reinforces everything together so the deployed checkpoint
    actually retains every skill — addresses the "phase 7 forgot one_obstacle"
    failure mode from the first scenario_10 run.

    `cl_method` defaults to `'ewc'` with `lam=400` (up from 100). Higher lam
    means weight updates that move important parameters far from the
    snapshot are penalized harder — keeps RT_drive_random at 100% even
    through the obstacle-heavy phases.

    Default budget: 200k × 9 phases = 1.8 M timesteps. With `--n-envs 6` on
    an M3 Air, ~2-3 hours wall-clock. The final checkpoint
    (`ckpt_phase_8_after_RT_mixed.zip`) is the robust deployable model.
    """
    # Per-phase plan: (terrain_id, budget_multiplier, ent_coef_override).
    #
    # Budget multipliers scale the CLI `train_timesteps`. Easy phases stay
    # at base; obstacle phases get 2-3× because prior runs showed self-
    # success well below convergence with a uniform budget (phase 5 was 0%
    # on its own training task). Sum = 14× — `--train-steps 400000` ⇒ ~5.6
    # M total env steps.
    #
    # `ent_coef` overrides PPO's entropy coefficient for that phase only.
    # Obstacle phases 3-5 use 0.05 (5× the default) to keep the policy
    # exploratory enough to climb out of the "freeze near start" local
    # minimum that the reports flagged. `None` means "use the PPO default
    # (0.01)".
    # Navigation-skill phases (1-6) come BEFORE the obstacle phases so the
    # rover enters obstacle training already knowing how to chase a
    # waypoint, lay into an arc, do a 90° turn, do a U-turn, and slalom.
    # Phases 7-9: obstacle ramp (one → field → dense). Phases 10-11: terrain
    # (hills → dunes). Phase 12: mixed-everything capstone.
    phase_plan: list[tuple[str, float, float | None]] = [
        ("RT_drive_random",       1.0, None),
        ("RT_with_waypoint",      1.0, None),
        ("RT_with_two_waypoints", 1.0, None),
        ("RT_waypoints_arc",      1.0, None),
        ("RT_waypoint_90",        1.0, None),
        ("RT_waypoint_180",       1.0, None),
        ("RT_slalom",             1.5, None),
        ("RT_one_obstacle",       2.0, 0.05),
        ("RT_obstacle_field",     2.0, 0.05),
        ("RT_dense_obstacles",    2.5, 0.05),
        ("RT_gentle_hills",       1.0, None),
        ("RT_dunes",              1.0, None),
        ("RT_mixed",              2.0, 0.02),
    ]
    cl_kwargs: dict = {"lam": ewc_lam} if cl_method == "ewc" else {}
    return Mission(
        name=f"scenario_10_robust_curriculum_{cl_method}",
        tasks=[
            _make_task(
                t, int(train_timesteps * mult), eval_episodes, max_steps,
                ent_coef=ec,
            )
            for t, mult, ec in phase_plan
        ],
        cl_method=cl_method,
        cl_kwargs=cl_kwargs,
        seed=seed,
    )


def scenario_11_robust_generalist(
    cl_method: str = "ewc",
    train_timesteps: int = 300_000,
    eval_episodes: int = 12,
    max_steps: int = 1500,
    seed: int = 0,
    ewc_lam: float = 400.0,
    enable_gate: bool = True,
    enable_interim_eval: bool = True,
) -> Mission:
    """Scenario 11: 7-phase MIXED-DISTRIBUTION curriculum for generalization.

    Rebuilt from scenario_10's lessons. The single biggest change: every
    phase samples internally across a sub-distribution that already covers
    earlier phases as an anchor. The capstone (RC_full_random) is then a
    uniform draw over the cross-product of those sub-distributions, so any
    held-out evaluation is *in-distribution* by construction.

    Phase progression:
        0. RC_locomotion         — drive to a goal, 4-way bearing mix +
                                   "stop" anchor. Foundation skill.
        1. RC_path_following     — 0-5 waypoints in various configurations
                                   (line, twisty, arc, slalom). Includes
                                   25% no-waypoint anchor.
        2. RC_obstacle_avoidance — 0/1-2/3-6/7-10 obstacles. 25% none.
        3. RC_path_and_obstacles — composes 1+2. 20% waypoint-only and
                                   40% obstacle-only anchors keep both
                                   precursor skills alive.
        4. RC_terrain            — heightmap-only at three elevations.
                                   30% flat anchor.
        5. RC_terrain_plus       — terrain + obstacles. 40% flat-with-
                                   obstacles anchor + 20% terrain-only.
        6. RC_full_random        — capstone uniform draw. Match what we
                                   evaluate on.

    Mechanisms beyond the curriculum itself:
        * `min_success_to_advance` per phase (when `enable_gate=True`).
          Each phase keeps training in 50k-step chunks until 8-episode
          eval success crosses the threshold OR cumulative steps reach
          `train_timesteps × max_budget_multiplier`. Means "easy phases
          stop early, hard phases get more budget".
        * Interim eval every 25k steps (when `enable_interim_eval=True`)
          logged to `results.json::interim_eval`. Catches mid-phase
          regressions without waiting for the post-phase eval.
        * EWC λ=400. With the within-phase anchoring above, the CL
          method has a much easier protection job — most of the work is
          already done by the data distribution.

    Default budget: 300k × phase multipliers ≈ 2.7 M base. With the gate,
    well-converging phases may stop earlier; the worst case (every phase
    hits its max budget) is 300k × ~22 = 6.6 M total env steps. With
    `--n-envs 6` on M3 Air: 3-5 hours wall-clock per seed.
    """
    # (terrain, base_multiplier, ent_coef_override, advance_threshold)
    #
    # Tuning notes from runs v1 and v2:
    #  - v1 (ent_coef default, n_envs=4): phase 0 reached ~0.33 success.
    #    v2 (ent_coef=0.03, n_envs=12):  phase 0 reached only ~0.08.
    #    Bumping entropy on the foundation phases ACTIVELY HURT — random-
    #    init PPO already explores enough; the extra entropy bonus
    #    prevented the policy from committing to forward driving once
    #    the progress reward signal appeared. Reverted phases 0-1 to
    #    default ent_coef.
    #  - Obstacle phases (2-3) DO benefit from ent_coef=0.03 — there the
    #    "freeze near start" basin is genuinely deep because collision
    #    penalty + step cost can beat slow progress. Kept the bump.
    #  - Advance thresholds dropped to 0.45-0.55 across the board. The
    #    realistic ceiling within 2× max budget per phase is ~0.40-0.60
    #    on the early phases; higher thresholds mean the gate never
    #    fires (effectively a no-op). The point of the curriculum is to
    #    bootstrap a policy good enough for later phases to refine, not
    #    to perfect every phase individually.
    #  - Terrain phases keep default ent_coef (heightmap diversity
    #    provides state-space exploration without action-space noise).
    phase_plan: list[tuple[str, float, float | None, float | None]] = [
        ("RC_locomotion",          1.0, None,  0.55),
        ("RC_path_following",      1.5, None,  0.50),
        # Phase 2 (obstacle_avoidance) now uses a density+distance
        # curriculum biased toward easy episodes (30% no-obs / 35%
        # 1-2 obs at short distance). With that anchor in place the
        # forced ent_coef=0.03 became counter-productive (same lesson
        # as phase 0): the curriculum itself does the bootstrapping;
        # extra entropy just prevents commitment. Reverted to default.
        ("RC_obstacle_avoidance",  2.0, None,  0.45),
        # Phase 3 keeps 0.03 — composition of waypoints + obstacles is
        # genuinely harder than either alone, and the local minimum
        # ("freeze when the obs slots show clutter") is deeper here.
        ("RC_path_and_obstacles",  1.5, 0.03,  0.45),
        ("RC_terrain",             1.5, None,  0.55),
        ("RC_terrain_plus",        1.5, 0.02,  0.45),
        ("RC_full_random",         2.0, 0.02,  None),
    ]
    cl_kwargs: dict = {"lam": ewc_lam} if cl_method == "ewc" else {}

    interim_every = 25_000 if enable_interim_eval else 0

    tasks = []
    for terrain, mult, ec, threshold in phase_plan:
        tasks.append(_make_task(
            terrain,
            int(train_timesteps * mult),
            eval_episodes, max_steps,
            ent_coef=ec,
            min_success_to_advance=threshold if enable_gate else None,
            max_budget_multiplier=2.0,
            gate_check_interval=50_000,
            gate_eval_episodes=16,
            interim_eval_every=interim_every,
            interim_eval_episodes=5,
        ))
    return Mission(
        name=f"scenario_11_robust_generalist_{cl_method}",
        tasks=tasks,
        cl_method=cl_method,
        cl_kwargs=cl_kwargs,
        seed=seed,
    )


def scenario_12_joint_training(
    cl_method: str = "naive",
    train_timesteps: int = 5_000_000,
    eval_episodes: int = 20,
    max_steps: int = 1500,
    seed: int = 0,
) -> Mission:
    """Joint-training baseline — PPO on a uniform mixture of all curriculum
    sub-distributions.

    Trains on `RC_joint_mixture`, which per-episode picks one of the four
    distributions visited sequentially by scenario_13 (foundation /
    navigation / navigation_terrain / full_random) with equal probability.
    This is the proper "no curriculum, no CL machinery" upper-bound baseline:
    every batch is a uniform draw over the full training distribution.

    Previous version pointed at `RC_full_random` only — the hardest
    distribution — which gave PPO no successful trajectories to bootstrap
    from. The mixture preserves the same overall coverage as the curriculum
    but lets early policies learn from the easy episodes.

    `cl_method` defaults to 'naive' (no CL machinery — there's nothing to
    protect across phases, only one phase). The Runner still creates the
    CL object but it just trains PPO normally.

    Default budget: 5M timesteps. With --n-envs 6 on M3 Air, ~6-12 hours
    wall-clock per seed.
    """
    tasks = [
        _make_task(
            "RC_joint_mixture",
            train_timesteps,
            eval_episodes, max_steps,
            ent_coef=0.02,
            # No advance gate (only one phase, nothing to advance to).
            min_success_to_advance=None,
            # Cap budget at exactly `train_timesteps`. The default
            # `_make_task` multiplier of 2.0 was for gated phases that
            # might need extra budget to cross a threshold — irrelevant
            # here. Without this, the actual training budget silently
            # doubled (the `5M / 10M` in the previous log).
            max_budget_multiplier=1.0,
            # Interim eval every 100k so we can watch the learning curve
            # without polluting the log.
            interim_eval_every=100_000,
            interim_eval_episodes=10,
        )
    ]
    return Mission(
        name=f"scenario_12_joint_training_{cl_method}",
        tasks=tasks,
        cl_method=cl_method,
        cl_kwargs={},
        seed=seed,
    )


def scenario_13_integrated_curriculum(
    cl_method: str = "ewc",
    train_timesteps: int = 600_000,
    eval_episodes: int = 12,
    max_steps: int = 1500,
    seed: int = 0,
    ewc_lam: float = 400.0,
    enable_gate: bool = True,
    enable_interim_eval: bool = True,
) -> Mission:
    """4-phase INTEGRATED curriculum.

    Headline difference vs scenario_11: no phase trains a single skill
    in isolation. After the locomotion-bootstrap foundation, every phase
    has obstacles AND waypoints in EVERY episode. The CL method's job
    becomes "preserve smoothly-improving skills" instead of "preserve
    skill A while task switches to skill B" — much easier.

    Phases:
        0. RC_foundation         — locomotion + simple paths, no obstacles.
                                   Distance-curriculum bootstrap; 25% have
                                   a single waypoint to gently introduce
                                   the waypoint mechanic.
        1. RC_navigation         — paths + obstacles together, flat ground.
                                   Density+distance curriculum within phase.
        2. RC_navigation_terrain — same but on a heightmap. Layered
                                   suspension challenge.
        3. RC_full_random        — capstone uniform draw.

    Default budget: 600k base × multipliers below ≈ 3M. Adaptive gate
    cuts easy phases short. With --n-envs 6 on M3 Air, 30-90 minutes
    wall-clock per seed depending on how much the gate trims.
    """
    # Phases 1 and 2 get the bulk of the budget because they're where
    # the integrated skills are actually learned.
    phase_plan: list[tuple[str, float, float | None, float | None]] = [
        ("RC_foundation",          1.0, None,  0.70),
        ("RC_navigation",          2.0, 0.02,  0.45),
        ("RC_navigation_terrain",  1.5, 0.02,  0.40),
        ("RC_full_random",         2.0, 0.02,  None),
    ]
    cl_kwargs: dict = {"lam": ewc_lam} if cl_method == "ewc" else {}
    interim_every = 50_000 if enable_interim_eval else 0

    tasks = []
    for terrain, mult, ec, threshold in phase_plan:
        tasks.append(_make_task(
            terrain,
            int(train_timesteps * mult),
            eval_episodes, max_steps,
            ent_coef=ec,
            min_success_to_advance=threshold if enable_gate else None,
            max_budget_multiplier=2.0,
            gate_check_interval=50_000,
            gate_eval_episodes=16,
            interim_eval_every=interim_every,
            interim_eval_episodes=5,
        ))
    return Mission(
        name=f"scenario_13_integrated_curriculum_{cl_method}",
        tasks=tasks,
        cl_method=cl_method,
        cl_kwargs=cl_kwargs,
        seed=seed,
    )


def scenario_14_skill_sequence(
    cl_method: str = "naive",
    train_timesteps: int = 1_000_000,
    eval_episodes: int = 30,
    max_steps: int = 1000,
    seed: int = 0,
    ewc_lam: float = 1000.0,
) -> Mission:
    """Clean 3-task continual-learning forgetting benchmark.

    Built after a learnability audit established that BLOCKING-obstacle
    avoidance is not reliably learnable in this env (~5%), while other skill
    axes are. This sequence uses only individually-learnable tasks (measured
    single-task success in parentheses), each a distinct navigation SKILL, so
    the retention matrix reflects genuine forgetting rather than tasks that
    never learned:

        Phase 0  RC_locomotion     — drive to goals over varied distance and
                                      bearing; requires turning-from-rest (~92%)
        Phase 1  RC_path_following — track multi-waypoint chains (arcs, slalom,
                                      twisty); a different control behaviour
                                      from single-goal driving              (~50%)
        Phase 2  RC_terrain        — traverse procedural heightmaps (flat →
                                      hills → dunes); terrain adaptation     (~45%)

    None of the three has obstacles, so the geodesic progress reward reduces
    to Euclidean and the tasks are exactly the learnable axes. The three
    demand different policies (single-goal turning vs waypoint tracking vs
    dune traversal), so naive sequential training is expected to forget
    earlier skills — the signal the CL methods are compared on.

    Fixed budget per task (no adaptive gate) so every method trains each skill
    under identical conditions; retention is evaluated on all seen tasks after
    every phase (Runner default).
    """
    task_terrains = ["RC_locomotion", "RC_path_following", "RC_terrain"]
    cl_kwargs: dict = {"lam": ewc_lam} if cl_method in ("ewc", "l2", "mas", "hybrid") else {}
    tasks = [
        _make_task(
            terrain, train_timesteps, eval_episodes, max_steps,
            ent_coef=0.02,
            interim_eval_every=100_000,
            interim_eval_episodes=10,
        )
        for terrain in task_terrains
    ]
    return Mission(
        name=f"scenario_14_skill_sequence_{cl_method}",
        tasks=tasks,
        cl_method=cl_method,
        cl_kwargs=cl_kwargs,
        seed=seed,
    )


def scenario_15_obstacle_field(
    cl_method: str = "naive",
    train_timesteps: int = 20_000_000,
    eval_episodes: int = 30,
    max_steps: int = 1800,
    seed: int = 0,
) -> Mission:
    """Reactive obstacle-field navigation with realistic Curiosity-style
    control. Single long phase on `RC_field` (scattered obstacles + waypoints).

    Uses the obstacle-aware env config — lidar range scan + `(v, ω)` mobility
    (independent corner steering + point-turn, matching real Curiosity) —
    and premise-consistent LOCAL perception only (no global map/planner). Reaches
    ~90% on 3-5 obstacles + 1-2 waypoints, robust to denser fields (see
    docs/decision-log.md). This is the "unknown terrain, react to what you
    sense" formulation; the earlier on-path obstacle scenarios were an
    adversarial mis-design that was not learnable.

    NOTE: a from-scratch run benefits from a brief close-goal locomotion
    bootstrap before the field distribution (the /tmp/field20m.py recipe);
    the single-phase form here is the deployable/eval definition.
    """
    ekw = dict(use_lidar=True, control_mode="vw", geo_heading_obs=False,
               progress_reward_mode="best", collision_terminate_steps=1,
               collision_penalty=0.0, hit_penalty=5.0,
               stuck_in_collision_penalty=25.0, angvel_penalty_scale=0.15)
    tasks = [
        _make_task(
            "RC_field", train_timesteps, eval_episodes, max_steps,
            ent_coef=0.01, max_budget_multiplier=1.0,
            interim_eval_every=500_000, interim_eval_episodes=20,
            env_kwargs=ekw,
        )
    ]
    return Mission(
        name=f"scenario_15_obstacle_field_{cl_method}",
        tasks=tasks, cl_method=cl_method, cl_kwargs={}, seed=seed,
    )


def scenario_16_full_curriculum(
    cl_method: str = "naive",
    train_timesteps: int = 1_000_000,
    eval_episodes: int = 30,
    max_steps: int = 2000,
    seed: int = 0,
    ewc_lam: float = 1000.0,
) -> Mission:
    """Complete skill curriculum for the CL comparison — one policy learns four
    skills in sequence and we measure how much each new skill erodes the old:

        Phase 0  RC_c_locomotion — drive to a goal (no obstacles, no waypoints)
        Phase 1  RC_c_tracking   — follow a 1-2 waypoint chain (no obstacles)
        Phase 2  RC_c_avoidance  — weave a 5-gate slalom (no waypoints)
        Phase 3  RC_c_combined   — slalom AND in-gap waypoints (everything)

    Every phase shares ONE obstacle-capable env config (lidar + Curiosity vw
    control), so the obs/action space is constant and the policy carries weights
    across phases — the prerequisite for measuring forgetting. After each phase
    the Runner re-evaluates all earlier phases, giving the retention matrix and
    forgetting metric across the whole skill sequence.

    Orthogonal to the CL method, run under any perception mode via
    `run_scenario.py --perception {privileged,reactive,slam}`: the obstacle
    phases (2, 3) exercise it; the obstacle-free phases (0, 1) are unaffected.
    This yields the full (CL method × perception) grid on a task that tests
    locomotion, target tracking, obstacle avoidance, AND forgetting together.
    """
    ekw = dict(use_lidar=True, control_mode="vw", progress_reward_mode="best",
               collision_terminate_steps=1, collision_penalty=0.0,
               hit_penalty=8.0, stuck_in_collision_penalty=25.0,
               proximity_penalty_scale=0.28, proximity_safety_dist=1.8,
               angvel_penalty_scale=0.15)
    phase_terrains = ["RC_c_locomotion", "RC_c_tracking",
                      "RC_c_avoidance", "RC_c_combined"]
    tasks = [
        _make_task(
            terrain, train_timesteps, eval_episodes, max_steps,
            ent_coef=0.01, env_kwargs=ekw,
        )
        for terrain in phase_terrains
    ]
    cl_kwargs = {"lam": ewc_lam} if cl_method in ("ewc", "hybrid") else {}
    return Mission(
        name=f"scenario_16_full_curriculum_{cl_method}",
        tasks=tasks, cl_method=cl_method, cl_kwargs=cl_kwargs, seed=seed,
    )


def scenario_02_threat_classes(**_kwargs) -> Mission:
    """Scenario 2 (threat classification track) — NOT YET IMPLEMENTED.

    The thesis has a parallel supervised track (sequential threat-class
    classification on telemetry) that shares only the CL methods with the nav
    track. The classifier env + dataset don't exist yet. See `docs/roadmap.md`
    §4 for the implementation outline (~300 LOC; small CNN/MLP, no MuJoCo).
    """
    raise NotImplementedError(
        "scenario_02_threat_classes: supervised threat-classification track is "
        "not implemented. The nav-side equivalent is scenario_02_three_terrains. "
        "See docs/roadmap.md §4."
    )


def scenario_06_fusion(**_kwargs) -> Mission:
    """Scenario 6 (multi-task fusion) — NOT YET IMPLEMENTED.

    Multi-task fusion of nav + threat tracks under a shared encoder. Blocked
    on scenario_02_threat_classes. See `docs/roadmap.md` §4.
    """
    raise NotImplementedError(
        "scenario_06_fusion: requires the threat-classification track. "
        "See docs/roadmap.md §4."
    )


SCENARIO_REGISTRY = {
    "scenario_01_sequential_terrains": scenario_01_sequential_terrains,
    "scenario_02_three_terrains": scenario_02_three_terrains,
    "scenario_03_order_sensitivity": scenario_03_order_sensitivity,
    "scenario_04_replay_sweep": scenario_04_replay_sweep,
    "scenario_05_full_terrain_curriculum": scenario_05_full_terrain_curriculum,
    "scenario_07_blocked_arc": scenario_07_blocked_arc,
    "scenario_08_blocked_arc_hills": scenario_08_blocked_arc_hills,
    "scenario_09_curriculum_arc": scenario_09_curriculum_arc,
    "scenario_10_robust_curriculum": scenario_10_robust_curriculum,
    "scenario_11_robust_generalist": scenario_11_robust_generalist,
    "scenario_12_joint_training": scenario_12_joint_training,
    "scenario_13_integrated_curriculum": scenario_13_integrated_curriculum,
    "scenario_14_skill_sequence": scenario_14_skill_sequence,
    "scenario_15_obstacle_field": scenario_15_obstacle_field,
    "scenario_16_full_curriculum": scenario_16_full_curriculum,
    # Stubs (raise NotImplementedError on call but registered so they're discoverable):
    "scenario_02_threat_classes": scenario_02_threat_classes,
    "scenario_06_fusion": scenario_06_fusion,
}


def get_scenario(name: str, perception: str | None = None, **kwargs) -> Mission:
    if name not in SCENARIO_REGISTRY:
        raise KeyError(f"Unknown scenario {name!r}. Known: {sorted(SCENARIO_REGISTRY)}")
    mission = SCENARIO_REGISTRY[name](**kwargs)
    # PERCEPTION MODE axis: rebuild task env factories with the override so any
    # scenario can be run under privileged / reactive / slam perception.
    mission = apply_perception(mission, perception)
    # Canonicalize the mission name to "{registry_key}_{cl_method}" so paths
    # under results/<scenario>/<method>/seed_<N>/ always match the name the user
    # passes to --compare. Without this, factories that built shorter labels
    # like "scenario_01_<method>" caused results to land under results/scenario_01/
    # but --compare scenario_01_sequential_terrains would look elsewhere.
    mission.name = f"{name}_{mission.cl_method}"
    return mission
