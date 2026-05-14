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


def _env_factory(terrain_name: str, max_steps: int = 500):
    def _make(seed: int):
        return RoverNavEnv(terrain=terrain_name, max_steps=max_steps, seed=seed)
    return _make


def _make_task(
    terrain_name: str,
    train_timesteps: int,
    eval_episodes: int,
    max_steps: int,
    *,
    ent_coef: float | None = None,
) -> Task:
    return Task(
        terrain_name,
        _env_factory(terrain_name, max_steps),
        train_timesteps=train_timesteps,
        eval_episodes=eval_episodes,
        eval_max_steps=max_steps,
        ent_coef=ent_coef,
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
    # Stubs (raise NotImplementedError on call but registered so they're discoverable):
    "scenario_02_threat_classes": scenario_02_threat_classes,
    "scenario_06_fusion": scenario_06_fusion,
}


def get_scenario(name: str, **kwargs) -> Mission:
    if name not in SCENARIO_REGISTRY:
        raise KeyError(f"Unknown scenario {name!r}. Known: {sorted(SCENARIO_REGISTRY)}")
    mission = SCENARIO_REGISTRY[name](**kwargs)
    # Canonicalize the mission name to "{registry_key}_{cl_method}" so paths
    # under results/<scenario>/<method>/seed_<N>/ always match the name the user
    # passes to --compare. Without this, factories that built shorter labels
    # like "scenario_01_<method>" caused results to land under results/scenario_01/
    # but --compare scenario_01_sequential_terrains would look elsewhere.
    mission.name = f"{name}_{mission.cl_method}"
    return mission
