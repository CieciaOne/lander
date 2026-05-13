# Design notes

A living knowledge base for the rover continual-learning project. The thesis
itself lives in `docs/plan.md` / `docs/research_overview.md`; these notes are
the **why** behind the code — design decisions, things we tried, things we
ruled out, and what to update when behavior changes.

## When to update what

- **You change env semantics** (action space, obs shape, reward terms, an
  early-termination rule, etc.) → update `environment.md`, `observations.md`,
  or `rewards.md` as appropriate, and bump the relevant entry in `changelog.md`.
- **You change how training is orchestrated** (PPO config, VecEnv,
  curriculum mechanics, CL hook structure) → update `training.md`.
- **You add or retire a scenario / terrain** → update `scenarios.md`.
- **You change the visual style or report contents** → update
  `evaluation.md`.
- **You fix a non-obvious bug or document a "looks broken but isn't"**
  → add or update the relevant section in the per-topic file AND mirror the
  one-liner into `CLAUDE.md`'s "Things that look broken but aren't" so it's
  visible from the entry point.

## Index

| File | Covers |
|---|---|
| [`environment.md`](environment.md) | MuJoCo setup, rover footprint, action space (Ackermann), terrain framework, hfield support, start-pose jitter |
| [`observations.md`](observations.md) | Observation design history (lidar → AABB bounding boxes), Minkowski inflation by rover radius, why this representation |
| [`rewards.md`](rewards.md) | Every reward term in `RoverNavEnv.step`, why each exists, the values we tuned and why, early-termination guards |
| [`training.md`](training.md) | PPO defaults, parallel rollout collection via SubprocVecEnv, CL methods (`naive` / `replay` / `ewc`), the `post_train` hook |
| [`scenarios.md`](scenarios.md) | Every scenario in the registry with its goal, expected outcome, and how to run it |
| [`evaluation.md`](evaluation.md) | `EpisodeTrajectory` capture, eval seed variation, top-down report contents, thesis plot style |
| [`changelog.md`](changelog.md) | Chronological summary of design changes |

## Conventions

- **Forward direction**: rover's body **+Y** axis. Mast and front wheels both
  point +Y. Inside the env `action[0] = throttle > 0` already produces +Y
  motion — the env negates internally because the wheel-axle convention
  rolls the rover in −Y under positive ctrl.
- **Reward sign**: positive numbers add to reward, negative numbers
  subtract. `progress = previous_distance - current_distance`, so reducing
  distance gives positive `progress`.
- **Seeds**: the mission seed is the offset; training envs see
  `seed + phase * 1000 [+ worker_idx]`, eval envs see
  `seed + 10000 * (phase + 1) + 100 * task_index + episode_index`. Adding
  10 000 keeps eval and training seeds disjoint.
- **Polish**: source assets in `data/` were originally Polish-labelled. All
  in-tree docs and code are English (translation happened in 2026-05);
  don't reintroduce Polish strings.
