# Rewards

> Code: `src/rover_cl/envs/nav.py::RoverNavEnv.__init__` (defaults) and
> `RoverNavEnv.step` (composition).

## Reward formula

Per-step reward is the sum of these terms:

```
reward = progress_reward_scale * progress
       − step_cost
       − proximity_penalty                       (disabled by default)
       − collision_penalty           if collision
       − hit_penalty                 if new_hit       (one-shot on contact entry)
       − tipped_penalty              if tipped
       − early_terminate_penalty     if stuck_in_collision OR stuck_no_progress
       + waypoint_reached_bonus      if a waypoint was crossed this step
       + speed_bonus                 if a checkpoint was crossed this step
       + goal_bonus                  if success this step
```

`progress = previous_distance_to_target − current_distance_to_target`, so
moving toward the current target gives positive progress.

## Defaults (in `nav.py`)

| Term | Default | Why this value |
|---|---|---|
| `progress_reward_scale` | **5.0** | At top speed ~0.6 m/s with 0.025 s/env-step that's ≈ +0.075 / step — dominant per-step signal. Tuned up from 1.0 → 3.0 → 5.0 across iterations as we kept finding the rover preferred "freeze" over "move." |
| `goal_bonus` | 50.0 | Big terminal payoff. |
| `waypoint_bonus` | 5.0 | Intermediate checkpoint payoff — 1/10 of goal so it doesn't dominate the final-goal signal. |
| `speed_bonus_scale` | **1.0** | At step 0 the rover gets DOUBLE the base bonus (`base × (1 + scale × 1.0)`); at the deadline, only the base. Linear discount on `1 − step_count / max_steps`. Pushes the policy to *rush* checkpoints instead of dawdling near them. |
| `step_cost` | 0.01 | Mild urgency. |
| `collision_penalty` | **3.0/step** | Sustained cost while in contact. Bumped from 1.0 once we discovered `_detect_collision` was filtering out obstacle hits entirely (see below) — once it actually fired, 1.0/step was too cheap. |
| `hit_penalty` | **10.0** | One-shot on collision *entry* (transition `False → True`). Without this, a 3-step graze only costs 9 vs the +50 goal bonus — not enough to deter. |
| `tipped_penalty` | 20.0 | Episode-ending. |
| `proximity_penalty_scale` | **0.0** (disabled) | Was 0.05 → 0.15 → 0.03 → 0.0 across iterations. Created freeze local optima — once the hit_penalty + collision_penalty deterrent worked, the proximity nudge stopped earning its keep. |
| `proximity_safety_dist` | 1.0 m | Only matters if `proximity_penalty_scale > 0`. |
| `early_terminate_penalty` | 5.0 | Small one-shot when either stuck-guard fires. The bigger signal is the *episode ending*, not the penalty. |

## Speed-bonus mechanic

When a checkpoint fires (`waypoint_reached_bonus > 0` or `success`):

```python
time_factor = max(0.0, 1.0 - step_count / max_steps)
speed_bonus = speed_bonus_scale * base_bonus * time_factor
```

Example with `speed_bonus_scale=1.0`:

- Waypoint hit at step 1 / 1500 → reward includes ≈ `5 + 5 × 0.999 ≈ 9.99`
- Same hit at step 751 / 1500 → ≈ `5 + 5 × 0.5 ≈ 7.5`
- Same hit at step 1500 / 1500 → `5 + 0 = 5`

Goal hit doubles in the same way: 50 → 100 if reached immediately.

## Early-termination guards

`collision_terminate_steps = 30` and `stuck_window_steps = 200` each end
the episode and apply `−early_terminate_penalty`. Reasons we added them:

- **Stay-stuck-in-collision** was the dominant failure mode. Before
  termination, a policy that rammed an obstacle and stayed there for 620
  out of 1500 steps got mean return ≈ −1866 — much worse than backing
  off, but PPO converged to it anyway because backing off lost positive
  progress reward without escaping the "drive toward goal" gradient.
  Terminating after 30 consecutive collision steps removes this
  equilibrium entirely.
- **Freeze-near-start** is the other local optimum: rover does nothing,
  earns `-step_cost ≈ -15` per episode, calls it a day. The no-progress
  guard turns this into a clear termination so the policy can't sit on
  it.

`stuck_min_progress = 0.5 m` — the rover must reduce `d_target` by at
least 0.5 m within `stuck_window_steps` to reset the counter. Tuned to
allow tight maneuvering (slow ≠ stuck) while still catching genuine
freezes.

## The collision-detection bug (historical, **fixed**)

Before commit X, `_detect_collision` filtered contacts by **other body
name**, rejecting anything attached to body 0 ("world"). But obstacles in
`compose_scene` are top-level `<geom>` tags in `<worldbody>` — they
**all** live in body 0. So every obstacle contact was silently rejected,
collision_penalty never fired, and the policy learned to drive freely
through obstacles. Fix: check the contact **geom name** for the
`obs_*` prefix instead.

Combined with the arm-stow fix (the arm at `ctrl=0` extended 2 m forward
and was hitting obstacles before the chassis would), this is what made
collision rewards finally start influencing behavior.

## What we explicitly chose NOT to do

- **Don't terminate the episode on collision** (only on persistent
  collision). One-step grazes are punished by `hit_penalty + collision_penalty`
  but don't kill the episode — otherwise the policy learns to be timid.
- **Don't add a heading-toward-target reward**. It's algebraically
  equivalent to part of `progress_reward_scale × progress`; adding it
  separately just complicates tuning.
- **Don't use reward shaping based on lidar minimum distance**. We tried
  this (`proximity_penalty_scale > 0`) twice and twice ended up reducing
  it to zero because it created local minima around tight gaps.
