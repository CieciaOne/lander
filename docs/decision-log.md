# Decision Log

Chronological record of major project decisions, with rationale and evidence, so
the experiment can be written up accurately later. Newest at the bottom.

Format per entry: **date — decision** · why · evidence/alternatives considered.

---

## 2026-07 — Diagnosis: weak CL results were "tasks never learned", not forgetting

The retention matrices from scenarios 10/11/13 were full of zeros that *looked*
like catastrophic forgetting but were mostly tasks that never learned in the
first place. Root causes found and fixed in `envs/nav.py`:

- **Obstacle-visibility bug (critical):** `_apply_terrain_roll` wrote rolled
  obstacle positions only to the MuJoCo model; the observation read them from
  the static `TerrainSpec`, so on every randomized (`RC_`/`RT_`) terrain the
  policy saw phantom obstacles at the origin and was blind to the real ones.
  This invalidates all prior obstacle-phase CL results. Fixed: obs reads live
  `data.geom_xpos` / `model.geom_size`.
- **Steering stall:** `MAX_STEER_RAD` 1.0→0.40 rad (full steer stalled the
  rover to ~-4°/s).
- Obs 40→44 (tilt + prev_action, fixed-scale normalization); reward rebalanced;
  EWC/L2/MAS/hybrid penalty moved into the PPO loss via gradient hooks.

## 2026-07 — Reframed the CL benchmark onto individually-learnable tasks (`scenario_14_skill_sequence`)

**Decision:** build the CL comparison on 3 learnable navigation SKILLS —
`RC_locomotion` → `RC_path_following` → `RC_terrain` — instead of the
obstacle-maze curricula.
**Why:** a clean forgetting benchmark needs tasks that each learn to a
measurable level; obstacle-maze phases weren't learnable (see below).
**Evidence:** single-task success ~92% / ~50% / ~45%; naive shows clear
forgetting (path_following 0.60→0.07 after the terrain phase), and the 7-method
comparison is clean (EWC best at avg-retention 0.744 / forgetting 0.000; all CL
methods beat naive 0.289 / 0.267). ~2M steps/phase — comparable to continual-RL
benchmarks (Continual World uses ~1–3M/task). This is the thesis's core result.

## 2026-07 — Realistic rover control: added `(v, ω)` mode with independent corner steering

**Decision:** replace the car-like Ackermann action (single mirrored steer,
~2.75 m turn radius, no point-turn) with an opt-in `control_mode="vw"` — a
proper rover mobility controller that sets each corner knuckle's angle and each
wheel's speed from ICR geometry (Curiosity-style explicit steering), enabling
point-turns.
**Why:** the Ackermann model was physically unrealistic for a rocker-bogie rover
and capped maneuverability, causing obstacles to be clipped.
**Evidence:** validated kinematically — commanded (v=0, ω=0.5) gives a true
point-turn (ω=0.49, net displacement 0.04 m). Lifted the 1-obstacle ceiling from
~40% (Ackermann) to a 60% peak.

## 2026-07 — Obstacle avoidance: end-to-end RL hits a persistent local optimum; choose a hierarchical (planner + tracker) approach

**Problem:** even with realistic `vw` control, lidar, a geodesic NavField reward,
and a planner-guided "bent bearing" observation, end-to-end PPO plateaus at
~15–40% single-obstacle success (60% only unstably). The rover clips/wedges into
obstacles as a robust LOCAL OPTIMUM that survived ~8 reward/control/perception
interventions. **More training steps will NOT fix this** — it is a local
optimum, not under-training (stable runs are flat, not slowly rising). MJX/GPU is
unavailable (Mac-only), so large-scale end-to-end training is also impractical.

**Considered then REJECTED — hierarchical global planner:** using the NavField
Dijkstra planner to supply a collision-free path violates the thesis premise
(rover dropped into UNKNOWN terrain, learning on the fly). A global planner needs
the full obstacle map up front, which the deployed rover would not have. The user
raised this and it is decisive.

**Key clarification (privileged information):**
- Using ground-truth obstacle positions in the REWARD during training is
  legitimate ("privileged reward shaping" — the reward exists only in sim at
  train time; the deployed policy never sees it).
- Using it in the OBSERVATION is NOT legitimate — it hands the deployed policy a
  global map it wouldn't have. **The "bent bearing" observation built earlier
  (the ~32–40% zero-shot obstacle result) was privileged in this sense and is
  therefore NOT premise-consistent — it should not count toward the thesis.**

**Decision — premise-consistent formulation:** obstacle avoidance stays
END-TO-END REACTIVE from LOCAL perception. Observation = lidar (local range scan)
+ straight-line goal bearing + pose/velocity/tilt; NO global map, NO bent bearing.
The geodesic NavField may be used ONLY as training-time reward shaping. Control =
`vw` (realistic). The policy must learn reactive avoidance ("lidar sees obstacle
→ steer around → resume toward goal"), which is the genuine "unknown terrain"
problem and is standard for isolated obstacles in the literature. Reaching high
success is a real RL tuning effort (moderate collision penalty + shaped reward +
curriculum), not a quick fix.
**Alternatives rejected:** (a) more steps — doesn't escape a local optimum;
(b) large-scale end-to-end RL — needs MJX/GPU we don't have; (c) hierarchical
global planner — violates the unknown-terrain premise.

## 2026-07 — Root cause of training instability: geodesic reward → switch to smooth Euclidean progress

**Finding:** every unstable obstacle-training run this session (repeated
catastrophic loco→0 collapses) used `progress_reward_mode="geodesic"`. The
grid-Dijkstra geodesic distance changes DISCONTINUOUSLY when the shortest path
jumps to a different route as the rover moves; reward = 5·Δgeodesic then spikes,
producing high-variance updates that collapse PPO. `target_kl`/lower-LR did not
fix it.
**Decision:** use smooth **Euclidean** progress shaping (`progress_reward_mode
="best"`) for reactive obstacle training. Bonus: it is fully UNPRIVILEGED
(uses no obstacle ground truth at all), so the reactive formulation now uses no
privileged info anywhere — cleanest possible for the thesis premise.
**Evidence:** switching reward (same everything else) turned repeated collapses
into stable training — loco steady at 1.00, 1-obstacle steady at ~0.52–0.60 (was
oscillating 0.00–0.50). The reactive policy sees only lidar + straight goal
bearing + pose; obstacle avoidance is learned purely from lidar + collision
penalty. Geodesic NavField retained only for the (privileged, train-time)
diagnostics, not the reward.

## 2026-07 — Reactive obstacle-avoidance ceiling reached (~50% single-obstacle); 80-90%-on-hard target not met via end-to-end PPO here

After the euclidean-reward stability fix, the premise-consistent reactive policy
(lidar + straight goal bearing, NO privileged info, vw control) trains STABLY but
plateaus: **loco 1.00, single-obstacle ~0.45-0.52, HARD (2-3 obstacles + waypoints)
~0-0.04.** ~16 obstacle-training configurations were tried (collision-penalty
sweep, clearance-speed penalty, geodesic vs euclidean reward, curriculum, LR/kl/
entropy, denser lidar, wider considerations, longer training).

**Measured failure mode (stable ~0.5 policy, 40 × 1-obstacle):** 45% success, 23%
clip (obstacle IN view — turns too late), 33% FREEZE (stops short — too timid).
Not a field-of-view problem (all collisions in-view). It is the classic
freeze-vs-clip tension of a scalar collision penalty: bolder → clips, more timid →
freezes; neither regime exceeds ~50%. Single-obstacle ~0.5 mathematically caps
HARD (2-3 obstacles) near 0.5²–0.5³ ≈ 0.12–0.25 at best, so HARD ≈ 0 follows.
**More steps does not help** (best is early; longer training degrades — confirmed).

**Honest conclusion:** 80-90% on hard multi-obstacle+waypoint via end-to-end
reactive PPO is not achievable in this setup (this rover, laptop-scale rapid
iteration, no MJX/GPU). Genuine paths that MIGHT reach it (each uncertain / larger
effort): recurrent policy (RecurrentPPO) for temporal memory; systematic reward+
curriculum engineering at GPU scale; or a learned local costmap + local planner
(premise-consistent since it uses only sensed data). The solid, comparable thesis
deliverable remains scenario_14 (7-method CL comparison on learnable terrain
skills). Decision pending with user.

## 2026-07 — Honest status after exhaustive obstacle-avoidance effort (~20 training runs + hand-controller diagnostics)

**Corrections to earlier claims:**
- The hand-coded classical controllers (VFH / decisive-VFH / APF) I built all scored ~0% on single-obstacle — but this was due to MY controller bugs (wrong turn sign, bang-bang oscillation, too-wide "blocked" window, creep-stall), NOT an env defect. The env is navigable: locomotion (goal-following) hits 100%, and the RL policy itself reached ~0.68 single-obstacle at one point — i.e. RL does BETTER than my hand controllers.
- So obstacle avoidance is NOT fundamentally unlearnable here; the blocker is TRAINING STABILITY.

**Established ceiling / findings:**
- Stable config = MlpPolicy [128,128] + Euclidean progress ("best") + vw control + lidar (15 rays) + immediate collision termination. Reaches loco 1.00, single-obstacle **~0.5** stably; HARD (2-3 obstacles + waypoints) **~0**.
- Single-obstacle ~0.5 mathematically caps HARD (chaining) near 0.
- The policy transiently reached ~0.68 single-obstacle but could not hold it together with locomotion (skills interfere in the small net; loco→0 collapses).
- Bigger net [256,256] made training WORSE (locomotion itself collapsed, even with no obstacles) — not a capacity fix, a destabilizer with these hyperparameters.
- Also found a real task-setup issue: obstacles can spawn ~1 m from the start (`sample_obstacles_along_path`, t from 0.15), giving the big/slow rover no reaction distance on some episodes.

**Honest conclusion:** >80% on hard multi-obstacle+waypoint was NOT achieved via end-to-end PPO in this setup after exhaustive tuning. The consistent failure is training instability / skill interference, not a single fixable bug. Highest-confidence path to the target: a CORRECT classical local reactive planner (DWA/VFH done properly — premise-consistent, uses only lidar+goal, robustly hits 80-90%), with RL/CL layered on terrain/dynamics adaptation (scenario_14 remains the solid CL result). Awaiting user direction; will not burn more compute on non-converging end-to-end tuning.

## 2026-07 — Grounded rover in real Curiosity specs + redesigned obstacle scenario + 20M training

**Real Curiosity (researched):** 3.0×2.8 m, 0.5 m wheels, top speed 0.04 m/s (!), each front+rear wheel INDEPENDENTLY steered → turns in place AND arcs; navigates by slow careful planning (~200 m/day). Implications: (1) our `vw` control (independent corner steering + point-turn) is exactly Curiosity's real mobility — the approach is realistic; (2) our rover at 0.66 m/s is already 16× faster than real, so slowness is realistic and we needn't add agility; reverted OMEGA_MAX to 0.8 (realistic moderate).

**Scenario redesign (the likely mis-design):** previously obstacles were placed ON the direct start→goal line, as close as ~1 m from the start (rover spawns almost touching one) — unrealistic and unavoidable on some episodes. And a STAGED curriculum carried one policy through changing distributions, causing loco↔obstacle interference / collapses. Mapless-nav RL that reaches 85-90% instead trains on a FIXED SCATTERED obstacle field, LONG. Redesigned: obstacles scattered across the arena with ≥3 m clearance from start/goal and ≥2.2 m apart; single long training on the full mixed distribution (brief loco bootstrap + 18M main) rather than destabilizing stages; 20M total (vs ~3-4M before — the user rightly noted our training was short vs literature's 10-50M).

**Benchmark method (point 1, TODO):** add JOINT/multitask training as the CL upper-bound benchmark on scenario_14 (train on all 3 tasks simultaneously — the ceiling all CL methods approach). Deferred until the 20M obstacle run frees the cores.

## 2026-07 — SOLVED: obstacle navigation reaches 80-90% — root cause was SCENARIO DESIGN, not RL

The redesigned scattered-obstacle-field scenario + long training (20M) + realistic vw control hit the target on the FIRST main-phase checkpoint and held it:
- loco 1.00, field (3-5 scattered obstacles) 1.00, HARD (3-5 obstacles + 1-2 waypoints) 0.88-0.96.
- Robustness (best policy, zero-shot on harder distributions): dense 6-8 obstacles 0.80; up-to-3 waypoints 0.90; tighter clearance (5-7 obs, 2.0 m) 0.77.

**Definitive conclusion:** the ~28 failed obstacle runs were fighting a MIS-DESIGNED task, not an RL limitation. Placing obstacles ON the start→goal line (as close as ~1 m from spawn) made avoidance adversarial/often-impossible, and staged curricula caused loco↔obstacle interference. Switching to a realistic SCATTERED obstacle field (≥3 m clearance from start/goal) + a single long training run on the full mixed distribution (no destabilizing stages) + Curiosity-style vw control made it readily learnable to ~90%. Best policy saved at results/_obstacle_nav/scattered_field_best.zip. Next: promote the scattered-field terrain + scenario into the codebase (reproducible), and it can serve as a learnable obstacle task in the CL comparison.

## 2026-07 — CORRECTION: the "SOLVED ~90%" was HOLLOW — scattered field was trivial, not genuine avoidance

The user inspected the generated maps and caught the flaw: "na żadnej mapie przeszkoda nie jest przeszkodą, dystanse są bardzo krótkie a nasz łazik telepie się na lewo i prawo." He was right, and I verified it by measurement:
- The `sample_scattered_obstacles` distribution placed obstacles arena-wide, so on a straight start→goal line **only ~2% of episodes had any obstacle actually blocking the path** (within footprint radius of the segment). The rover scored ~90% by driving essentially straight — hollow locomotion, not avoidance.
- Confirming: the saved "90%" policy, tested on a GENUINELY blocking distribution, scored **0.17**. It never learned to avoid.

**The over-correction lesson:** the earlier `sample_obstacles_along_path` was too hard (obstacles ~1 m from spawn, unavoidable). I over-corrected to scattered (obstacles nowhere near the path). Neither is a fair-but-genuine avoidance task.

**The middle ground — `sample_obstacles_corridor` (new, in randomization.py):** obstacles spread over path fraction t∈[0.30, 0.90] with lateral ±1.1 m, giving BOTH (a) ~2.8 m reaction clearance from the spawn (fair) AND (b) genuine blocking — measured **85% of episodes blocked, mean 2.9 blocking obstacles**. `RC_field` now uses this. This is the honest hard task.

**Action:** retraining 20M on the corrected corridor-blocking `RC_field` (loco bootstrap + 18M main). Will report the HONEST success rate — whatever it is. The `obstacle-avoidance-solved` memory has been corrected to `obstacle-avoidance-status` (not solved; retrain in progress). Wobble ("telepie się") still to address if the retrain converges.

## 2026-07 — Genuine obstacle avoidance: diagnosed flat-zero, fixed with proximity shaping + anchored curriculum

Retrained on the corrected corridor-blocking `RC_field` (85% episodes genuinely blocked). Findings, in order:

1. **Flat zero (2.55M steps): field/HARD = exactly 0.00 while loco = 1.00.** With immediate collision termination (`collision_terminate_steps=1`) and NO dense obstacle gradient, the only obstacle signal is a terminal −30 on contact. Learning avoidance requires randomly discovering a complete detour-to-goal trajectory, but any exploratory graze ends the episode first — so with 85% of straight paths blocked and 6 envs, the avoidance gradient is flat. This is an artificial-potential-field problem fed to RL with the potential field OFF.

2. **Fix 1 — dense proximity shaping.** The env already had a `proximity_penalty` (penalty ∝ closeness to inflated obstacle AABB, fades to 0 at `proximity_safety_dist`), disabled by default. Enabling it gives a continuous steer-away gradient that combines with the forward progress pull into a go-around resultant. Validation: single-obstacle success climbed off zero to 0.64 within ~0.5M steps — proving avoidance IS learnable; the earlier wall was a missing reward term, not an RL limitation.

3. **Instability was skill isolation, not hyperparameters.** With proximity on but the c1 stage training on 100% single-obstacle episodes, locomotion (open-field) got zero gradient and collapsed to 0.00 while c1 stayed high — the policy thrashed between skills. Fix 2 — **bake an obstacle-free anchor (~15-30% of episodes) into EVERY curriculum stage** so no skill is ever removed from the gradient (the integrated-curriculum lesson, applied to the training DISTRIBUTION not just the scenario). With the anchor, loco held rock-stable at 1.00 across all obstacle stages.

4. **Shaping magnitude matters.** At `proximity_penalty_scale=0.15` the penalty at the inflated boundary (~0.15) only matched the per-step progress reward (~0.165) — they cancelled to a stall, so c1 climbed slowly (0.08→0.32 over ~0.6M). Bumped to 0.28 (+ `safety_dist=1.7`, lr 2e-4) so the penalty EXCEEDS progress near obstacles → net lateral deflection. Anchor keeps this stable.

**Current design:** warm-start from the loco-stable checkpoint → curriculum A(1 obs)→B(1-2)→C(2-3)→D(2-5 + 35% waypoints), obstacle-free anchor in every stage, proximity 0.28. Recipe /tmp/field_curric3.py. Honest number pending. Key reframe: obstacle avoidance here is NOT unlearnable — the blockers were (a) a switched-off shaping term and (b) skill isolation in the training distribution, both now fixed.

## 2026-07 — Reactive-shaping ceiling established; switching to geodesic progress reward

Anchored curriculum + proximity shaping (0.28) + relaxed training termination (=8) result: single-obstacle avoidance (c1) climbs to a solid **~0.72** with loco rock-stable at 1.00, but **field (3-5 obstacles) plateaus at ~0.04-0.08** across the whole stage-B run. The gap is structural, not a tuning issue: a purely REPULSIVE proximity field has local minima wherever obstacles cluster (pockets between two obstacles, corridors), and multi-obstacle blocking compounds them — so reactive shaping caps dense-field success well below the 80-90% target. This confirms the earlier "reactive ceiling" note with a clean number (c1 0.72 vs field 0.08).

**Switching to `progress_reward_mode="geodesic"`:** the env computes an obstacle-aware Dijkstra distance-to-target on the inflated-obstacle grid and rewards reducing IT. The gradient then points along the collision-free route rather than straight through the obstacle — eliminating the head-on local minimum that traps the repulsive field. This is premise-consistent: the obstacle map is used only in the TRAINING REWARD (shaping), never in the observation (the policy still sees only lidar + straight goal bearing). Earlier geodesic attempts collapsed loco via reward-variance, but that was BEFORE the obstacle-free anchor existed; with the anchor, n=0 episodes have geodesic≡Euclidean so locomotion stays supervised and stable. Warm-starting from the best reactive policy (curric4_best: loco 1.0, c1 0.72). Recipe /tmp/field_curric5.py.

## 2026-07 — THE feasibility bug: field task was ~74% physically impossible; fixed with guaranteed-feasible slalom

Measured feasibility of the corridor `field` configs with the env's own NavField (Dijkstra on the 0.9 m-inflated grid = "can the 1.8 m-diameter rover fit"): only **~26% of 3-5 obstacle configs had ANY collision-free path**. Two compounding causes, both from placing obstacles by PATH FRACTION while inflation is ABSOLUTE (~1.5 m):
  1. Random ±1.1 m lateral offsets packed inflated obstacles (~3 m wide) into a narrow band → merged into an impassable wall.
  2. Obstacles near t≈0.9 on a short path sit <1 m from the goal; their inflation ENGULFS the goal cell → Dijkstra can't propagate out → whole config unreachable.

So every "field" number I'd been chasing (reactive 0.08, geodesic peak 0.32) was near the FEASIBLE ceiling — the policy wasn't failing to learn, the task was mostly impossible. This is the original `along_path` mistake in a subtler guise, and I over-corrected into it twice (trivial-scattered → impassable-corridor).

**Fix — `sample_obstacles_slalom` (randomization.py):** each obstacle straddles the centreline (straight path 100% blocked, genuine weave required) but is pushed to one alternating side so a clear band ≥ `gap_min`=2.2 m (rover-centre clearance, ~0.4 m margin) always remains on the other side. Obstacles kept ≥ `clearance_ends`=2.6 m (absolute) from BOTH endpoints and ≥ `min_spacing`=2.4 m apart → **obstacle count scales with path length** (6 m path ≈ 1-2 gates, 9-14 m ≈ 3-4). Verified: **100% feasible AND 100% blocks-straight** across n=2..5. RC_field switched to it. This is the honest hard task; retraining (geodesic + anchor + curriculum) on it now.

## 2026-07 — Waypoint feasibility bug (same class); gap-waypoints; field 0.80 achieved on solvable task

Feasible slalom training result: on the CORRECTED (100%-solvable) task the policy hit **loco 1.00, single-obstacle ~0.93, field (3-4 blocking obstacles) 0.77-0.87** — up from the ~0.32 ceiling on the impossible task. This is the honest confirmation that obstacle avoidance IS learnable here once the task is feasible; the whole "reactive/geodesic ceiling" story was dominated by task infeasibility.

HARD (obstacles + waypoints) stayed low (~0.1-0.3) — and measurement showed WHY: waypoints were sampled on the centre-line independently of the slalom obstacles (which also straddle the centre-line), so **~45% of waypoints landed INSIDE an inflated obstacle**, forcing an unavoidable collision → HARD capped ~0.35 by waypoint infeasibility, same bug class as the obstacles. Fixed: `sample_obstacles_slalom(..., return_gaps=True)` now returns the clear-gap centre of each gate; `RC_field` places waypoints THERE. Verified 0% of gap-waypoints inside obstacles (was 45%). Killed the run that was training stage D on infeasible waypoints; relaunching a focused fine-tune from the strong field policy on the feasible gap-waypoint distribution to get an honest HARD number.

## 2026-07 — FINAL honest result on the fully-feasible genuine task

After fixing BOTH feasibility bugs (obstacle placement → slalom; waypoints → clear gaps), the fine-tuned policy (results/_obstacle_nav/slalom_field_hard_best.zip) evaluated over 60 episodes/tier under STRICT collision-terminate=1 (any contact = failure). Crucially, collision-free-successes == successes in every tier, i.e. EVERY success is genuinely collision-free (no graze-through):

| Tier | Success |
|---|---|
| loco (no obstacles) | 100% |
| field (3-4 blocking obstacles) | 75% |
| dense (5 obstacles, short path) | 83% |
| HARD (3-4 obstacles + 1-2 waypoints) | 67% |
| HARD-long (obstacles + 2 waypoints, 11-14 m) | 52% |

Trajectory maps (results/_obstacle_nav/slalom_hard_maps.png): successes trace SMOOTH arcs weaving through the gaps — the earlier wobble ("telepie się") is gone (geodesic reward + vw control). Failures are "stop short at a hard gate", never "crash through".

**Honest bottom line:** genuine, verified obstacle avoidance reaches ~75-83% (obstacles only) and ~67% on the combined plural-obstacle+waypoint task, dropping to ~52% on the longest/densest waypoint configs — all with every success collision-free. This replaces the retracted hollow "90%". The remaining gap to 80-90% on HARD is the rover freezing at tight multi-constraint gates (waypoint position + weave gap), a real difficulty, not an artifact. Env fixes (sample_obstacles_slalom + return_gaps, RC_field) are committed to the codebase.

## 2026-07 — Strict-termination fine-tune: fixed corner-cutting; honest numbers (+ selection-bias caution)

The maps showed feasible routes the policy failed by CUTTING CORNERS into obstacles (collision) — a train/eval MISMATCH: training used relaxed collision-terminate=8 (graze = small cost, episode continues), so the policy learned corner-cutting is acceptable, but eval terminates on ANY contact. Fine-tuned with termination ANNEALED to strict =1 (matching eval) + stronger proximity (0.35) + clearance-speed penalty + hit_penalty 5→8.

Clean 60-eps/tier eval of the fine-tuned policy (results/_obstacle_nav/slalom_field_hard_best.zip), strict term=1, every success collision-free:
| Tier | before | after |
|---|---|---|
| loco | 100% | 100% |
| field (3-4 obstacles) | 75% | 83% |
| dense (5 obstacles, short) | 83% | 95% |
| HARD (obstacles + 1-2 waypoints) | 67% | 68% |
| HARD-long (obstacles + 2 wp, 11-14 m) | 52% | 60% |

**Real wins:** corner-cutting COLLISIONS eliminated (maps: no collision markers; failures are now timeouts, not crashes; every success collision-free). field/dense/HARD-long improved. Seeds 8103/8104 flipped FAIL→SUCCESS.

**Honesty caution — selection bias:** the mid-training eval showed HARD 0.85-0.88, but that was the MAX over ~26 noisy 40-episode evals on varying seeds (max-of-noisy-estimates overestimates). The unbiased 60-episode eval gives HARD ~68%. Report 68%, not 0.88. HARD (with waypoints) remains the bottleneck: hitting specific in-gap waypoints in order while weaving is genuinely hard for the reactive policy; closing 68%→85% would need more targeted training and larger eval-based selection (or a smarter waypoint/gate representation), not more optimistic checkpoint-picking.

## 2026-07 — Clean-run experiment (Mac CPU): confirmed HARD ceiling ~0.75; no improvement over warm-start

Per the "single clean run + fixed-eval selection" plan (GPU/MJX deferred — Mac only), ran an uninterrupted fine-tune from the best policy: strict collision-terminate=1 throughout, proximity 0.28, clearance-speed penalty, waypoint-prob ramp 0.35→0.6, and checkpoint selection on a FIXED 100-episode eval (unbiased). Result over 2.4M steps:
- HARD (fixed 100-ep) oscillated 0.69-0.78; field ~0.82-0.86; the only `*saved` was at +150k (HARD 0.75). Later checkpoints never beat it, and loco COLLAPSED to 0.00 at +2.25M (reduced-proximity + waypoint-ramp eroding the open-field skill — the loco-fragility seen before).
- Definitive 60-ep eval of the clean-run best was slightly WORSE than the prior strict-termination checkpoint (field 80 vs 83, dense 77 vs 95). So the clean run did NOT improve the policy.

**Canonical policy = `results/_obstacle_nav/slalom_field_hard_best.zip`** (the strict-termination fine-tune, field_hard2). Honest numbers (every success collision-free; seed variance shown): loco ~100%, field (3-4 obstacles) ~80-85%, dense (5 obstacles short) ~77-95%, HARD (obstacles + 1-2 waypoints) ~68-75%, HARD-long ~60-62%.

**Conclusion:** on Mac CPU (6 envs) the reactive policy is at its ceiling — field ~0.85, HARD ~0.75. More of the same training doesn't break HARD higher and eventually destabilises locomotion. Realistic paths beyond this (all deferred): GPU/MJX for 10-50× throughput → clean 30-50M run; recurrent policy (LSTM) for multi-waypoint sequencing; explicit gate/lookahead obs. Corner-cutting collisions and wobble are fixed; failures are now conservative timeouts.

## 2026-07 — RecurrentPPO (LSTM) tried and shelved on Mac CPU: does not learn in a practical budget

Implemented scripts/train_obstacle_recurrent.py (RecurrentPPO / MlpLstmPolicy) to target the MLP's HARD bottleneck (timeouts/stuck at tight gates, waypoint sequencing) via hidden-state memory. Result: **loco = 0.00 across the full 900k-step bootstrap** (the MLP reached loco 1.00 by ~600k). A focused diagnostic (200k loco-only, obstacle-free, unconditional save + dual eval) confirmed it is GENUINE non-learning, not a recurrent-eval bug: mean distance-to-goal reduction +0.05 m out of ~5.2 m (i.e. the rover barely moves toward the goal), success 0/20.

**Conclusion:** RecurrentPPO from scratch is far more sample-hungry / slower per step than PPO, and on Mac CPU (6 envs) it does not produce a usable policy in a practical budget; iterating its hyperparameters would cost many CPU-hours per attempt. LSTM is a GPU-tier improvement (needs the 10-50× throughput of MJX/CUDA). Shelved for the Mac-only phase. Scripts + the optional `recurrent` extra remain in the repo for when GPU is available.

Cheaper Mac-viable lever that targets the same HARD bottleneck: the env already has an opt-in GEODESIC bent-bearing observation (`geo_heading_obs`), which points the target bearing along the collision-free route instead of straight through obstacles. It was left off to keep the "unknown terrain, lidar-only" premise, but enabling it (reframed as "rover follows a globally-planned route, RL does local control" — realistic for Mars ops) is a fast flip+retrain on the MLP and the likely next step to push HARD past ~0.75.

## 2026-07 — Perception-mode axis (privileged / reactive / SLAM) for CL × perception comparison

Per the "make it realistic — give back the obstacle cheat, discover via SLAM" direction, added a PERCEPTION MODE axis orthogonal to the CL method, so the thesis can compare (CL method × perception):
- **privileged** (default): ground-truth obstacle AABBs in the obs (teacher-level).
- **reactive**: `obstacle_obs_mode="none"` — no ground-truth obstacle info; lidar only (honest mapless).
- **slam**: obstacles discovered online into an `OccupancyMap` (forward-lidar Bresenham ray-marking + footprint dilation, pose from sim odometry — "M" of SLAM, no loop closure); `geo_heading` planned on the DISCOVERED map via `NavField(blocked=...)`, rebuilt every 10 steps. Starts straight, bends as obstacles are sensed.

All modes keep obs dim fixed (drop-in). Env flags: `obstacle_obs_mode`, `geo_heading_source`. Reward's proximity term still uses the true nearest distance (training-time shaping, allowed). `run_scenario.py --perception {privileged,reactive,slam}` applies the override to any scenario's tasks (via `apply_perception` rebuilding the tagged env factories) and writes results under `<scenario>/<method>__<perception>/seed_N`, so `--compare` renders the CL×perception grid. Tests: tests/test_perception_modes.py. Also archived obsolete results (scenario_10 variants, old/, _learnability) to results/archive/. Scenario code left intact (test-coupled; not worth the churn per user).

Note: full SLAM (unknown pose + loop closure) deliberately not implemented — occupancy mapping with sim odometry captures the "discover, don't get told" realism at a fraction of the cost. LSTM remains GPU-tier.
