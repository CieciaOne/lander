"""Tests for the MJX backend (`rover_cl.envs.nav_mjx` + `mjx_vec_env`).

Sanity-level coverage: model loads in MJX, env resets, env steps, autoreset
fires on done, VecEnv contract is satisfied. Heavy training is exercised by
the scenario tests with the CPU backend; we just want to know the JAX path
hasn't bit-rotted.

Marked as MJX tests so a CI without JAX could skip the whole module.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.simplefilter("ignore")

jax = pytest.importorskip("jax")
mjx = pytest.importorskip("mujoco.mjx")

from rover_cl.envs.mjx_vec_env import MjxVecEnv
from rover_cl.envs.nav_mjx import MAX_TARGETS, MjxNavEnv
from rover_cl.envs.terrains import TERRAIN_CATALOG


# ---------------------------------------------------------------------------- model
def test_mjx_accepts_rover_on_every_terrain():
    """Every terrain in the catalog must compile down to a valid mjx.Model.

    This is the strict check: it verifies no collision pair we use is in
    MJX's "not implemented" list. Was the failure mode before we switched
    wheel collision from cylinder to sphere.
    """
    import mujoco
    from rover_cl.envs.terrains import compile_scene

    for name, builder in TERRAIN_CATALOG.items():
        spec = builder(seed=0)
        model, _ = compile_scene(spec)
        # If this raises, the rover XML or the terrain composer produced
        # a geometry pair MJX can't handle.
        mjx.put_model(model)


# ---------------------------------------------------------------------------- env
def test_mjx_env_reset_returns_correct_shape():
    env = MjxNavEnv(terrain="T1_flat", n_envs=2, seed=0, max_steps=50)
    obs, info = env.reset(seed=42)
    assert obs.shape == (2, env.obs_dim)
    assert obs.dtype == np.float32
    # Pose obs sub-fields should be finite at rest.
    assert np.all(np.isfinite(np.asarray(obs)))


def test_mjx_env_step_advances_rover():
    """All-throttle forward should reduce distance-to-target after a few
    steps on a flat empty arena."""
    env = MjxNavEnv(terrain="T1_flat", n_envs=2, seed=0, max_steps=100)
    env.reset(seed=42)

    actions = jax.numpy.zeros((2, 2), dtype=jax.numpy.float32)
    actions = actions.at[:, 0].set(1.0)

    # Step a single time to compile, then capture distance.
    _, _, _, info0 = env.step(actions)
    dist0 = np.asarray(info0["distance_to_target"])
    for _ in range(20):
        _, _, _, info1 = env.step(actions)
    dist1 = np.asarray(info1["distance_to_target"])
    # Per env, the rover must have moved at least somewhat closer.
    assert np.all(dist1 < dist0), f"rover did not advance: dist0={dist0} dist1={dist1}"


def test_mjx_env_max_targets_covers_catalog():
    """Pool sampler should never need to truncate a waypoint chain."""
    from rover_cl.envs.terrains import TERRAIN_CATALOG

    for name, builder in TERRAIN_CATALOG.items():
        spec = builder(seed=0)
        # `+1` for the final goal which is always appended.
        n_targets = len(spec.waypoints) + 1
        # Some random rollouts produce different waypoint counts; sample a
        # few rolls to find the maximum.
        if spec.randomize_on_reset is not None:
            rng = np.random.default_rng(0)
            for _ in range(8):
                r = spec.randomize_on_reset(rng)
                n_targets = max(n_targets, len(r.waypoints) + 1)
        assert n_targets <= MAX_TARGETS, (
            f"terrain {name} produces {n_targets} targets, MAX_TARGETS={MAX_TARGETS}"
        )


# ---------------------------------------------------------------------------- vec env
def test_mjx_vec_env_basic_step():
    vec = MjxVecEnv(terrain="T1_flat", n_envs=2, seed=0, max_steps=80)
    assert vec.num_envs == 2
    assert vec.observation_space.shape == (40,)
    assert vec.action_space.shape == (2,)

    obs = vec.reset()
    assert obs.shape == (2, 40)
    assert obs.dtype == np.float32

    actions = np.zeros((2, 2), dtype=np.float32)
    actions[:, 0] = 1.0
    obs, rew, done, info = vec.step(actions)
    assert obs.shape == (2, 40)
    assert rew.shape == (2,)
    assert done.shape == (2,)
    assert len(info) == 2
    # Required SB3 info keys.
    for inf in info:
        assert "is_success" in inf
        assert "distance_to_target" in inf

    vec.close()


def test_mjx_vec_env_autoresets_on_done():
    """If we crank max_steps low and let the rover stall, episodes should
    end with done=True and the next obs should look like a fresh start."""
    vec = MjxVecEnv(terrain="T1_flat", n_envs=2, seed=0, max_steps=5)
    vec.reset()
    actions = np.zeros((2, 2), dtype=np.float32)
    dones_seen = np.zeros(2, dtype=bool)
    # Step until both envs have hit done at least once.
    for _ in range(40):
        _, _, done, info = vec.step(actions)
        dones_seen |= done
        if dones_seen.all():
            # Episode info dict from monitor should have been populated.
            for i, inf in enumerate(info):
                if done[i]:
                    assert "episode" in inf, "monitor episode dict missing on done"
                    assert "terminal_observation" in inf
            break
    assert dones_seen.all(), "expected both envs to hit done within step budget"
    vec.close()


def test_mjx_vec_env_smoke_train_ppo():
    """End-to-end: SB3 PPO uses MjxVecEnv. 64 timesteps, ~30 s on CPU JAX."""
    from stable_baselines3 import PPO

    vec = MjxVecEnv(terrain="T1_flat", n_envs=2, seed=0, max_steps=50)
    model = PPO(
        "MlpPolicy", vec,
        n_steps=32, batch_size=16, n_epochs=1, verbose=0,
        learning_rate=3e-4, gamma=0.995,
        policy_kwargs={"net_arch": [16, 16]},
    )
    model.learn(total_timesteps=64)
    vec.close()
