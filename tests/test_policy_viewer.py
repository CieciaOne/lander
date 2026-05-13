"""Tests for the policy-replay business logic in scripts/visualize_rover.py.

These tests intentionally do NOT open a MuJoCo viewer (no display in CI).
Instead they exercise the small helpers the script factors out:

    - build_env_from_terrain_name(name)
    - policy_step(env, policy, obs)

and verify that an SB3 PPO checkpoint trained on a tiny `RoverNavEnv` can be
loaded back and used to drive the env via those helpers.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO

from rover_cl.cl import NaiveCL
from rover_cl.envs.nav import RoverNavEnv


pytestmark = pytest.mark.slow


# --------------------------------------------------------------------------- helpers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIZ_PATH = PROJECT_ROOT / "scripts" / "visualize_rover.py"


def _load_viz_module():
    """Import scripts/visualize_rover.py as a module without executing main()."""
    if "visualize_rover" in sys.modules:
        return sys.modules["visualize_rover"]
    spec = importlib.util.spec_from_file_location("visualize_rover", str(VIZ_PATH))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["visualize_rover"] = mod
    spec.loader.exec_module(mod)
    return mod


# Tiny PPO so training completes well under a second.
FAST_PPO = {
    "n_steps": 64,
    "batch_size": 32,
    "policy_kwargs": {"net_arch": [16, 16]},
}


@pytest.fixture(scope="module")
def trained_policy_path(tmp_path_factory) -> Path:
    """Train a tiny PPO on RoverNavEnv(T1_flat, max_steps=50) for 256 steps."""
    env = RoverNavEnv(terrain="T1_flat", max_steps=50)
    cl = NaiveCL(ppo_kwargs=FAST_PPO)
    cl.train_on(env, total_timesteps=256, task_id="T1")
    path = tmp_path_factory.mktemp("ckpt") / "policy.zip"
    cl.save(path)
    env.close()
    return path


# --------------------------------------------------------------------------- tests


def test_load_policy_and_predict(trained_policy_path: Path):
    """Saved SB3 checkpoint round-trips and predict() returns env-shaped actions."""
    env = RoverNavEnv(terrain="T1_flat", max_steps=50)
    policy = PPO.load(str(trained_policy_path))
    obs, _ = env.reset()
    action, _ = policy.predict(obs, deterministic=True)
    action = np.asarray(action)
    # Shape matches the env's action_space; values are castable to its dtype.
    assert action.shape == env.action_space.shape
    cast = action.astype(env.action_space.dtype, copy=False)
    # Either the raw action is contained, or after clipping to action_space bounds it is.
    clipped = np.clip(cast, env.action_space.low, env.action_space.high)
    assert env.action_space.contains(clipped)
    env.close()


def test_policy_step_returns_valid_obs(trained_policy_path: Path):
    """policy_step yields env-shaped observations and finite rewards."""
    viz = _load_viz_module()
    env = RoverNavEnv(terrain="T1_flat", max_steps=50)
    policy = PPO.load(str(trained_policy_path))
    obs, _ = env.reset()

    for _ in range(5):
        next_obs, terminated, truncated, info = viz.policy_step(env, policy, obs)
        assert next_obs.shape == env.observation_space.shape
        assert np.all(np.isfinite(next_obs))
        assert np.isfinite(info["_reward"])
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
    env.close()


@pytest.mark.parametrize("name", ["T1_flat", "T2_corridor"])
def test_build_env_from_terrain_name(name: str):
    viz = _load_viz_module()
    env = viz.build_env_from_terrain_name(name)
    assert isinstance(env, RoverNavEnv)
    assert env.terrain.name == name
    env.close()


def test_build_env_from_terrain_name_unknown_raises():
    viz = _load_viz_module()
    with pytest.raises(KeyError):
        viz.build_env_from_terrain_name("not_a_terrain")
