"""Fast smoke tests for the CL methods. No project deps outside src/rover_cl/cl/."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from rover_cl.cl import NaiveCL, ReplayCL, make_cl


# Tiny PPO so 'training' completes in well under a second per task.
FAST_PPO = {
    "n_steps": 64,
    "batch_size": 32,
    "policy_kwargs": {"net_arch": [16, 16]},
}
TINY_STEPS = 128


def _env() -> gym.Env:
    return gym.make("Pendulum-v1")


@pytest.mark.fast
def test_make_cl_factory():
    assert isinstance(make_cl("naive", ppo_kwargs=FAST_PPO), NaiveCL)
    assert isinstance(make_cl("replay", ppo_kwargs=FAST_PPO), ReplayCL)
    with pytest.raises(ValueError):
        make_cl("does-not-exist")


@pytest.mark.slow
def test_naive_trains_on_two_tasks():
    cl = NaiveCL(ppo_kwargs=FAST_PPO)
    env_a, env_b = _env(), _env()
    cl.train_on(env_a, TINY_STEPS, task_id="T1")
    cl.train_on(env_b, TINY_STEPS, task_id="T2")

    for env in (env_a, env_b):
        obs, _ = env.reset(seed=0)
        action, _ = cl.predict(obs, deterministic=True)
        assert env.action_space.contains(action.astype(env.action_space.dtype))

    assert cl.seen_task_ids == ["T1", "T2"]
    env_a.close()
    env_b.close()


@pytest.mark.slow
def test_replay_buffer_grows():
    cl = ReplayCL(
        buffer_size_per_task=32,
        rehearsal_batch_size=8,
        rehearsal_steps=4,
        ppo_kwargs=FAST_PPO,
    )
    env_a, env_b = _env(), _env()
    cl.train_on(env_a, TINY_STEPS, task_id="T1")
    cl.train_on(env_b, TINY_STEPS, task_id="T2")

    assert "T1" in cl.buffers and "T2" in cl.buffers
    assert len(cl.buffers["T1"]) > 0
    assert len(cl.buffers["T2"]) > 0
    # Rehearsal must have actually executed during the second task (past_ids was non-empty).
    assert cl.last_rehearsal_steps_run == 4
    env_a.close()
    env_b.close()


@pytest.mark.slow
def test_save_load_roundtrip(tmp_path: Path):
    cl = ReplayCL(
        buffer_size_per_task=16,
        rehearsal_batch_size=4,
        rehearsal_steps=2,
        ppo_kwargs=FAST_PPO,
    )
    env = _env()
    cl.train_on(env, TINY_STEPS, task_id="T1")

    obs, _ = env.reset(seed=123)
    before, _ = cl.predict(obs, deterministic=True)

    path = tmp_path / "model.zip"
    cl.save(path)
    restored = ReplayCL.load(path)
    after, _ = restored.predict(obs, deterministic=True)

    np.testing.assert_allclose(before, after, atol=1e-5)
    # Buffer should round-trip too.
    assert "T1" in restored.buffers
    assert len(restored.buffers["T1"]) == len(cl.buffers["T1"])
    env.close()


@pytest.mark.slow
def test_replay_rehearses_without_error():
    """Structural check: rehearsal_steps actually ran on past-task data.

    With tiny training budgets the 'replay retains task-A behavior better than naive'
    signal is too noisy to assert reliably; we instead verify the rehearsal loop
    executed the requested number of gradient steps on past-task transitions.
    """
    cl = ReplayCL(
        buffer_size_per_task=32,
        rehearsal_batch_size=8,
        rehearsal_steps=20,
        ppo_kwargs=FAST_PPO,
    )
    env_a, env_b = _env(), _env()
    cl.train_on(env_a, TINY_STEPS, task_id="A")
    assert cl.last_rehearsal_steps_run == 0  # no past tasks yet

    cl.train_on(env_b, TINY_STEPS, task_id="B")
    assert cl.last_rehearsal_steps_run == 20

    # Compare against Naive: both should still produce valid actions on env_a after env_b.
    naive = NaiveCL(ppo_kwargs=FAST_PPO)
    naive.train_on(_env(), TINY_STEPS, task_id="A")
    naive.train_on(_env(), TINY_STEPS, task_id="B")

    obs, _ = env_a.reset(seed=7)
    a_replay, _ = cl.predict(obs, deterministic=True)
    a_naive, _ = naive.predict(obs, deterministic=True)
    assert env_a.action_space.contains(a_replay.astype(env_a.action_space.dtype))
    assert env_a.action_space.contains(a_naive.astype(env_a.action_space.dtype))

    env_a.close()
    env_b.close()
