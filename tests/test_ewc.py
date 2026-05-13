"""Tests for EwcCL — Elastic Weight Consolidation on a shared PPO policy."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch as th

from rover_cl.cl import EwcCL, NaiveCL, make_cl


# Tiny PPO so 'training' completes in well under a second per task.
# `seed=0` inside PPO keeps these tests deterministic even when other tests in
# the suite advance the global torch/numpy RNG before this file runs.
FAST_PPO = {
    "n_steps": 64,
    "batch_size": 32,
    "policy_kwargs": {"net_arch": [16, 16]},
    "seed": 0,
}
TINY_STEPS = 128


def _env() -> gym.Env:
    env = gym.make("Pendulum-v1")
    env.reset(seed=0)
    return env


pytestmark = pytest.mark.fast


@pytest.mark.fast
def test_make_cl_factory_ewc():
    cl = make_cl("ewc", ppo_kwargs=FAST_PPO)
    assert isinstance(cl, EwcCL)
    assert cl.lam == pytest.approx(1000.0)
    cl2 = make_cl("ewc", lam=100.0, ppo_kwargs=FAST_PPO)
    assert isinstance(cl2, EwcCL)
    assert cl2.lam == pytest.approx(100.0)


@pytest.mark.slow
def test_fisher_computed_after_train_on():
    cl = EwcCL(
        lam=1000.0,
        fisher_sample_size=64,
        penalty_steps=4,
        ppo_kwargs=FAST_PPO,
    )
    env = _env()
    cl.train_on(env, TINY_STEPS, task_id="T1")
    env.close()

    assert "T1" in cl.fisher
    fisher_t1 = cl.fisher["T1"]
    assert len(fisher_t1) > 0
    for name, t in fisher_t1.items():
        assert th.isfinite(t).all(), f"non-finite Fisher entry for {name}"
        assert (t >= 0).all(), f"negative Fisher entry for {name}"


@pytest.mark.slow
def test_no_penalty_on_first_task():
    cl = EwcCL(
        lam=1000.0,
        fisher_sample_size=64,
        penalty_steps=5,
        ppo_kwargs=FAST_PPO,
    )
    env_a, env_b = _env(), _env()
    cl.train_on(env_a, TINY_STEPS, task_id="T1")
    assert cl.last_penalty_steps_run == 0

    cl.train_on(env_b, TINY_STEPS, task_id="T2")
    assert cl.last_penalty_steps_run > 0
    assert cl.last_penalty_steps_run == 5

    env_a.close()
    env_b.close()


@pytest.mark.slow
def test_lambda_zero_disables_protection():
    """With lam=0 the penalty-step loop still runs but produces zero gradient.

    Verification: we manually invoke _apply_ewc_penalty after clearing Adam's
    internal momentum buffers, then compare params. With lam=0 the loss is
    identically zero, its gradient is zero, and with no carried momentum Adam
    must leave the params unchanged.
    """
    cl = EwcCL(
        lam=0.0,
        fisher_sample_size=64,
        penalty_steps=10,
        ppo_kwargs=FAST_PPO,
    )
    env_a, env_b = _env(), _env()
    cl.train_on(env_a, TINY_STEPS, task_id="A")
    cl.train_on(env_b, TINY_STEPS, task_id="B")
    assert cl.last_penalty_steps_run > 0

    # Clear any optimizer momentum from PPO's prior updates so we can isolate
    # the effect of the EWC penalty step itself. We rebuild a fresh Adam at
    # the same LR so its internal state is empty.
    old_opt = cl.model.policy.optimizer
    lr = old_opt.param_groups[0]["lr"]
    cl.model.policy.optimizer = th.optim.Adam(
        cl.model.policy.parameters(), lr=lr
    )

    before = {
        n: p.detach().clone()
        for n, p in cl.model.policy.named_parameters()
        if p.requires_grad
    }
    cl._apply_ewc_penalty([tid for tid in cl.fisher if tid != "B"])
    after = {
        n: p.detach().clone()
        for n, p in cl.model.policy.named_parameters()
        if p.requires_grad
    }
    max_abs_diff = max(
        float((after[n] - before[n]).abs().max().item()) for n in before
    )
    assert max_abs_diff < 1e-8, (
        f"lam=0 should be a no-op for penalty steps, but params drifted by "
        f"{max_abs_diff}"
    )

    env_a.close()
    env_b.close()


@pytest.mark.slow
def test_save_load_roundtrips_fisher(tmp_path: Path):
    cl = EwcCL(
        lam=500.0,
        fisher_sample_size=64,
        penalty_steps=3,
        ppo_kwargs=FAST_PPO,
    )
    env = _env()
    cl.train_on(env, TINY_STEPS, task_id="T1")
    env.close()

    path = tmp_path / "ewc.zip"
    cl.save(path)
    restored = EwcCL.load(path)

    assert "T1" in restored.fisher
    assert set(restored.fisher["T1"].keys()) == set(cl.fisher["T1"].keys())
    for name, t in cl.fisher["T1"].items():
        rt = restored.fisher["T1"][name]
        assert rt.shape == t.shape
        np.testing.assert_allclose(
            rt.detach().cpu().numpy(),
            t.detach().cpu().numpy(),
            atol=1e-5,
        )
    # theta_star round-trips too.
    assert "T1" in restored.theta_star
    for name, t in cl.theta_star["T1"].items():
        rt = restored.theta_star["T1"][name]
        np.testing.assert_allclose(
            rt.detach().cpu().numpy(),
            t.detach().cpu().numpy(),
            atol=1e-5,
        )
    assert restored.lam == pytest.approx(cl.lam)


def _param_diff_norm(
    params: dict[str, th.Tensor],
    snapshot: dict[str, th.Tensor],
    fisher: dict[str, th.Tensor] | None = None,
) -> float:
    """L2 distance between current and snapshot params.

    When ``fisher`` is provided, only params with at least one nonzero Fisher
    entry contribute — EWC only protects parameters that have meaningful
    Fisher diagonal mass, so a fair structural comparison ignores the rest
    (notably the value head, whose log_prob has zero gradient wrt value-net
    weights and therefore zero Fisher).
    """
    total = 0.0
    for n, p in params.items():
        if n not in snapshot:
            continue
        if fisher is not None:
            f = fisher.get(n)
            if f is None or float(f.max().item()) == 0.0:
                continue
        total += float(((p - snapshot[n]) ** 2).sum().item())
    return float(np.sqrt(total))


@pytest.mark.slow
def test_ewc_pulls_params_toward_theta_star():
    """EWC pulls post-B params closer to theta_star[A] than a no-penalty run.

    Direct EWC-vs-NaiveCL comparison would conflate two effects: (1) different
    random PPO trajectories during task A leave the two methods at different
    starting points for task B, and (2) the EWC penalty itself. To isolate (2)
    we compare two EwcCL runs with identical hyperparameters and seeds, except
    one has ``penalty_steps=0`` (effectively Naive on B) and the other has the
    full penalty active. Both have the same post-A model + theta_star, so the
    only difference is whether the penalty fires.

    We use ``lam=1e3`` (the default) — much larger values saturate Adam's
    normalization and cause the penalty to oscillate around theta_star instead
    of converging, weakening the structural signal.
    """
    seed = 0

    th.manual_seed(seed)
    np.random.seed(seed)
    ewc_off = EwcCL(
        lam=1e3,
        fisher_sample_size=128,
        penalty_steps=0,
        ppo_kwargs=FAST_PPO,
    )
    env_a, env_b = _env(), _env()
    ewc_off.train_on(env_a, TINY_STEPS, task_id="A")
    theta_a_off = {n: t.clone() for n, t in ewc_off.theta_star["A"].items()}
    ewc_off.train_on(env_b, TINY_STEPS, task_id="B")
    params_off = {
        n: p.detach().clone()
        for n, p in ewc_off.model.policy.named_parameters()
        if p.requires_grad
    }
    dist_off = _param_diff_norm(
        params_off, theta_a_off, fisher=ewc_off.fisher["A"]
    )
    env_a.close()
    env_b.close()

    th.manual_seed(seed)
    np.random.seed(seed)
    ewc_on = EwcCL(
        lam=1e3,
        fisher_sample_size=128,
        penalty_steps=50,
        ppo_kwargs=FAST_PPO,
    )
    env_a, env_b = _env(), _env()
    ewc_on.train_on(env_a, TINY_STEPS, task_id="A")
    theta_a_on = {n: t.clone() for n, t in ewc_on.theta_star["A"].items()}
    ewc_on.train_on(env_b, TINY_STEPS, task_id="B")
    params_on = {
        n: p.detach().clone()
        for n, p in ewc_on.model.policy.named_parameters()
        if p.requires_grad
    }
    dist_on = _param_diff_norm(
        params_on, theta_a_on, fisher=ewc_on.fisher["A"]
    )
    env_a.close()
    env_b.close()

    # Structural guarantee: penalty did run.
    assert ewc_on.last_penalty_steps_run == 50
    assert ewc_off.last_penalty_steps_run == 0

    # The penalty must pull params closer to theta_star_A than no penalty.
    assert dist_on < dist_off, (
        f"EWC penalty drift ({dist_on:.4f}) is not less than no-penalty drift "
        f"({dist_off:.4f}); ratio = {dist_on / max(dist_off, 1e-12):.3f}"
    )
