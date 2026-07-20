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
        ppo_kwargs=FAST_PPO,
    )
    env_a, env_b = _env(), _env()
    cl.train_on(env_a, TINY_STEPS, task_id="T1")
    assert cl.last_penalty_steps_run == 0

    cl.train_on(env_b, TINY_STEPS, task_id="T2")
    # With the hook mechanism, last_penalty_steps_run counts backward passes
    # that carried the penalty gradient — one per PPO minibatch update.
    assert cl.last_penalty_steps_run > 0
    assert cl.last_penalty_value >= 0.0

    env_a.close()
    env_b.close()


@pytest.mark.slow
def test_penalty_hook_adds_analytic_gradient():
    """The gradient hook must add exactly lam * F * (p - theta*) to grads.

    We train one task to populate fisher/theta_star, then compute gradients
    of a fixed loss twice — with and without hooks installed — and check the
    difference against the analytic penalty gradient. This verifies the
    penalty is integrated into the backward pass (not a separate pull-back)
    and that lam=0 makes it an exact no-op.
    """
    cl = EwcCL(lam=123.0, fisher_sample_size=64, ppo_kwargs=FAST_PPO)
    env = _env()
    cl.train_on(env, TINY_STEPS, task_id="A")
    env.close()

    policy = cl.model.policy
    obs_t = th.randn(8, 3)   # Pendulum obs dim = 3
    act_t = th.randn(8, 1)   # Pendulum act dim = 1

    def grads_of_fixed_loss() -> dict[str, th.Tensor]:
        policy.zero_grad()
        _v, log_prob, _e = policy.evaluate_actions(obs_t, act_t)
        (-log_prob.mean()).backward()
        return {
            n: p.grad.detach().clone()
            for n, p in policy.named_parameters()
            if p.grad is not None
        }

    grads_plain = grads_of_fixed_loss()
    handles = cl._install_penalty_hooks(["A"])
    try:
        grads_hooked = grads_of_fixed_loss()
    finally:
        for h in handles:
            h.remove()
    policy.zero_grad()

    for n, g_hooked in grads_hooked.items():
        if n not in cl.fisher["A"]:
            continue
        expected = cl.lam * cl.fisher["A"][n] * (
            dict(policy.named_parameters())[n].detach() - cl.theta_star["A"][n]
        )
        actual = g_hooked - grads_plain[n]
        np.testing.assert_allclose(
            actual.cpu().numpy(), expected.cpu().numpy(),
            rtol=1e-4, atol=1e-6,
            err_msg=f"penalty gradient mismatch on {n}",
        )

    # lam=0 → hooks must be an exact no-op.
    cl.lam = 0.0
    handles = cl._install_penalty_hooks(["A"])
    try:
        grads_lam0 = grads_of_fixed_loss()
    finally:
        for h in handles:
            h.remove()
    policy.zero_grad()
    for n, g in grads_lam0.items():
        np.testing.assert_allclose(
            g.cpu().numpy(), grads_plain[n].cpu().numpy(),
            rtol=1e-6, atol=1e-8,
            err_msg=f"lam=0 hook changed gradient on {n}",
        )


@pytest.mark.slow
def test_save_load_roundtrips_fisher(tmp_path: Path):
    cl = EwcCL(
        lam=500.0,
        fisher_sample_size=64,
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
    one has ``lam=0`` (penalty gradient identically zero — effectively Naive
    on B) and the other has the full penalty active. Both have the same
    post-A model + theta_star, so the only difference is the penalty.
    """
    seed = 0

    th.manual_seed(seed)
    np.random.seed(seed)
    ewc_off = EwcCL(
        lam=0.0,
        fisher_sample_size=128,
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

    # Structural guarantee: penalty-carrying backward passes did run in the
    # "on" variant. (The lam=0 variant also installs hooks — past task exists
    # — but its penalty gradient is identically zero.)
    assert ewc_on.last_penalty_steps_run > 0

    # The penalty must pull params closer to theta_star_A than no penalty.
    assert dist_on < dist_off, (
        f"EWC penalty drift ({dist_on:.4f}) is not less than no-penalty drift "
        f"({dist_off:.4f}); ratio = {dist_on / max(dist_off, 1e-12):.3f}"
    )
