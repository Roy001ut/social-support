import math

import numpy as np
import pytest
from scipy.optimize import minimize

from pimarket.best_response import waterfill
from pimarket.config import ModelConfig

cfg = ModelConfig()   # theta_scale k = 1


def objective(w, u, k=1.0):
    # sum_k w_k * theta(u_k), plain loop with math.exp (independent of production code)
    return sum(wk * (1.0 - math.exp(-k * uk)) for wk, uk in zip(w, u))


def test_golden_example():
    u, ok, message = waterfill([1.0, 0.5, 0.25], 2.0, cfg)
    assert ok and message == "ok"
    assert u == pytest.approx([1.34657359, 0.65342641, 0.0], abs=1e-8)
    assert u.sum() == pytest.approx(2.0, abs=1e-12)
    assert objective([1.0, 0.5, 0.25], u) == pytest.approx(0.97973990, abs=1e-8)


def test_budget_fully_spent_whenever_any_weight_positive():
    rng = np.random.default_rng(0)
    for _ in range(50):
        w = rng.random(7) + 0.01
        budget = rng.uniform(0.1, 6.0)
        u, ok, _ = waterfill(w, budget, cfg)
        assert ok
        assert u.sum() == pytest.approx(budget, abs=1e-12)
        assert np.all(u >= 0)


def test_zero_weight_options_get_exactly_zero():
    u, ok, _ = waterfill([0.8, 0.0, 0.5, 0.0], 3.0, cfg)
    assert ok
    assert u[1] == 0.0 and u[3] == 0.0
    assert u.sum() == pytest.approx(3.0, abs=1e-12)


def test_all_zero_weights_is_degenerate():
    u, ok, message = waterfill([0.0, 0.0, 0.0], 3.0, cfg)
    assert not ok
    assert message.startswith("degenerate")
    assert np.all(u == 0.0)


def test_marginal_benefit_is_equalized_on_funded_options():
    w = np.array([1.0, 0.5, 0.25, 0.9])
    u, _, _ = waterfill(w, 2.0, cfg)
    funded = u > 0
    marginal = w * np.exp(-u)                       # w_k * theta'(u_k), k = 1
    level = marginal[funded]
    assert level == pytest.approx(np.full(level.shape, level[0]), abs=1e-9)
    # unfunded options are too weak: even their first unit (w_k * theta'(0) = w_k) is below the level
    assert np.all(w[~funded] <= level[0] + 1e-9)


def test_beats_random_feasible_allocations():
    rng = np.random.default_rng(3)
    w = np.array([1.0, 0.5, 0.25, 0.7, 0.1])
    budget = 2.0
    u, _, _ = waterfill(w, budget, cfg)
    best = objective(w, u)
    for _ in range(200):
        trial = rng.dirichlet(np.ones(5)) * budget * rng.uniform(0.0, 1.0)
        assert objective(w, trial) <= best + 1e-12


def test_agrees_with_scipy_slsqp():
    w = np.array([1.0, 0.5, 0.25, 0.7, 0.1])
    budget = 2.0
    u, _, _ = waterfill(w, budget, cfg)
    result = minimize(
        lambda x: -objective(w, x),
        x0=np.full(5, budget / 5),
        bounds=[(0.0, budget)] * 5,
        constraints=[{"type": "ineq", "fun": lambda x: budget - x.sum()}],
        method="SLSQP",
    )
    assert objective(w, u) >= -result.fun - 1e-7


def test_works_with_another_theta_scale():
    cfg2 = ModelConfig(theta_scale=2.0)
    w = np.array([1.0, 0.5, 0.25])
    u, ok, _ = waterfill(w, 1.5, cfg2)
    assert ok and u.sum() == pytest.approx(1.5, abs=1e-12)
    best = objective(w, u, k=2.0)
    rng = np.random.default_rng(5)
    for _ in range(100):
        trial = rng.dirichlet(np.ones(3)) * 1.5
        assert objective(w, trial, k=2.0) <= best + 1e-12
