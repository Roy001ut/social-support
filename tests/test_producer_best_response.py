import math

import numpy as np
import pytest

from helpers import random_feasible_state
from pimarket.best_response import producer_best_response
from pimarket.config import ModelConfig
from pimarket.state import MarketState
from pimarket.utilities import producer_utilities, social_welfare

cfg5 = ModelConfig(N=5)


# ---------- independent oracle: U_p(z; x) straight from Eq. 9, using only np/math ----------

def oracle_utility(s, cfg, z, x):
    """U_p(z) if producer z used topic x (x scalar or array). Plain loop over consumers."""
    th = lambda mu: 1.0 - math.exp(-cfg.theta_scale * mu)
    x = np.asarray(x, dtype=float)
    total = 0.0
    for y in range(cfg.N):
        if y == z:
            continue
        c = th(s.mu_infl[z]) * th(s.mu_to_infl[y]) + th(s.mu[y, z])
        total = total + c * np.exp(-cfg.f_scale * np.abs(x - s.interests[y]))
    return cfg.r_p * np.exp(-cfg.g_scale * np.abs(x - s.interests[z])) * total


@pytest.mark.parametrize("seed", range(5))
def test_agrees_with_brute_force_grid(seed):
    s = random_feasible_state(cfg5, np.random.default_rng(seed))
    grid = np.linspace(0.0, 1.0, 100_000)
    for z in range(cfg5.N):
        new, result = producer_best_response(z, s, cfg5)
        brute = oracle_utility(s, cfg5, z, grid).max()
        assert result.value >= brute - 1e-9              # never beaten by the fine grid
        assert brute >= result.value - 1e-3              # and not suspiciously far above it
        # the utility engine and the oracle agree at the chosen topic
        assert result.value == pytest.approx(oracle_utility(s, cfg5, z, new.topics[z]), abs=1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_optimum_is_at_a_kink_or_endpoint(seed):
    # With exponential f and g, each piece of U_p between the kinks 0, 1, y_i is convex,
    # so the maximum sits on {0, 1, y_1, ..., y_N}.
    s = random_feasible_state(cfg5, np.random.default_rng(seed))
    kinks = np.concatenate([[0.0, 1.0], s.interests])
    for z in range(cfg5.N):
        _, result = producer_best_response(z, s, cfg5)
        assert result.value == pytest.approx(oracle_utility(s, cfg5, z, kinks).max(), abs=1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_producer_best_response_basic_properties(seed):
    s = random_feasible_state(cfg5, np.random.default_rng(seed))
    topics_before = s.topics.copy()
    phi0 = social_welfare(s, cfg5)
    for z in range(cfg5.N):
        new, result = producer_best_response(z, s, cfg5)
        assert result.ok
        assert new.check_feasible(cfg5) == []                                   # topic stays in [0, 1]
        assert result.value >= producer_utilities(s, cfg5)[z] - 1e-12           # never worse
        d_phi = social_welfare(new, cfg5) - phi0
        assert d_phi == pytest.approx(result.value - producer_utilities(s, cfg5)[z], abs=1e-9)
        assert d_phi >= -1e-12                                                  # Phi never drops
    assert np.array_equal(s.topics, topics_before)                              # input not mutated


def test_idempotent_and_keeps_current_topic_when_no_real_gain():
    s = random_feasible_state(cfg5, np.random.default_rng(11))
    once, _ = producer_best_response(1, s, cfg5)
    twice, _ = producer_best_response(1, once, cfg5)
    assert np.array_equal(once.topics, twice.topics)       # second call moves nothing


# ---------- why a hill climber can get stuck ----------

def test_hill_climber_gets_stuck_but_best_response_does_not():
    # Producer 0 (interest 0.2). Consumer 1 (interest 0.1) attends a little, consumer 2
    # (interest 0.9) attends a lot: two bumps, a small one near 0.1 and a big one at 0.9.
    cfg = ModelConfig(N=3, g_scale=0.5)
    s = MarketState(
        interests=[0.2, 0.1, 0.9], topics=[0.2, 0.1, 0.9],
        mu=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [3.0, 0.0, 0.0]],
        mu_to_infl=[0.0, 0.0, 0.0], lam=[0.0, 0.0, 0.0], mu_infl=[0.0, 0.0, 0.0],
    )
    assert s.check_feasible(cfg) == []

    # naive hill climber from the current topic: step to a better neighbour until none exists
    x, h = 0.2, 1e-3
    for _ in range(2000):
        moves = [c for c in (x - h, x + h) if 0.0 <= c <= 1.0
                 and oracle_utility(s, cfg, 0, c) > oracle_utility(s, cfg, 0, x)]
        if not moves:
            break
        x = max(moves, key=lambda c: oracle_utility(s, cfg, 0, c))
    stuck_value = float(oracle_utility(s, cfg, 0, x))
    assert x == pytest.approx(0.1, abs=2e-3)               # stuck on the small bump
    assert stuck_value == pytest.approx(0.45628, abs=1e-4)

    new, result = producer_best_response(0, s, cfg)
    assert new.topics[0] == pytest.approx(0.9, abs=1e-6)   # the global best: the big bump
    assert result.value == pytest.approx(0.69476, abs=1e-4)
    assert result.value > stuck_value + 0.2
