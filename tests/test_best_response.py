import math

import numpy as np
import pytest

from helpers import random_feasible_state
from pimarket.best_response import (consumer_best_response, consumer_weights,
                                    influencer_best_response, influencer_weights)
from pimarket.config import ModelConfig
from pimarket.state import MarketState
from pimarket.utilities import (consumer_utilities, influencer_utility,
                                social_welfare)

# ---------- golden N = 2 example ----------

cfg2 = ModelConfig(N=2)
golden = MarketState(
    interests=[0.2, 0.4],
    topics=[0.3, 0.5],
    mu=[[0.0, 1.0], [0.5, 0.0]],
    mu_to_infl=[0.5, 1.0],
    lam=[1.5, 1.5],
    mu_infl=[2.0, 4.0],
)


def test_golden_consumer_weights():
    # [direct_0, direct_1, influencer, outside] for consumer 0
    B01 = math.exp(-1.1)                              # B[0, 1] = q(0.5|0.4) * p(0.5|0.2)
    expected = [0.0, B01, (1 - math.exp(-4.0)) * B01, 0.3]
    assert consumer_weights(0, golden, cfg2) == pytest.approx(expected, abs=1e-9)


def test_golden_consumer_best_response():
    new, result = consumer_best_response(0, golden, cfg2)
    assert result.ok
    assert new.mu[0] == pytest.approx([0.0, 1.04082], abs=1e-5)
    assert new.mu_to_infl[0] == pytest.approx(1.02233, abs=1e-5)
    assert new.lam[0] == pytest.approx(0.93685, abs=1e-5)
    assert result.value == pytest.approx(0.60697, abs=1e-5)
    assert consumer_utilities(golden, cfg2)[0] == pytest.approx(0.57205, abs=1e-5)  # before


def test_golden_influencer_best_response():
    assert influencer_weights(golden, cfg2) == pytest.approx([0.38340, 0.13097], abs=1e-5)
    new, result = influencer_best_response(golden, cfg2)
    assert result.ok
    assert new.mu_infl == pytest.approx([3.53704, 2.46296], abs=1e-5)
    assert result.value == pytest.approx(0.49206, abs=1e-5)
    assert influencer_utility(golden, cfg2) == pytest.approx(0.46009, abs=1e-5)     # before


def test_assumption_4_uniform_split_when_nobody_follows_the_influencer():
    nobody = MarketState(
        interests=[0.2, 0.4], topics=[0.3, 0.5],
        mu=[[0.0, 1.0], [0.5, 0.0]], mu_to_infl=[0.0, 0.0],
        lam=[1.5, 1.5], mu_infl=[2.0, 4.0],
    )
    new, result = influencer_best_response(nobody, cfg2)
    assert result.ok and "assumption 4" in result.message
    assert new.mu_infl == pytest.approx([3.0, 3.0])        # M_infl / N = 6 / 2


def test_best_responses_do_not_mutate_the_input_state():
    before = golden.mu.copy(), golden.mu_infl.copy()
    consumer_best_response(0, golden, cfg2)
    influencer_best_response(golden, cfg2)
    assert np.array_equal(golden.mu, before[0])
    assert np.array_equal(golden.mu_infl, before[1])


# ---------- basic properties on random feasible states ----------

cfg5 = ModelConfig(N=5)


@pytest.mark.parametrize("seed", range(5))
def test_consumer_best_response_basic_properties(seed):
    s = random_feasible_state(cfg5, np.random.default_rng(seed))
    for y in range(cfg5.N):
        new, result = consumer_best_response(y, s, cfg5)
        assert result.ok
        assert new.check_feasible(cfg5) == []
        assert new.mu[y, y] == 0.0                                           # no self-following
        spend = new.mu[y].sum() + new.mu_to_infl[y] + new.lam[y]
        assert spend == pytest.approx(cfg5.M, abs=1e-9)                      # budget fully used
        assert result.value >= consumer_utilities(s, cfg5)[y] - 1e-12        # never worse


@pytest.mark.parametrize("seed", range(5))
def test_influencer_best_response_basic_properties(seed):
    s = random_feasible_state(cfg5, np.random.default_rng(seed))
    new, result = influencer_best_response(s, cfg5)
    assert result.ok
    assert new.check_feasible(cfg5) == []
    assert new.mu_infl.sum() == pytest.approx(cfg5.M_infl, abs=1e-9)
    assert result.value >= influencer_utility(s, cfg5) - 1e-12


# ---------- no tiny feasible shift of rate between two options improves the utility ----------

DELTA = 1e-4


@pytest.mark.parametrize("seed", range(5))
def test_no_profitable_shift_for_consumer(seed):
    s = random_feasible_state(cfg5, np.random.default_rng(seed))
    y = seed % cfg5.N
    new, result = consumer_best_response(y, s, cfg5)
    u = np.concatenate([new.mu[y], [new.mu_to_infl[y], new.lam[y]]])         # N + 2 slots
    for i in range(len(u)):
        if u[i] < 10 * DELTA:
            continue
        for j in range(len(u)):
            if j == i or j == y:                                             # slot y is "myself": not allowed
                continue
            v = u.copy()
            v[i] -= DELTA
            v[j] += DELTA
            moved = new.with_consumer(y, v[:cfg5.N], v[cfg5.N], v[cfg5.N + 1])
            assert consumer_utilities(moved, cfg5)[y] <= result.value + 1e-10


@pytest.mark.parametrize("seed", range(5))
def test_no_profitable_shift_for_influencer(seed):
    s = random_feasible_state(cfg5, np.random.default_rng(seed))
    new, result = influencer_best_response(s, cfg5)
    u = new.mu_infl
    for i in range(cfg5.N):
        if u[i] < 10 * DELTA:
            continue
        for j in range(cfg5.N):
            if j == i:
                continue
            v = u.copy()
            v[i] -= DELTA
            v[j] += DELTA
            assert influencer_utility(new.with_influencer(v), cfg5) <= result.value + 1e-10


# ---------- the locality identity again, now on best-response moves ----------

@pytest.mark.parametrize("seed", range(5))
def test_locality_identity_on_best_response_moves(seed):
    s = random_feasible_state(cfg5, np.random.default_rng(seed))
    phi0 = social_welfare(s, cfg5)
    for y in range(cfg5.N):
        new, result = consumer_best_response(y, s, cfg5)
        d_phi = social_welfare(new, cfg5) - phi0
        d_u = result.value - consumer_utilities(s, cfg5)[y]
        assert d_phi == pytest.approx(d_u, abs=1e-9)
        assert d_phi >= -1e-12                                               # Phi never drops
    new, result = influencer_best_response(s, cfg5)
    d_phi = social_welfare(new, cfg5) - phi0
    assert d_phi == pytest.approx(result.value - influencer_utility(s, cfg5), abs=1e-9)
    assert d_phi >= -1e-12


def test_consumer_best_response_is_idempotent():
    # a best response depends only on the OTHERS' strategies, so repeating it changes nothing
    s = random_feasible_state(cfg5, np.random.default_rng(7))
    once, _ = consumer_best_response(2, s, cfg5)
    twice, _ = consumer_best_response(2, once, cfg5)
    assert twice.mu == pytest.approx(once.mu, abs=1e-9)
    assert twice.lam == pytest.approx(once.lam, abs=1e-9)
