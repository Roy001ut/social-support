"""Utility tests. The oracle below is a plain double loop written straight from the
paper's formulas. It uses only math.exp, NOT pimarket.primitives, so it is
independent of the production code and can actually disagree with it."""
import math

import numpy as np
import pytest

from pimarket.config import ModelConfig
from pimarket.state import MarketState, initial_state
from pimarket.utilities import (consumer_utilities, consumer_utility_parts,
                                influencer_utility, producer_utilities,
                                theta_infl_matrix)


# ---------- independent oracle (plain loops) ----------

def o_theta(mu, cfg):
    return 1.0 - math.exp(-cfg.theta_scale * mu)


def o_B(s, cfg, y, z):
    # B(z|y) = q(x_z | y_z) * p(x_z | y_y)
    q = math.exp(-cfg.g_scale * abs(s.topics[z] - s.interests[z]))
    p = math.exp(-cfg.f_scale * abs(s.topics[z] - s.interests[y]))
    return q * p


def o_theta_infl(s, cfg, y, z):
    return o_theta(s.mu_infl[z], cfg) * o_theta(s.mu_to_infl[y], cfg)


def o_consumer_parts(s, cfg):
    via, direct, outside = [], [], []
    for y in range(cfg.N):
        v = d = 0.0
        for z in range(cfg.N):
            if z == y:
                continue
            v += o_theta_infl(s, cfg, y, z) * o_B(s, cfg, y, z)
            d += o_theta(s.mu[y, z], cfg) * o_B(s, cfg, y, z)
        via.append(cfg.r_p * v)
        direct.append(cfg.r_p * d)
        outside.append(cfg.r_0 * o_theta(s.lam[y], cfg) * cfg.B_0)
    return np.array(via), np.array(direct), np.array(outside)


def o_influencer(s, cfg):
    total = 0.0
    for z in range(cfg.N):
        for y in range(cfg.N):
            if y != z:
                total += o_theta_infl(s, cfg, y, z) * o_B(s, cfg, y, z)
    return cfg.r_p * total


def o_producers(s, cfg):
    out = []
    for z in range(cfg.N):
        total = 0.0
        for y in range(cfg.N):
            if y != z:
                total += (o_theta_infl(s, cfg, y, z) + o_theta(s.mu[y, z], cfg)) * o_B(s, cfg, y, z)
        out.append(cfg.r_p * total)
    return np.array(out)


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
TOL = 1e-8   # golden numbers are given to 8 decimals


def test_golden_consumer_parts():
    via, direct, outside = consumer_utility_parts(golden, cfg2)
    assert via == pytest.approx([0.12857568, 0.33151288], abs=TOL)
    assert direct == pytest.approx([0.21041466, 0.23865122], abs=TOL)
    assert outside == pytest.approx([0.23306095, 0.23306095], abs=TOL)


def test_golden_totals():
    assert consumer_utilities(golden, cfg2) == pytest.approx([0.57205129, 0.80322505], abs=TOL)
    assert influencer_utility(golden, cfg2) == pytest.approx(0.46008857, abs=TOL)
    assert producer_utilities(golden, cfg2) == pytest.approx([0.57016410, 0.33899034], abs=TOL)


def test_golden_theta_infl_matrix_entry():
    # [consumer 0, producer 1] = theta(mu_infl(1)) * theta(mu(infl|0)) = theta(4) * theta(0.5)
    M = theta_infl_matrix(golden, cfg2)
    assert M[0, 1] == pytest.approx((1 - math.exp(-4.0)) * (1 - math.exp(-0.5)))


# ---------- vectorized code agrees with the plain loops on random states ----------

def random_state(cfg, seed):
    rng = np.random.default_rng(seed)
    N = cfg.N
    mu = rng.random((N, N))
    np.fill_diagonal(mu, 0.0)
    return MarketState(
        interests=rng.random(N), topics=rng.random(N), mu=mu,
        mu_to_infl=rng.random(N), lam=rng.random(N), mu_infl=rng.random(N),
    )


@pytest.mark.parametrize("seed", range(5))
def test_matches_plain_loops_on_random_states(seed):
    cfg = ModelConfig(N=5)
    s = random_state(cfg, seed)
    via, direct, outside = consumer_utility_parts(s, cfg)
    o_via, o_direct, o_outside = o_consumer_parts(s, cfg)
    assert via == pytest.approx(o_via, abs=1e-12)
    assert direct == pytest.approx(o_direct, abs=1e-12)
    assert outside == pytest.approx(o_outside, abs=1e-12)
    assert influencer_utility(s, cfg) == pytest.approx(o_influencer(s, cfg), abs=1e-12)
    assert producer_utilities(s, cfg) == pytest.approx(o_producers(s, cfg), abs=1e-12)


# ---------- invariants ----------

def test_influencer_utility_is_sum_of_via_influencer_parts():
    s = initial_state(ModelConfig())
    cfg = ModelConfig()
    via, _, _ = consumer_utility_parts(s, cfg)
    assert influencer_utility(s, cfg) == pytest.approx(via.sum())


def test_zero_influencer_rates_kill_every_via_influencer_term():
    cfg = ModelConfig()
    s = initial_state(cfg).with_influencer(np.zeros(cfg.N))
    via, _, _ = consumer_utility_parts(s, cfg)
    assert np.all(via == 0.0)
    assert influencer_utility(s, cfg) == 0.0


def test_utility_upper_bounds_hold():
    cfg = ModelConfig()
    s = initial_state(cfg)
    N = cfg.N
    assert np.all(consumer_utilities(s, cfg) < cfg.r_p * (N - 1) + cfg.r_0 * cfg.B_0)
    assert np.all(producer_utilities(s, cfg) < 2 * cfg.r_p * (N - 1))
    assert influencer_utility(s, cfg) < cfg.r_p * N * (N - 1)
