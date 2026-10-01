from dataclasses import replace

import numpy as np
import pytest

from pimarket.config import ModelConfig
from pimarket.state import MarketState, initial_state

cfg = ModelConfig()


def test_initial_state_is_feasible():
    assert initial_state(cfg).check_feasible(cfg) == []
    assert initial_state(cfg, grid=True).check_feasible(cfg) == []


def test_initial_state_values():
    s = initial_state(cfg)
    assert s.mu.shape == (6, 6)
    assert np.all(np.diag(s.mu) == 0.0)
    assert s.mu[0, 1] == pytest.approx(3.0 / 7)           # M / (N + 1)
    assert s.mu_infl == pytest.approx(np.full(6, 1.0))    # M_infl / N
    assert np.all(np.diff(s.interests) >= 0)              # sorted
    assert np.array_equal(s.topics, s.interests)          # x_i = y_i


def test_same_seed_gives_identical_state_and_other_seed_differs():
    a, b = initial_state(cfg), initial_state(cfg)
    assert np.array_equal(a.interests, b.interests)
    c = initial_state(ModelConfig(seed=1))
    assert not np.array_equal(a.interests, c.interests)


def test_overspent_consumer_flagged():
    s = initial_state(cfg)
    lam = s.lam.copy()
    lam[2] += 1.0
    assert "a consumer spends more than M" in replace(s, lam=lam).check_feasible(cfg)


def test_overspent_influencer_flagged():
    s = initial_state(cfg)
    bad = s.with_influencer(s.mu_infl + 0.5)
    assert "the influencer spends more than M_infl" in bad.check_feasible(cfg)


def test_nonzero_diagonal_flagged():
    s = initial_state(cfg)
    mu = s.mu.copy()
    mu[1, 1] = 0.1
    assert any("diagonal" in m for m in replace(s, mu=mu).check_feasible(cfg))


def test_negative_rate_flagged():
    s = initial_state(cfg)
    lam = s.lam.copy()
    lam[0] = -0.1
    assert "lam has a negative rate" in replace(s, lam=lam).check_feasible(cfg)


def test_topic_outside_unit_interval_flagged():
    s = initial_state(cfg)
    assert "a topic is outside [0, 1]" in s.with_topic(0, 1.5).check_feasible(cfg)


def test_updates_return_new_state_and_leave_old_untouched():
    s = initial_state(cfg)
    old_topic = s.topics[2]
    s2 = s.with_topic(2, 0.9)
    assert s2.topics[2] == 0.9
    assert s.topics[2] == old_topic


def test_state_arrays_are_read_only():
    s = initial_state(cfg)
    with pytest.raises(ValueError):
        s.topics[0] = 0.5


def test_golden_n2_state_is_feasible():
    cfg2 = ModelConfig(N=2)
    s = MarketState(
        interests=[0.2, 0.4],
        topics=[0.3, 0.5],
        mu=[[0.0, 1.0], [0.5, 0.0]],
        mu_to_infl=[0.5, 1.0],
        lam=[1.5, 1.5],
        mu_infl=[2.0, 4.0],
    )
    assert s.check_feasible(cfg2) == []
