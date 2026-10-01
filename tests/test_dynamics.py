import numpy as np
import pytest

from pimarket.best_response import (consumer_best_response, influencer_best_response,
                                    producer_best_response)
from pimarket.config import ModelConfig
from pimarket.dynamics import run, state_distance
from pimarket.state import initial_state
from pimarket.utilities import (consumer_utilities, influencer_utility,
                                producer_utilities, social_welfare)

cfg = ModelConfig()


def test_state_distance_normalisation():
    s = initial_state(cfg)
    assert state_distance(s, s, cfg) == 0.0
    # topics are normalised by 1
    assert state_distance(s, s.with_topic(0, s.topics[0] + 0.3), cfg) == pytest.approx(0.3)
    # consumer rates are normalised by M = 3:  0.6 / 3 = 0.2
    moved = s.with_consumer(0, s.mu[0], s.mu_to_infl[0], s.lam[0] + 0.6)
    assert state_distance(s, moved, cfg) == pytest.approx(0.2)
    # influencer rates are normalised by M_infl = 6:  1.2 / 6 = 0.2
    assert state_distance(s, s.with_influencer(s.mu_infl + 1.2), cfg) == pytest.approx(0.2)


@pytest.mark.parametrize("order", ["fixed", "random"])
def test_default_run_converges_cleanly(order):
    result = run(initial_state(cfg), cfg, order=order)
    assert result.converged
    assert result.phi_violations == []
    assert result.solver_failures == []
    assert result.final_state.check_feasible(cfg) == []
    assert result.history[-1].state_change < 1e-9


@pytest.mark.parametrize("order", ["fixed", "random"])
@pytest.mark.parametrize("seed", range(4))
def test_welfare_never_drops(seed, order):
    cfg4 = ModelConfig(N=4, seed=seed)
    s0 = initial_state(cfg4)
    result = run(s0, cfg4, order=order)
    assert result.phi_violations == []                      # checked after EVERY single update
    welfare = [social_welfare(s0, cfg4)] + [h.welfare for h in result.history]
    assert np.all(np.diff(welfare) >= -1e-9)                # and therefore sweep to sweep
    assert welfare[-1] > welfare[0]                         # the run actually improved things


def test_runs_are_reproducible():
    for order in ("fixed", "random"):
        a = run(initial_state(cfg), cfg, order=order)
        b = run(initial_state(cfg), cfg, order=order)
        assert np.array_equal(a.final_state.topics, b.final_state.topics)
        assert len(a.history) == len(b.history)


def test_run_does_not_mutate_the_initial_state():
    s0 = initial_state(cfg)
    topics, mu = s0.topics.copy(), s0.mu.copy()
    run(s0, cfg)
    assert np.array_equal(s0.topics, topics) and np.array_equal(s0.mu, mu)


def test_sweep_cap_is_respected():
    result = run(initial_state(cfg), cfg, max_sweeps=1)
    assert len(result.history) == 1
    assert not result.converged                             # one sweep is not enough here


def test_history_matches_final_state():
    result = run(initial_state(cfg), cfg)
    last = result.history[-1]
    assert last.state is result.final_state
    assert last.welfare == pytest.approx(social_welfare(result.final_state, cfg))


def test_converged_state_passes_a_unilateral_gain_check():
    # "stopped moving" is not the same as "verified Nash": check that no agent can gain by deviating
    final = run(initial_state(cfg), cfg).final_state
    for y in range(cfg.N):
        _, r = consumer_best_response(y, final, cfg)
        assert r.value - consumer_utilities(final, cfg)[y] <= 1e-7
    _, r = influencer_best_response(final, cfg)
    assert r.value - influencer_utility(final, cfg) <= 1e-7
    for z in range(cfg.N):
        _, r = producer_best_response(z, final, cfg)
        assert r.value - producer_utilities(final, cfg)[z] <= 1e-7


def test_invalid_order_is_rejected():
    with pytest.raises(ValueError):
        run(initial_state(cfg), cfg, order="alphabetical")
