import numpy as np
import pytest

from pimarket.config import ModelConfig
from helpers import (random_consumer_strategy, random_feasible_state,
                     random_influencer_strategy)
from pimarket.state import MarketState
from pimarket.utilities import (consumer_utilities, consumer_utility_parts,
                                influencer_utility, producer_utilities,
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


def test_golden_welfare():
    assert social_welfare(golden, cfg2) == pytest.approx(1.37527635, abs=1e-8)


def test_welfare_is_sum_of_the_three_consumer_parts():
    via, direct, outside = consumer_utility_parts(golden, cfg2)
    assert social_welfare(golden, cfg2) == pytest.approx(via.sum() + direct.sum() + outside.sum())


def test_producer_utilities_sum_to_influencer_plus_direct():
    # sum_z U_p(z) = U_infl + sum of all direct terms
    _, direct, _ = consumer_utility_parts(golden, cfg2)
    total = producer_utilities(golden, cfg2).sum()
    assert total == pytest.approx(influencer_utility(golden, cfg2) + direct.sum())
    assert total == pytest.approx(0.90915444, abs=1e-8)


def test_welfare_upper_bound():
    cfg = ModelConfig()
    from pimarket.state import initial_state
    assert social_welfare(initial_state(cfg), cfg) < cfg.N * (cfg.r_p * (cfg.N - 1) + cfg.r_0 * cfg.B_0)


# ---------- the locality test: Delta Phi == Delta U_i for one-agent deviations ----------

@pytest.mark.parametrize("seed", range(30))
def test_locality_identity_for_every_agent(seed):
    cfg = ModelConfig(N=5)
    rng = np.random.default_rng(seed)
    s = random_feasible_state(cfg, rng)
    assert s.check_feasible(cfg) == []
    phi0 = social_welfare(s, cfg)
    uc0, ui0, up0 = consumer_utilities(s, cfg), influencer_utility(s, cfg), producer_utilities(s, cfg)

    moved = 0   # guard against a vacuous pass: the deviations must actually change things

    # every consumer deviates (one at a time)
    for y in range(cfg.N):
        s2 = s.with_consumer(y, *random_consumer_strategy(cfg, rng, y))
        assert s2.check_feasible(cfg) == []
        d_phi = social_welfare(s2, cfg) - phi0
        d_u = consumer_utilities(s2, cfg)[y] - uc0[y]
        assert d_phi == pytest.approx(d_u, abs=1e-9)
        moved += abs(d_phi) > 1e-6

    # the influencer deviates
    s2 = s.with_influencer(random_influencer_strategy(cfg, rng))
    d_phi = social_welfare(s2, cfg) - phi0
    assert d_phi == pytest.approx(influencer_utility(s2, cfg) - ui0, abs=1e-9)
    moved += abs(d_phi) > 1e-6

    # every producer deviates (picks a new topic)
    for z in range(cfg.N):
        s2 = s.with_topic(z, rng.random())
        d_phi = social_welfare(s2, cfg) - phi0
        d_u = producer_utilities(s2, cfg)[z] - up0[z]
        assert d_phi == pytest.approx(d_u, abs=1e-9)
        moved += abs(d_phi) > 1e-6

    assert moved >= 8   # out of 2N + 1 = 11 deviations


def test_locality_identity_is_specific_to_the_deviating_agent():
    # Sanity: the identity is not automatic. When producer 0 changes topic,
    # Delta Phi equals producer 0's utility change, NOT consumer 0's.
    cfg = ModelConfig(N=5)
    rng = np.random.default_rng(0)
    s = random_feasible_state(cfg, rng)
    phi0 = social_welfare(s, cfg)
    s2 = s.with_topic(0, 0.99)
    d_phi = social_welfare(s2, cfg) - phi0
    right = producer_utilities(s2, cfg)[0] - producer_utilities(s, cfg)[0]
    wrong = consumer_utilities(s2, cfg)[0] - consumer_utilities(s, cfg)[0]  # the wrong agent
    assert d_phi == pytest.approx(right, abs=1e-9)
    assert d_phi != pytest.approx(wrong, abs=1e-9)
