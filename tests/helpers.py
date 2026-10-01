"""Shared test helpers: random feasible states and random one-agent strategies."""
import numpy as np

from pimarket.state import MarketState


def random_consumer_strategy(cfg, rng, y):
    # N+2 slots: N producers (own slot forced to 0), influencer, outside; spend <= M
    w = rng.random(cfg.N + 2)
    w[y] = 0.0
    w *= cfg.M * rng.uniform(0.3, 1.0) / w.sum()
    return w[:cfg.N], w[cfg.N], w[cfg.N + 1]


def random_influencer_strategy(cfg, rng):
    w = rng.random(cfg.N)
    return w * cfg.M_infl * rng.uniform(0.3, 1.0) / w.sum()


def random_feasible_state(cfg, rng):
    N = cfg.N
    mu = np.zeros((N, N))
    mu_to_infl = np.zeros(N)
    lam = np.zeros(N)
    for y in range(N):
        mu[y], mu_to_infl[y], lam[y] = random_consumer_strategy(cfg, rng, y)
    return MarketState(
        interests=rng.random(N), topics=rng.random(N),
        mu=mu, mu_to_infl=mu_to_infl, lam=lam,
        mu_infl=random_influencer_strategy(cfg, rng),
    )
