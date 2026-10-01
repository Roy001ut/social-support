"""Market state (everyone's current strategy) and feasibility checks.

All N x N arrays are indexed [consumer, producer].
"""
from dataclasses import dataclass, replace

import numpy as np


def _frozen_copy(a):
    # read-only float copy, so a state is never modified in place
    a = np.array(a, dtype=float)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class MarketState:
    interests: np.ndarray    # y_i,        (N,)    fixed, in [0, 1]
    topics: np.ndarray       # x_i,        (N,)    producer choice, in [0, 1]
    mu: np.ndarray           # mu(z | y),  (N, N)  mu[y, z], zero diagonal
    mu_to_infl: np.ndarray   # mu(infl|y), (N,)    consumer choice
    lam: np.ndarray          # lambda(y),  (N,)    consumer choice (outside community)
    mu_infl: np.ndarray      # mu_infl(z), (N,)    influencer choice

    def __post_init__(self):
        for name in ("interests", "topics", "mu", "mu_to_infl", "lam", "mu_infl"):
            object.__setattr__(self, name, _frozen_copy(getattr(self, name)))

    # Updates return a new state; the original is never modified.

    def with_topic(self, z, x):
        topics = self.topics.copy()
        topics[z] = x
        return replace(self, topics=topics)

    def with_consumer(self, y, mu_row, mu_to_infl_y, lam_y):
        mu = self.mu.copy()
        mu[y, :] = mu_row
        mu_to_infl = self.mu_to_infl.copy()
        mu_to_infl[y] = mu_to_infl_y
        lam = self.lam.copy()
        lam[y] = lam_y
        return replace(self, mu=mu, mu_to_infl=mu_to_infl, lam=lam)

    def with_influencer(self, mu_infl):
        return replace(self, mu_infl=mu_infl)

    def check_feasible(self, cfg, tol=1e-9):
        """Return a list of violated constraints (empty if feasible)."""
        N = cfg.N
        shapes = {"interests": (N,), "topics": (N,), "mu": (N, N),
                  "mu_to_infl": (N,), "lam": (N,), "mu_infl": (N,)}
        problems = []
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                problems.append(f"{name} has shape {getattr(self, name).shape}, expected {shape}")
        if problems:
            return problems

        if np.any(self.interests < -tol) or np.any(self.interests > 1 + tol):
            problems.append("an interest is outside [0, 1]")
        if np.any(self.topics < -tol) or np.any(self.topics > 1 + tol):
            problems.append("a topic is outside [0, 1]")
        if np.any(np.diag(self.mu) != 0.0):
            problems.append("mu has a nonzero diagonal (no self-following)")
        for name in ("mu", "mu_to_infl", "lam", "mu_infl"):
            if np.any(getattr(self, name) < -tol):
                problems.append(f"{name} has a negative rate")

        # consumer budget: sum_{z != y} mu(z|y) + mu(infl|y) + lambda(y) <= M
        spend = self.mu.sum(axis=1) + self.mu_to_infl + self.lam
        if np.any(spend > cfg.M + tol):
            problems.append("a consumer spends more than M")
        # influencer budget: sum_z mu_infl(z) <= M_infl
        if self.mu_infl.sum() > cfg.M_infl + tol:
            problems.append("the influencer spends more than M_infl")
        return problems


def initial_state(cfg, grid=False):
    """Deterministic starting state (given cfg.seed).

    interests: sorted uniform random in [0, 1] (evenly spaced if grid=True)
    topics:    each producer starts at its own interest
    consumers: attention split evenly over the other N-1 producers, the influencer
               and the outside community (M / (N + 1) each)
    influencer: M_infl / N on every producer
    """
    N = cfg.N
    if grid:
        interests = np.linspace(0.0, 1.0, N)
    else:
        interests = np.sort(np.random.default_rng(cfg.seed).random(N))
    share = cfg.M / (N + 1)
    return MarketState(
        interests=interests,
        topics=interests,
        mu=share * (1.0 - np.eye(N)),
        mu_to_infl=np.full(N, share),
        lam=np.full(N, share),
        mu_infl=np.full(N, cfg.M_infl / N),
    )
