"""Best-response dynamics: agents take turns playing an exact best response to the
current state (Gauss-Seidel). One sweep = every consumer, the influencer and every
producer each move once. There is no damping, so by the potential property
(Lemma 2) welfare cannot drop; this is checked after every update and any
violation is recorded.
"""
from dataclasses import dataclass, field

import numpy as np

from pimarket.best_response import (consumer_best_response, influencer_best_response,
                                    producer_best_response)
from pimarket.state import MarketState
from pimarket.utilities import (consumer_utilities, influencer_utility,
                                producer_utilities, social_welfare)

PHI_DROP_TOLERANCE = 1e-9   # numerical choice


@dataclass(frozen=True)
class SweepRecord:
    sweep: int                     # 1, 2, 3, ...
    state: MarketState             # state after this sweep
    welfare: float                 # Phi after this sweep
    consumer_utilities: np.ndarray
    influencer_utility: float
    producer_utilities: np.ndarray
    state_change: float            # normalised distance moved during this sweep


@dataclass
class RunResult:
    final_state: MarketState
    history: list = field(default_factory=list)       # list of SweepRecord
    converged: bool = False
    phi_violations: list = field(default_factory=list)  # messages: Phi dropped after an update
    solver_failures: list = field(default_factory=list) # messages: a best response reported ok = False


def state_distance(a, b, cfg):
    """Normalised max-change between two states: topics by 1, consumer rates by M,
    influencer rates by M_infl. 0 means identical."""
    return max(
        np.abs(a.topics - b.topics).max() / 1.0,
        np.abs(a.mu - b.mu).max() / cfg.M,
        np.abs(a.mu_to_infl - b.mu_to_infl).max() / cfg.M,
        np.abs(a.lam - b.lam).max() / cfg.M,
        np.abs(a.mu_infl - b.mu_infl).max() / cfg.M_infl,
    )


def _agents(cfg):
    # fixed order: consumers, then the influencer, then producers
    return ([("consumer", y) for y in range(cfg.N)]
            + [("influencer", 0)]
            + [("producer", z) for z in range(cfg.N)])


def _best_response(kind, index, state, cfg):
    if kind == "consumer":
        return consumer_best_response(index, state, cfg)
    if kind == "influencer":
        return influencer_best_response(state, cfg)
    return producer_best_response(index, state, cfg)


def run(state, cfg, order="fixed", max_sweeps=200, tol=1e-9):
    """Run sweeps until the state stops moving (change < tol) or max_sweeps is reached.

    order = "fixed"  : consumers, influencer, producers, the same every sweep
    order = "random" : a fresh random order each sweep, seeded by cfg.seed
    (the order can change WHICH equilibrium is reached)
    """
    if order not in ("fixed", "random"):
        raise ValueError("order must be 'fixed' or 'random'")
    rng = np.random.default_rng(cfg.seed)
    agents = _agents(cfg)
    result = RunResult(final_state=state)

    for sweep in range(1, max_sweeps + 1):
        sweep_start = state
        sequence = [agents[i] for i in rng.permutation(len(agents))] if order == "random" else agents
        for kind, index in sequence:
            phi_before = social_welfare(state, cfg)
            new_state, response = _best_response(kind, index, state, cfg)
            if not response.ok:
                result.solver_failures.append(f"sweep {sweep}, {kind} {index}: {response.message}")
            if social_welfare(new_state, cfg) < phi_before - PHI_DROP_TOLERANCE:
                result.phi_violations.append(f"sweep {sweep}, {kind} {index}: Phi dropped")
            state = new_state

        change = state_distance(sweep_start, state, cfg)
        result.history.append(SweepRecord(
            sweep=sweep, state=state, welfare=float(social_welfare(state, cfg)),
            consumer_utilities=consumer_utilities(state, cfg),
            influencer_utility=float(influencer_utility(state, cfg)),
            producer_utilities=producer_utilities(state, cfg),
            state_change=float(change),
        ))
        if change < tol:
            result.converged = True
            break

    result.final_state = state
    return result
