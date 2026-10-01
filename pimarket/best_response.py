"""Best responses: water-filling (paper Eqs. 18-19), the consumer, influencer and
producer best responses. They never mutate a state; they return a new one."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from pimarket.primitives import p, q, relevance_matrix, theta, theta_inv_deriv
from pimarket.utilities import consumer_utilities, influencer_utility, producer_utilities


def waterfill(w, budget, cfg):
    """Solve max sum_k w_k * theta(u_k) subject to sum_k u_k <= budget, u >= 0.

    Returns (u, ok, message). Since theta is strictly concave, every funded option
    has the same marginal benefit w_k * theta'(u_k) = alpha, i.e. u_k = G(alpha / w_k),
    with alpha chosen so that the budget is spent (found by bisection).
    """
    w = np.asarray(w, dtype=float)
    u = np.zeros_like(w)
    positive = w > 0
    if not positive.any():
        return u, False, "degenerate: every weight is zero"
    if budget <= 0:
        return u, False, "budget must be > 0"

    def allocate(alpha):
        out = np.zeros_like(w)
        out[positive] = theta_inv_deriv(alpha / w[positive], cfg)
        return out

    # total spend falls as alpha rises; it is 0 at alpha_hi
    alpha_hi = cfg.theta_scale * w.max()
    alpha_lo = alpha_hi / 2
    for _ in range(2000):
        if allocate(alpha_lo).sum() >= budget:
            break
        alpha_lo /= 2
    else:
        return u, False, "could not bracket alpha"

    for _ in range(200):                  # bisection in log space
        alpha_mid = np.sqrt(alpha_lo * alpha_hi)
        if allocate(alpha_mid).sum() >= budget:
            alpha_lo = alpha_mid
        else:
            alpha_hi = alpha_mid

    u = allocate(alpha_lo)
    u = u * (budget / u.sum())            # remove the last rounding slack
    return u, True, "ok"


@dataclass(frozen=True)
class BestResponse:
    value: float      # the agent's utility after the move
    ok: bool          # whether the solver succeeded
    message: str


def consumer_weights(y, state, cfg):
    """Weights w of consumer y's problem U_c(y) = sum_k w_k * theta(u_k), ordered
    [direct_0, ..., direct_{N-1}, influencer, outside]:
      direct z   : r_p * B[y, z]   (0 for z = y)
      influencer : r_p * sum_{z != y} theta(mu_infl[z]) * B[y, z]
      outside    : r_0 * B_0
    """
    B = relevance_matrix(state.topics, state.interests, cfg)
    row = B[y].copy()
    row[y] = 0.0
    direct = cfg.r_p * row
    influencer = cfg.r_p * (theta(state.mu_infl, cfg) * row).sum()
    outside = cfg.r_0 * cfg.B_0
    return np.concatenate([direct, [influencer, outside]])


def consumer_best_response(y, state, cfg):
    w = consumer_weights(y, state, cfg)
    u, ok, message = waterfill(w, cfg.M, cfg)
    if not ok:
        return state, BestResponse(consumer_utilities(state, cfg)[y], False, message)
    new_state = state.with_consumer(y, u[:cfg.N], u[cfg.N], u[cfg.N + 1])
    return new_state, BestResponse(consumer_utilities(new_state, cfg)[y], True, message)


def influencer_weights(state, cfg):
    """Weights of the influencer's problem U_infl = sum_z w_z * theta(mu_infl[z]):
    w_z = r_p * sum_{y != z} theta(mu(infl | y)) * B[y, z]."""
    B = relevance_matrix(state.topics, state.interests, cfg)
    mask = ~np.eye(cfg.N, dtype=bool)
    followers = theta(state.mu_to_infl, cfg)[:, None]
    return cfg.r_p * (followers * B * mask).sum(axis=0)


def influencer_best_response(state, cfg):
    if state.mu_to_infl.sum() == 0.0:
        # Assumption 4: nobody follows the influencer, so use the uniform split
        new_state = state.with_influencer(np.full(cfg.N, cfg.M_infl / cfg.N))
        return new_state, BestResponse(influencer_utility(new_state, cfg), True,
                                       "assumption 4: nobody follows the influencer, uniform split")
    w = influencer_weights(state, cfg)
    u, ok, message = waterfill(w, cfg.M_infl, cfg)
    if not ok:
        return state, BestResponse(influencer_utility(state, cfg), False, message)
    new_state = state.with_influencer(u)
    return new_state, BestResponse(influencer_utility(new_state, cfg), True, message)


GRID_POINTS = 2001      # numerical choice: resolution of the global search over topics
GAIN_TOLERANCE = 1e-12  # numerical choice: keep the current topic unless the gain exceeds this


def producer_objective(z, state, cfg):
    """Return f(x) = U_p(z) as a function of producer z's topic x:
    f(x) = r_p * q(x | y_z) * sum_{y != z} c_y * p(x | y_y),
    where c_y = theta(mu_infl[z]) * theta(mu(infl|y)) + theta(mu(z|y)) is how much
    consumer y attends to z."""
    c = theta(state.mu_infl[z], cfg) * theta(state.mu_to_infl, cfg) + theta(state.mu[:, z], cfg)
    c[z] = 0.0

    def f(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        quality = q(x, state.interests[z], cfg)
        interest = p(x[:, None], state.interests[None, :], cfg)
        return cfg.r_p * quality * (interest * c[None, :]).sum(axis=1)

    return f


def producer_best_response(z, state, cfg):
    # f is not concave in general, so search a grid (plus 0, 1 and the interests,
    # where f has kinks) and then refine locally around the best point.
    f = producer_objective(z, state, cfg)
    candidates = np.unique(np.concatenate([
        np.linspace(0.0, 1.0, GRID_POINTS), [0.0, 1.0], state.interests,
    ]))
    values = f(candidates)
    i = int(np.argmax(values))
    best_x, best_value = candidates[i], values[i]

    for lo, hi in ((candidates[max(i - 1, 0)], candidates[i]),
                   (candidates[i], candidates[min(i + 1, len(candidates) - 1)])):
        if hi > lo:
            result = minimize_scalar(lambda x: -float(f(x)[0]), bounds=(lo, hi),
                                     method="bounded", options={"xatol": 1e-12})
            if -result.fun > best_value:
                best_x, best_value = float(result.x), -result.fun

    if best_value - float(f(state.topics[z])[0]) <= GAIN_TOLERANCE:
        new_state = state
    else:
        new_state = state.with_topic(z, best_x)
    return new_state, BestResponse(producer_utilities(new_state, cfg)[z], True, "ok")
