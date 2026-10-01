"""Utilities and social welfare (pure functions of a state).

Arrays are indexed [consumer, producer]: consumer quantities are row sums,
producer quantities are column sums. The diagonal is masked out of every sum.
"""
import numpy as np

from pimarket.primitives import relevance_matrix, theta


def _off_diagonal(cfg):
    return ~np.eye(cfg.N, dtype=bool)


def theta_infl_matrix(state, cfg):
    # Eq. 4: theta_infl(z | y) = theta(mu_infl(z)) * theta(mu(infl | y)), entry [y, z]
    hop1 = theta(state.mu_infl, cfg)[None, :]       # influencer -> z
    hop2 = theta(state.mu_to_infl, cfg)[:, None]    # y -> influencer
    return hop1 * hop2


def consumer_utility_parts(state, cfg):
    """Eq. 6 split into its three terms (each of shape (N,)):
    via the influencer, direct, and outside community."""
    mask = _off_diagonal(cfg)
    B = relevance_matrix(state.topics, state.interests, cfg)
    via_influencer = cfg.r_p * (theta_infl_matrix(state, cfg) * B * mask).sum(axis=1)
    direct = cfg.r_p * (theta(state.mu, cfg) * B * mask).sum(axis=1)
    outside = cfg.r_0 * theta(state.lam, cfg) * cfg.B_0
    return via_influencer, direct, outside


def consumer_utilities(state, cfg):
    # Eq. 6
    via_influencer, direct, outside = consumer_utility_parts(state, cfg)
    return via_influencer + direct + outside


def influencer_utility(state, cfg):
    # Eq. 7
    mask = _off_diagonal(cfg)
    B = relevance_matrix(state.topics, state.interests, cfg)
    return cfg.r_p * (theta_infl_matrix(state, cfg) * B * mask).sum()


def producer_utilities(state, cfg):
    # Eq. 9: column sums over the consumers who receive z's content
    mask = _off_diagonal(cfg)
    B = relevance_matrix(state.topics, state.interests, cfg)
    reach = theta_infl_matrix(state, cfg) + theta(state.mu, cfg)
    return cfg.r_p * (reach * B * mask).sum(axis=0)


def social_welfare(state, cfg):
    # Eq. 11: sum of consumer utilities. It is an exact potential (Lemma 2):
    # a unilateral change by agent i changes it by exactly i's change in utility.
    return consumer_utilities(state, cfg).sum()
