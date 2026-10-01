"""Primitives: topic distance, interest p, ability q, attention response theta,
and the relevance matrix B.

Topic space is T = [0, 1] with d(x, y) = |x - y|. The exponential forms of f, g
and theta are numerical choices; the paper only requires the properties stated
in Assumptions 2-3.
"""
import numpy as np


def distance(x, y):
    # d(x, y) = |x - y|; broadcasts over arrays
    return np.abs(np.asarray(x, dtype=float) - np.asarray(y, dtype=float))


def p(x, y, cfg):
    # Eq. 1: p(x | y) = f(d(x, y)), with f(d) = exp(-f_scale * d)
    # interest of a consumer with main interest y in topic x
    return np.exp(-cfg.f_scale * distance(x, y))


def q(x, y, cfg):
    # Eq. 2: q(x | y) = g(d(x, y)), with g(d) = exp(-g_scale * d)
    # ability of a producer with main interest y to write on topic x
    return np.exp(-cfg.g_scale * distance(x, y))


def theta(mu, cfg):
    # theta(mu) = 1 - exp(-k * mu), k = theta_scale; expm1 keeps precision for small mu
    mu = np.asarray(mu, dtype=float)
    return -np.expm1(-cfg.theta_scale * mu)


def theta_deriv(mu, cfg):
    # theta'(mu) = k * exp(-k * mu)
    mu = np.asarray(mu, dtype=float)
    return cfg.theta_scale * np.exp(-cfg.theta_scale * mu)


def theta_inv_deriv(b, cfg):
    # G(b) = (theta')^{-1}(b) = ln(k / b) / k for 0 < b < k, and 0 for b >= k
    k = cfg.theta_scale
    b = np.asarray(b, dtype=float)
    return np.log(k / np.minimum(b, k)) / k


def relevance_matrix(topics, interests, cfg):
    # B[y, z] = B(z | y) = q(x_z | y_z) * p(x_z | y_y), indexed [consumer, producer].
    # The diagonal is computed but masked out of every sum downstream.
    topics = np.asarray(topics, dtype=float)
    interests = np.asarray(interests, dtype=float)
    quality = q(topics, interests, cfg)                      # (N,), per producer z
    interest = p(topics[None, :], interests[:, None], cfg)   # (N, N)
    return interest * quality[None, :]
