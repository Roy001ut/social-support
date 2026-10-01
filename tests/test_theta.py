import numpy as np
import pytest

from pimarket.config import ModelConfig
from pimarket.primitives import theta, theta_deriv, theta_inv_deriv

cfg = ModelConfig()  # theta_scale = 1


def test_theta_at_zero_and_one():
    assert theta(0.0, cfg) == 0.0
    assert theta(1.0, cfg) == pytest.approx(0.63212056)  # 1 - exp(-1)


def test_theta_is_strictly_increasing_and_below_one():
    mu = np.linspace(0.0, 10.0, 101)
    vals = theta(mu, cfg)
    assert np.all(np.diff(vals) > 0)
    assert vals.max() < 1.0


def test_theta_is_strictly_concave():
    mu = np.linspace(0.0, 5.0, 51)
    v = theta(mu, cfg)
    second_difference = v[2:] - 2 * v[1:-1] + v[:-2]
    assert np.all(second_difference < 0)


def test_theta_inv_deriv_undoes_theta_deriv():
    mu = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    assert theta_inv_deriv(theta_deriv(mu, cfg), cfg) == pytest.approx(mu)
    # also with a different scale k = 2
    cfg2 = ModelConfig(theta_scale=2.0)
    assert theta_inv_deriv(theta_deriv(mu, cfg2), cfg2) == pytest.approx(mu)


def test_theta_inv_deriv_is_zero_when_b_at_least_k():
    assert theta_inv_deriv(1.0, cfg) == 0.0   # b = k
    assert theta_inv_deriv(2.0, cfg) == 0.0   # b > k
