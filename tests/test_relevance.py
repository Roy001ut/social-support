import numpy as np
import pytest

from pimarket.config import ModelConfig
from pimarket.primitives import q, relevance_matrix

cfg = ModelConfig()

# golden N = 2 example
interests = np.array([0.2, 0.4])
topics = np.array([0.3, 0.5])
GOLDEN_B = np.array([[0.60653066, 0.33287108],
                     [0.60653066, 0.60653066]])


def test_golden_matrix():
    B = relevance_matrix(topics, interests, cfg)
    assert B.shape == (2, 2)
    assert B == pytest.approx(GOLDEN_B, abs=1e-8)


def test_not_symmetric_so_a_transpose_bug_would_fail():
    B = relevance_matrix(topics, interests, cfg)
    assert B[0, 1] != pytest.approx(B[1, 0])
    assert B[0, 1] == pytest.approx(0.33287108)   # consumer 0, producer 1
    assert B[1, 0] == pytest.approx(0.60653066)   # consumer 1, producer 0


def test_entries_match_the_formula_by_plain_loop():
    # independent oracle: double loop straight from B(z|y) = q(x_z|y_z) * p(x_z|y_y)
    rng = np.random.default_rng(1)
    y = rng.random(5)
    x = rng.random(5)
    B = relevance_matrix(x, y, cfg)
    for consumer in range(5):
        for producer in range(5):
            quality = np.exp(-cfg.g_scale * abs(x[producer] - y[producer]))
            interest = np.exp(-cfg.f_scale * abs(x[producer] - y[consumer]))
            assert B[consumer, producer] == pytest.approx(quality * interest)


def test_bounds():
    rng = np.random.default_rng(2)
    y = rng.random(6)
    x = rng.random(6)
    B = relevance_matrix(x, y, cfg)
    assert np.all(B >= 0.0) and np.all(B <= 1.0)
    # B[y, z] <= q(x_z | y_z): p is at most 1
    assert np.all(B <= q(x, y, cfg)[None, :] + 1e-15)
