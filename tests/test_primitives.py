import numpy as np
import pytest

from pimarket.config import ModelConfig
from pimarket.primitives import distance, p, q

cfg = ModelConfig()  # f_scale=3, g_scale=2


def test_distance_is_absolute_difference():
    assert distance(0.2, 0.5) == pytest.approx(0.3)
    assert distance(0.5, 0.2) == pytest.approx(0.3)  # symmetric


def test_p_and_q_are_one_at_own_interest():
    assert p(0.7, 0.7, cfg) == 1.0
    assert q(0.7, 0.7, cfg) == 1.0


def test_p_and_q_decrease_with_distance():
    xs = np.array([0.0, 0.1, 0.5, 1.0])  # distances from y = 0
    assert np.all(np.diff(p(xs, 0.0, cfg)) < 0)
    assert np.all(np.diff(q(xs, 0.0, cfg)) < 0)


def test_golden_values():
    assert p(0.5, 0.2, cfg) == pytest.approx(0.40656966)  # exp(-0.9)
    assert q(0.5, 0.4, cfg) == pytest.approx(0.81873075)  # exp(-0.2)


def test_broadcasting_gives_consumer_by_topic_matrix():
    interests = np.array([0.2, 0.4])          # shape (2,)
    topics = np.array([0.3, 0.5])             # shape (2,)
    out = p(topics[None, :], interests[:, None], cfg)
    assert out.shape == (2, 2)
    # entry [row, col] uses interest[row] and topic[col]
    assert out[0, 1] == pytest.approx(np.exp(-3.0 * abs(0.5 - 0.2)))
