import pytest
from dataclasses import FrozenInstanceError

from pimarket.config import ModelConfig


def test_defaults_are_valid():
    cfg = ModelConfig()
    assert cfg.N == 6 and cfg.B_0 == 0.3


def test_rejects_bad_values():
    with pytest.raises(ValueError):
        ModelConfig(N=1)
    with pytest.raises(ValueError):
        ModelConfig(B_0=1.5)


def test_config_is_frozen():
    cfg = ModelConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.N = 10
