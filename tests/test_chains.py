"""Tests for chain registry."""

import pytest

from defi_sentinel.chains.registry import ChainRegistry


def test_register_ethereum():
    registry = ChainRegistry()
    adapter = registry.add("ethereum")
    assert adapter.config.chain_id == 1
    assert adapter.config.name == "ethereum"


def test_list_available():
    registry = ChainRegistry()
    available = registry.list_available()
    assert "ethereum" in available
    assert "arbitrum" in available
    assert "sepolia" in available


def test_unknown_chain_raises():
    registry = ChainRegistry()
    with pytest.raises(ValueError, match="Unknown chain"):
        registry.add("bitcoin")


def test_get_unregistered_raises():
    registry = ChainRegistry()
    with pytest.raises(KeyError):
        registry.get("ethereum")
