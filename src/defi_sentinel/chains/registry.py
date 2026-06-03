"""Chain registry — manages chain adapter instances."""

from __future__ import annotations

import logging
from typing import Any

from defi_sentinel.chains.base import ChainAdapter, ChainConfig

logger = logging.getLogger(__name__)


class ChainRegistry:
    """Registry of active chain adapters.

    Usage:
        registry = ChainRegistry()
        registry.add("ethereum")
        registry.add("arbitrum", rpc_url="https://...")

        eth = registry.get("ethereum")
        await eth.connect()
    """

    def __init__(self):
        self._chains: dict[str, ChainAdapter] = {}

    def add(self, name: str, rpc_url: str | None = None) -> ChainAdapter:
        """Add a chain adapter by name.

        Args:
            name: Chain name (ethereum, arbitrum, base, etc.)
            rpc_url: Optional custom RPC URL override.

        Returns:
            The created chain adapter (not yet connected).
        """
        from defi_sentinel.chains.evm import EVMAdapter, CHAIN_PRESETS

        if name not in CHAIN_PRESETS:
            raise ValueError(f"Unknown chain: {name}. Available: {list(CHAIN_PRESETS.keys())}")

        config = CHAIN_PRESETS[name]
        if rpc_url:
            config = ChainConfig(
                name=config.name,
                chain_id=config.chain_id,
                rpc_url=rpc_url,
                explorer_url=config.explorer_url,
                native_token=config.native_token,
                block_time_ms=config.block_time_ms,
            )

        adapter = EVMAdapter(config)
        self._chains[name] = adapter
        logger.info("Registered chain: %s (id=%d)", name, config.chain_id)
        return adapter

    def get(self, name: str) -> ChainAdapter:
        """Get a chain adapter by name."""
        if name not in self._chains:
            raise KeyError(f"Chain '{name}' not registered. Available: {list(self._chains.keys())}")
        return self._chains[name]

    def list_all(self) -> dict[str, ChainAdapter]:
        """Return all registered chains."""
        return dict(self._chains)

    def list_available(self) -> list[str]:
        """Return names of all available chain presets."""
        from defi_sentinel.chains.evm import CHAIN_PRESETS
        return list(CHAIN_PRESETS.keys())
