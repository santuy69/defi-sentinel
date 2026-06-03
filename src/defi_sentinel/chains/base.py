"""Abstract chain adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ChainConfig:
    """Chain configuration."""
    name: str
    chain_id: int
    rpc_url: str
    explorer_url: str
    native_token: str
    block_time_ms: int


class ChainAdapter(ABC):
    """Abstract base class for chain adapters.

    Implement this interface to add support for a new blockchain.
    """

    def __init__(self, config: ChainConfig):
        self.config = config
        self._connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to chain RPC."""
        ...

    @abstractmethod
    async def get_block(self, block_id: int | str = "latest") -> dict[str, Any]:
        """Fetch a block by number or tag."""
        ...

    @abstractmethod
    async def get_transaction(self, tx_hash: str) -> dict[str, Any]:
        """Fetch transaction details."""
        ...

    @abstractmethod
    async def get_logs(
        self,
        address: str | None = None,
        topics: list[str] | None = None,
        from_block: int | str = "latest",
        to_block: int | str = "latest",
    ) -> list[dict[str, Any]]:
        """Fetch event logs."""
        ...

    @abstractmethod
    async def simulate_transaction(self, tx: dict[str, Any]) -> dict[str, Any]:
        """Simulate a transaction without broadcasting (eth_call)."""
        ...

    @abstractmethod
    async def get_balance(self, address: str) -> int:
        """Get native token balance in wei."""
        ...

    @abstractmethod
    async def get_token_balance(self, token_address: str, wallet: str) -> int:
        """Get ERC-20 token balance."""
        ...

    @abstractmethod
    async def estimate_gas(self, tx: dict[str, Any]) -> int:
        """Estimate gas for a transaction."""
        ...

    @abstractmethod
    async def broadcast_transaction(self, signed_tx: str) -> str:
        """Broadcast a signed transaction, return tx hash."""
        ...

    @property
    def is_connected(self) -> bool:
        return self._connected
