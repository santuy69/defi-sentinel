"""EVM chain adapter — supports all EVM-compatible chains."""

from __future__ import annotations

import logging
from typing import Any

from defi_sentinel.chains.base import ChainAdapter, ChainConfig

logger = logging.getLogger(__name__)

# Pre-configured chain configs
CHAIN_PRESETS: dict[str, ChainConfig] = {
    "ethereum": ChainConfig(
        name="ethereum", chain_id=1,
        rpc_url="https://eth.llamarpc.com",
        explorer_url="https://etherscan.io",
        native_token="ETH", block_time_ms=12000,
    ),
    "arbitrum": ChainConfig(
        name="arbitrum", chain_id=42161,
        rpc_url="https://arb1.arbitrum.io/rpc",
        explorer_url="https://arbiscan.io",
        native_token="ETH", block_time_ms=250,
    ),
    "base": ChainConfig(
        name="base", chain_id=8453,
        rpc_url="https://mainnet.base.org",
        explorer_url="https://basescan.org",
        native_token="ETH", block_time_ms=2000,
    ),
    "polygon": ChainConfig(
        name="polygon", chain_id=137,
        rpc_url="https://polygon-rpc.com",
        explorer_url="https://polygonscan.com",
        native_token="POL", block_time_ms=2000,
    ),
    "optimism": ChainConfig(
        name="optimism", chain_id=10,
        rpc_url="https://mainnet.optimism.io",
        explorer_url="https://optimistic.etherscan.io",
        native_token="ETH", block_time_ms=2000,
    ),
    "bsc": ChainConfig(
        name="bsc", chain_id=56,
        rpc_url="https://bsc-dataseed.binance.org",
        explorer_url="https://bscscan.com",
        native_token="BNB", block_time_ms=3000,
    ),
    "sepolia": ChainConfig(
        name="sepolia", chain_id=11155111,
        rpc_url="https://sepolia.drpc.org",
        explorer_url="https://sepolia.etherscan.io",
        native_token="ETH", block_time_ms=12000,
    ),
}

# Minimal ERC-20 ABI for balanceOf
ERC20_BALANCE_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
]


class EVMAdapter(ChainAdapter):
    """EVM-compatible chain adapter using web3.py."""

    def __init__(self, config: ChainConfig):
        super().__init__(config)
        self._w3 = None

    async def connect(self) -> bool:
        """Connect to EVM chain via RPC."""
        try:
            from web3 import Web3

            self._w3 = Web3(Web3.HTTPProvider(self.config.rpc_url))
            connected = self._w3.is_connected()

            if connected:
                chain_id = self._w3.eth.chain_id
                assert chain_id == self.config.chain_id, (
                    f"Chain ID mismatch: expected {self.config.chain_id}, got {chain_id}"
                )
                self._connected = True
                logger.info("Connected to %s (chain %d)", self.config.name, chain_id)
            else:
                logger.error("Failed to connect to %s RPC: %s", self.config.name, self.config.rpc_url)

            return connected
        except Exception as e:
            logger.error("Connection error on %s: %s", self.config.name, e)
            return False

    async def get_block(self, block_id: int | str = "latest") -> dict[str, Any]:
        """Fetch block details."""
        self._ensure_connected()
        block = self._w3.eth.get_block(block_id, full_transactions=True)
        return {
            "number": block.number,
            "hash": block.hash.hex(),
            "timestamp": block.timestamp,
            "transactions": len(block.transactions),
            "gas_used": block.gasUsed,
            "gas_limit": block.gasLimit,
            "base_fee": getattr(block, "baseFeePerGas", None),
        }

    async def get_transaction(self, tx_hash: str) -> dict[str, Any]:
        """Fetch transaction details."""
        self._ensure_connected()
        tx = self._w3.eth.get_transaction(tx_hash)
        receipt = self._w3.eth.get_transaction_receipt(tx_hash)
        return {
            "hash": tx.hash.hex(),
            "from": tx["from"],
            "to": tx.to,
            "value": self._w3.from_wei(tx.value, "ether"),
            "gas": tx.gas,
            "gas_price": self._w3.from_wei(tx.gasPrice, "gwei"),
            "status": receipt.status,
            "block": tx.blockNumber,
        }

    async def get_logs(self, address=None, topics=None, from_block="latest", to_block="latest"):
        """Fetch event logs."""
        self._ensure_connected()
        filter_params = {"fromBlock": from_block, "toBlock": to_block}
        if address:
            filter_params["address"] = address
        if topics:
            filter_params["topics"] = topics
        return [dict(log) for log in self._w3.eth.get_logs(filter_params)]

    async def simulate_transaction(self, tx: dict[str, Any]) -> dict[str, Any]:
        """Simulate via eth_call (does not broadcast)."""
        self._ensure_connected()
        try:
            result = self._w3.eth.call(tx)
            return {"success": True, "result": result.hex()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_balance(self, address: str) -> int:
        """Get native token balance in wei."""
        self._ensure_connected()
        return self._w3.eth.get_balance(address)

    async def get_token_balance(self, token_address: str, wallet: str) -> int:
        """Get ERC-20 token balance."""
        self._ensure_connected()
        contract = self._w3.eth.contract(
            address=self._w3.to_checksum_address(token_address),
            abi=ERC20_BALANCE_ABI,
        )
        return contract.functions.balanceOf(self._w3.to_checksum_address(wallet)).call()

    async def estimate_gas(self, tx: dict[str, Any]) -> int:
        """Estimate gas for a transaction."""
        self._ensure_connected()
        return self._w3.eth.estimate_gas(tx)

    async def broadcast_transaction(self, signed_tx: str) -> str:
        """Broadcast a signed transaction."""
        self._ensure_connected()
        tx_hash = self._w3.eth.send_raw_transaction(signed_tx)
        return tx_hash.hex()

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(f"Not connected to {self.config.name}. Call connect() first.")
