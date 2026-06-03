"""Transaction simulator — test before broadcast."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class TxSimulator:
    """Simulate transactions via eth_call before broadcasting.

    This is the primary safety rail — every proposed transaction
    must pass simulation before the agent considers executing it.

    Usage:
        sim = TxSimulator()
        result = sim.simulate({
            "to": "0x...",
            "data": "0x...",
            "value": 0,
            "from": "0x...",
        })
        if result["success"]:
            # Safe to broadcast
    """

    def simulate(self, tx: dict[str, Any], chain: str = "ethereum") -> dict[str, Any]:
        """Simulate a transaction without broadcasting.

        Args:
            tx: Transaction dict with to, data, value, from.
            chain: Target chain name.

        Returns:
            Dict with 'success' (bool), 'gas_estimate' (int),
            'return_data' (str), and 'error' (str if failed).
        """
        logger.info("Simulating tx on %s: to=%s", chain, tx.get("to", "?"))

        try:
            from defi_sentinel.chains.registry import ChainRegistry

            registry = ChainRegistry()
            adapter = registry.add(chain)

            # Use eth_call for simulation
            result = adapter._w3.eth.call({
                "to": tx.get("to"),
                "data": tx.get("data", "0x"),
                "value": tx.get("value", 0),
                "from": tx.get("from", "0x0000000000000000000000000000000000000000"),
            })

            gas_estimate = adapter._w3.eth.estimate_gas({
                "to": tx.get("to"),
                "data": tx.get("data", "0x"),
                "value": tx.get("value", 0),
            })

            return {
                "success": True,
                "gas_estimate": gas_estimate,
                "return_data": result.hex(),
                "error": None,
            }

        except Exception as e:
            logger.warning("Simulation failed: %s", e)
            return {
                "success": False,
                "gas_estimate": 0,
                "return_data": None,
                "error": str(e),
            }

    def simulate_batch(self, txs: list[dict], chain: str = "ethereum") -> list[dict]:
        """Simulate multiple transactions in order.

        Returns list of results, one per transaction.
        Stops at first failure if stop_on_failure=True.
        """
        results = []
        for i, tx in enumerate(txs):
            result = self.simulate(tx, chain)
            results.append(result)
            if not result["success"]:
                logger.warning("Batch simulation failed at tx %d/%d", i + 1, len(txs))
                break
        return results
