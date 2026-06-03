"""Core agent loop — Reason → Plan → Execute → Observe."""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any

from defi_sentinel.inference.engine import LLMEngine
from defi_sentinel.chains.registry import ChainRegistry
from defi_sentinel.safety.simulator import TxSimulator
from defi_sentinel.safety.limits import SpendingLimits

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """You are DeFi Sentinel, an autonomous agent that monitors
on-chain activity and executes DeFi strategies. You reason about blockchain data,
assess risks, and propose actions. Always simulate before executing. Never exceed
spending limits. If uncertain, explain your reasoning and recommend manual review."""


@dataclass
class AgentConfig:
    """Agent configuration."""
    model: str = "TheBloke/Llama-2-7B-Chat-GPTQ"
    backend: str = "auto"  # auto, cuda, rocm, cpu
    max_iterations: int = 10
    max_budget_usd: float = 100.0
    simulate_before_execute: bool = True
    chains: list[str] = field(default_factory=lambda: ["ethereum"])


@dataclass
class AgentState:
    """Mutable agent state across iterations."""
    iteration: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    actions_taken: list[dict] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)


class SentinelAgent:
    """Main agent that orchestrates monitoring, analysis, and execution.

    The agent runs a loop:
    1. OBSERVE — collect on-chain data from monitors
    2. REASON — LLM analyzes observations and context
    3. PLAN — LLM proposes actions (or decides to wait)
    4. EXECUTE — run proposed actions through safety rails
    5. LEARN — store outcomes for future reference
    """

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.state = AgentState()
        self.chain_registry = ChainRegistry()
        self.llm: LLMEngine | None = None
        self.simulator = TxSimulator()
        self.limits = SpendingLimits(max_usd=self.config.max_budget_usd)

    def initialize(self) -> None:
        """Load model and connect to chains."""
        logger.info("Initializing Sentinel Agent...")
        self.llm = LLMEngine(
            model=self.config.model,
            backend=self.config.backend,
        )
        for chain_name in self.config.chains:
            self.chain_registry.add(chain_name)
        logger.info(
            "Agent ready — model=%s, chains=%s",
            self.config.model,
            self.config.chains,
        )

    def run(self, query: str, context: dict[str, Any] | None = None) -> dict:
        """Run the agent loop for a given query.

        Args:
            query: Natural language instruction or question.
            context: Optional on-chain context to inject.

        Returns:
            Dict with 'response', 'actions', 'state'.
        """
        if self.llm is None:
            self.initialize()

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        if context:
            messages.insert(1, {
                "role": "system",
                "content": f"On-chain context:\n{json.dumps(context, indent=2)}",
            })

        while self.state.iteration < self.config.max_iterations:
            self.state.iteration += 1
            logger.info("Iteration %d/%d", self.state.iteration, self.config.max_iterations)

            # Reason
            response = self.llm.chat(messages)
            self.state.tokens_used += response.get("tokens", 0)

            # Parse actions from response
            actions = self._extract_actions(response["content"])

            if not actions:
                # No actions proposed — return reasoning
                return {
                    "response": response["content"],
                    "actions": [],
                    "state": self._state_snapshot(),
                }

            # Execute actions through safety rails
            results = []
            for action in actions:
                if self.config.simulate_before_execute:
                    sim_result = self.simulator.simulate(action)
                    if not sim_result["success"]:
                        results.append({
                            "action": action,
                            "status": "blocked",
                            "reason": sim_result["error"],
                        })
                        continue

                if not self.limits.check(action.get("value_usd", 0)):
                    results.append({
                        "action": action,
                        "status": "blocked",
                        "reason": "spending limit exceeded",
                    })
                    continue

                results.append({"action": action, "status": "executed"})
                self.state.actions_taken.append(action)

            # Feed results back to LLM
            messages.append({"role": "assistant", "content": response["content"]})
            messages.append({
                "role": "user",
                "content": f"Action results:\n{json.dumps(results, indent=2)}\nContinue or conclude.",
            })

        return {
            "response": "Max iterations reached.",
            "actions": self.state.actions_taken,
            "state": self._state_snapshot(),
        }

    def _extract_actions(self, content: str) -> list[dict]:
        """Extract structured actions from LLM response."""
        actions = []
        try:
            # Look for JSON blocks in response
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    actions = parsed
                elif isinstance(parsed, dict) and "actions" in parsed:
                    actions = parsed["actions"]
        except (json.JSONDecodeError, IndexError):
            pass
        return actions

    def _state_snapshot(self) -> dict:
        """Return serializable state snapshot."""
        return {
            "iterations": self.state.iteration,
            "tokens_used": self.state.tokens_used,
            "cost_usd": self.state.cost_usd,
            "actions_count": len(self.state.actions_taken),
        }
