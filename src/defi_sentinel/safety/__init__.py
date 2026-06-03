"""Safety package — execution rails for the agent."""

from defi_sentinel.safety.simulator import TxSimulator
from defi_sentinel.safety.limits import SpendingLimits
from defi_sentinel.safety.killswitch import KillSwitch

__all__ = ["TxSimulator", "SpendingLimits", "KillSwitch"]
