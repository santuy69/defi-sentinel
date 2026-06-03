"""Kill switch — emergency stop for all agent operations."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

KILL_SWITCH_FILE = Path.home() / ".defi-sentinel" / "killswitch"


class KillSwitch:
    """Emergency stop mechanism for the agent.

    The kill switch can be triggered:
    1. Programmatically via kill()
    2. By creating the kill switch file manually
    3. Via CLI: sentinel stop

    The agent checks is_active() before every action.
    When active, ALL pending and new actions are blocked.

    Usage:
        ks = KillSwitch()

        # Agent checks before every action
        if ks.is_active():
            print("Agent is stopped. Clear with ks.clear()")

        # Emergency stop
        ks.kill(reason="Manual intervention")

        # Resume
        ks.clear()
    """

    def __init__(self, path: Path | None = None):
        self.path = path or KILL_SWITCH_FILE

    def is_active(self) -> bool:
        """Check if kill switch is active."""
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                reason = data.get("reason", "unknown")
                timestamp = data.get("timestamp", 0)
                age = time.time() - timestamp
                logger.warning(
                    "KILL SWITCH ACTIVE — reason: %s (triggered %.0fs ago)",
                    reason, age,
                )
                return True
            except (json.JSONDecodeError, KeyError):
                # File exists = kill switch active regardless
                return True
        return False

    def kill(self, reason: str = "Manual stop") -> None:
        """Activate the kill switch.

        Args:
            reason: Why the kill switch was triggered.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "reason": reason,
            "timestamp": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.path.write_text(json.dumps(data, indent=2))
        logger.critical("KILL SWITCH ACTIVATED: %s", reason)

    def clear(self) -> None:
        """Deactivate the kill switch."""
        if self.path.exists():
            self.path.unlink()
            logger.info("Kill switch cleared — agent can resume")

    def get_info(self) -> dict | None:
        """Get kill switch info if active."""
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, KeyError):
            return {"reason": "unknown", "timestamp": 0}
