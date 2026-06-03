"""Tests for kill switch."""

import json
import tempfile
from pathlib import Path

from defi_sentinel.safety.killswitch import KillSwitch


def test_kill_switch_off_by_default():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KillSwitch(path=Path(tmp) / "killswitch")
        assert ks.is_active() is False


def test_kill_switch_activation():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KillSwitch(path=Path(tmp) / "killswitch")
        ks.kill(reason="test stop")
        assert ks.is_active() is True
        data = ks.get_info()
        assert data["reason"] == "test stop"


def test_kill_switch_clear():
    with tempfile.TemporaryDirectory() as tmp:
        ks = KillSwitch(path=Path(tmp) / "killswitch")
        ks.kill(reason="test")
        assert ks.is_active() is True
        ks.clear()
        assert ks.is_active() is False
        assert ks.get_info() is None
