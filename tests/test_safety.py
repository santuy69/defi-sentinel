"""Tests for spending limits."""

from defi_sentinel.safety.limits import SpendingLimits


def test_allows_within_limits():
    limits = SpendingLimits(max_usd=100, max_per_tx=50, max_per_hour=25, max_per_day=75)
    assert limits.check(10.0) is True
    assert limits.check(50.0) is True  # at per-tx max


def test_blocks_over_tx_limit():
    limits = SpendingLimits(max_per_tx=50)
    assert limits.check(51.0) is False


def test_blocks_over_total_budget():
    limits = SpendingLimits(max_usd=100)
    limits.record(60.0, action="swap")
    assert limits.check(50.0) is False  # 60 + 50 > 100


def test_summary():
    limits = SpendingLimits(max_usd=100)
    limits.record(30.0, action="swap")
    s = limits.summary()
    assert s["total_usd"] == 30.0
    assert s["budget_remaining"] == 70.0
    assert s["records"] == 1


def test_clear_resets():
    limits = SpendingLimits(max_usd=100)
    limits.record(90.0, action="big_swap")
    assert limits.check(20.0) is False
    limits._records.clear()
    assert limits.check(20.0) is True
