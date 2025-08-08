from __future__ import annotations

import pytest
from forest.utils.timeframes import normalize_timeframe, to_minutes


def test_normalize_timeframe_variants():
    assert normalize_timeframe("H") == "1h"
    assert normalize_timeframe("1H") == "1h"
    assert normalize_timeframe("15m") == "15m"
    assert normalize_timeframe("D") == "1d"
    assert normalize_timeframe("  4h ") == "4h"


def test_to_minutes():
    assert to_minutes("1m") == 1
    assert to_minutes("15m") == 15
    assert to_minutes("1h") == 60
    assert to_minutes("4h") == 240
    assert to_minutes("1d") == 1440


def test_invalid_timeframe():
    with pytest.raises(ValueError):
        normalize_timeframe("2h")   # nie mamy 2h w predefiniowanej liście

