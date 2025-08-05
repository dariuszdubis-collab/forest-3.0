from forest.backtest.trace import DecisionTrace
from forest.utils.log import setup_logger


def test_trace_dataclass():
    tr = DecisionTrace("2025-01-01", "EURUSD", {"atr": True}, "BUY")
    assert tr.final == "BUY"
    assert tr.filters["atr"] is True


def test_setup_logger_no_exception():
    setup_logger("DEBUG")  # nie rzuca wyjątków

