import numpy as np
import pandas as pd

from forest.backtest.engine import run_backtest
from forest.backtest.risk import RiskManager


def synthetic_prices(n=100):
    """Generuje price‑series z delikatnym trendem, by sprawdzić equity > 0."""
    rng = np.random.default_rng(42)
    base = np.cumsum(rng.normal(0.05, 0.5, n)) + 100
    df = pd.DataFrame(
        {
            "open": base,
            "high": base + 0.3,
            "low": base - 0.3,
            "close": base + rng.normal(0, 0.1, n),
        },
        index=pd.date_range("2025-01-01", periods=n, freq="h"),
    )
    return df


def test_backtester_positive_equity():
    df = synthetic_prices()
    rm = RiskManager(capital=10_000, risk_per_trade=0.02, max_drawdown=0.20)

    out = run_backtest(df, rm)

    assert not out["equity"].isna().all()
    # ostatnia wartość equity powinna być > start (losowy, ale trend wzrostowy)
    assert out["equity"].iloc[-1] > rm.capital
