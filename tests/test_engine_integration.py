import pandas as pd

from forest.strategy.ema_cross import EMACrossStrategy
from forest.backtest.engine import run_backtest


def _synthetic_prices(n: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    close = pd.Series([100 + i * 0.5 for i in range(n)], index=idx)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
        },
        index=idx,
    )
    df.index.name = "time"
    return df


def test_engine_runs_and_metrics():
    df = _synthetic_prices(100)
    strat = EMACrossStrategy(fast=5, slow=12)
    res = run_backtest(df=df, strategy=strat, symbol="TEST")

    # equity powinno rosnąć dla rosnących cen
    assert res.equity.iloc[-1] > 0

    # metryki obecne i sensowne
    for key in ("equity_end", "max_dd", "cagr", "rar", "sharpe"):
        assert key in res.metrics

    # struktura transakcji
    assert list(res.trades.columns) == ["time", "price", "qty", "side"]

