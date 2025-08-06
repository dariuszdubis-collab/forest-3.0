"""
Testy modułu ParamGrid Runner.
Zakładamy, że run_backtest przyjmuje parametry fast/slow EMA.
"""

import numpy as np
import pandas as pd

from forest.backtest.grid import param_grid, run_grid
from forest.backtest.risk import RiskManager


def synthetic_prices(n: int = 50) -> pd.DataFrame:
    """Mały DF OHLC z łagodnym trendem + stały spread 0.2."""
    base = np.linspace(100, 105, n)
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 0.2,
            "low": base - 0.2,
            "close": base,
        },
        index=pd.date_range("2025-01-01", periods=n, freq="h"),
    )


# ---------------------------------------------------------------------------#
#  Jednostkowe                                                                #
# ---------------------------------------------------------------------------#
def test_param_grid():
    grid = list(param_grid(fast=[5, 10], slow=[20, 30]))
    assert len(grid) == 4
    assert {"fast": 5, "slow": 20} in grid


# ---------------------------------------------------------------------------#
#  Integracyjne – pojedynczy przebieg                                         #
# ---------------------------------------------------------------------------#
def test_run_grid_small():
    df = synthetic_prices()
    results = run_grid(
        df,
        param_grid(fast=[5], slow=[20]),
        make_risk=lambda: RiskManager(capital=1_000),
        n_jobs=1,  # bez multiprocessing w CI
    )
    assert len(results) == 1
    row = results.iloc[0]
    assert row["equity_end"] > 0
    assert 0 <= row["max_dd"] <= 1


# ---------------------------------------------------------------------------#
#  Ten sam grid – wersja równoległa (n_jobs=1 w CI)                           #
# ---------------------------------------------------------------------------#
def test_run_grid_small_parallel():
    df = synthetic_prices()
    res = run_grid(
        df,
        param_grid(fast=[5], slow=[20]),
        make_risk=lambda: RiskManager(capital=1_000),
        n_jobs=1,  # w CI trzymamy 1 proces, lokalnie można -1
    )
    assert res.iloc[0]["equity_end"] > 0

