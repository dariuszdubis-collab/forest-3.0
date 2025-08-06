import time
from pathlib import Path

import numpy as np
import pandas as pd

from forest.backtest.grid import param_grid, run_grid
from forest.backtest.risk import RiskManager


def test_joblib_cache(tmp_path: Path):
    # mini‑DF (20 świec)
    base = np.linspace(100, 101, 20)
    df = pd.DataFrame(
        {"open": base, "high": base + 0.1, "low": base - 0.1, "close": base},
        index=pd.date_range("2025-01-01", periods=20, freq="h"),
    )
    grid = param_grid(fast=[5], slow=[20])

    def make_risk() -> RiskManager:    # ← def zamiast lambda
        return RiskManager(capital=1_000)

    # pierwsze uruchomienie – liczy od zera
    t0 = time.time()
    run_grid(df, grid, make_risk=make_risk, n_jobs=1, use_cache=True)
    first = time.time() - t0

    # drugie – odczyt z cache
    t1 = time.time()
    run_grid(df, grid, make_risk=make_risk, n_jobs=1, use_cache=True)
    second = time.time() - t1

    assert second < first  # powinno być zauważalnie szybciej

