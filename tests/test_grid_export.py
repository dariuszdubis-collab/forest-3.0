from pathlib import Path

import numpy as np
import pandas as pd

from forest.backtest.grid import param_grid, run_grid
from forest.backtest.risk import RiskManager


def test_grid_export_tmp(tmp_path: Path):
    base = np.linspace(100, 101, 20)
    df = pd.DataFrame(
        {"open": base, "high": base + 0.2, "low": base - 0.2, "close": base},
        index=pd.date_range("2025-01-01", periods=20, freq="h"),
    )
    out_file = tmp_path / "grid.parquet"
    run_grid(
        df,
        param_grid(fast=[5], slow=[20]),
        make_risk=lambda: RiskManager(capital=1_000),
        n_jobs=1,
        export_path=out_file,
    )
    assert out_file.exists()
    loaded = pd.read_parquet(out_file)
    assert len(loaded) == 1 and "equity_end" in loaded.columns

