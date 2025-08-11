import pandas as pd
from forest.strategy.ema_cross import EMACrossStrategy

def test_ema_cross_basic_signals():
    df = pd.DataFrame({
        "time": pd.date_range("2022-01-01", periods=10, freq="D"),
        "open": [1,1,1,1,1,1,1,1,1,1],
        "high": [1,1,1,1,1,1,1,1,1,1],
        "low":  [1,1,1,1,1,1,1,1,1,1],
        "close":[1,1,1,2,3,4,5,5,4,3],
    }).set_index("time")
    strat = EMACrossStrategy(fast=2, slow=3)
    out = strat.init(df.copy())
    # sprawdź, czy dodało kolumny EMA:
    assert "ema_fast" in out.columns and "ema_slow" in out.columns

