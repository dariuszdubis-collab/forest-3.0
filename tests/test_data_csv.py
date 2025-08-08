from __future__ import annotations

import pandas as pd

from forest.data.csv_source import CSVConfig, iter_stream, load_history_csv


def _sample_df(n: int = 10) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    base = 100 + pd.Series(range(n), index=idx).astype(float)
    return pd.DataFrame(
        {
            "time": idx,
            "open": base,
            "high": base + 0.3,
            "low": base - 0.3,
            "close": base + 0.1,
            "volume": 1000,
        }
    )


def test_load_history_csv_roundtrip(tmp_path):
    raw = _sample_df(10)
    path = tmp_path / "prices.csv"
    raw.to_csv(path, index=False)

    cfg = CSVConfig(path=path, timeframe="1h", tz="UTC", symbol="SYN")
    out = load_history_csv(cfg)

    assert list(out.columns[:4]) == ["open", "high", "low", "close"]
    assert len(out) == 10
    assert out.index.tz is not None  # tz-aware


def test_iter_stream_len(tmp_path):
    raw = _sample_df(5)
    path = tmp_path / "prices.csv"
    raw.to_csv(path, index=False)

    cfg = CSVConfig(path=path, timeframe="1h", tz="UTC")
    out = load_history_csv(cfg)

    count = sum(1 for _ in iter_stream(out))
    assert count == 5

