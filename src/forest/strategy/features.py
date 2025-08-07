"""
forest.strategy.features
------------------------

Bardzo prosty pipe → z czasem podmienimy na bardziej rozbudowany.
Zwraca macierz float32 (wymóg onnxruntime).
"""

from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Z klasycznej ramki OHLC tworzy kilka cech technicznych."""
    out = pd.DataFrame(index=df.index)

    out["ret1"] = df["close"].pct_change().fillna(0.0)

    out["ma_fast"] = df["close"].rolling(5).mean().bfill()
    out["ma_slow"] = df["close"].rolling(20).mean().bfill()
    out["ma_diff"] = out["ma_fast"] - out["ma_slow"]

    return out.astype("float32")

