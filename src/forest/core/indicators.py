from __future__ import annotations

import numpy as np

# --- Hot‑fix dla pandas‑ta vs NumPy 2 ------------------------------
if not hasattr(np, "NaN"):
    np.NaN = np.nan
# ------------------------------------------------------------------

import pandas as pd
import pandas_ta as pta

__all__ = ["ema", "atr"]


def ema(prices: np.ndarray, period: int) -> np.ndarray:
    """EMA – zwraca tablicę float64 tej samej długości co wejście."""
    if period <= 0:
        raise ValueError("period must be > 0")
    ser = pd.Series(prices, dtype="float64")
    return pta.ema(ser, length=period).to_numpy()


def atr(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    period: int = 14,
) -> np.ndarray:
    ser = pd.Series(close, dtype="float64")  # indeks potrzebny, ale mało istotny
    df = pd.DataFrame({"high": high, "low": low, "close": ser})
    return pta.atr(df["high"], df["low"], df["close"], length=period).to_numpy()

