"""Techniczne wskaźniki używane w FOREST 3.0."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as pta

__all__ = ["ema", "atr", "ema_cross_strategy"]


# --------------------------------------------------------------------------- #
#  EMA
# --------------------------------------------------------------------------- #
def ema(prices: np.ndarray | pd.Series, period: int) -> np.ndarray:
    """Exponential Moving Average.

    Parametry
    ---------
    prices : array‑like
        Wektor cen (float64) lub ``pd.Series``.
    period : int
        Długość EMA (musi być > 0).

    Zwraca
    -------
    np.ndarray
        EMA jako tablica `float64` o tej samej długości co wejście.
    """
    if period <= 0:  # defensive
        raise ValueError("period must be > 0")

    series = pd.Series(prices, copy=False)  # konwersja, bez kopiowania danych
    out = pta.ema(series, length=period)

    # starsze / ubogie wersje *pandas‑ta* potrafią zwrócić ``None``,
    # dlatego mamy rezerwę w postaci natywnego obliczenia.
    if out is None:
        out = series.ewm(span=period, adjust=False).mean()

    return out.to_numpy(dtype=float)


# --------------------------------------------------------------------------- #
#  ATR
# --------------------------------------------------------------------------- #
def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range (ATR)."""
    out = pta.atr(high, low, close, length=period)

    if out is None:  # fallback – klasyczna formuła z Wildera
        tr1 = (high - low).abs()
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        out = true_range.rolling(window=period, min_periods=period).mean()

    return out.astype(float)


# --------------------------------------------------------------------------- #
#  Prosta strategia EMA‑cross
# --------------------------------------------------------------------------- #
def ema_cross_strategy(
    df: pd.DataFrame,
    fast: int = 5,
    slow: int = 20,
) -> pd.Series:
    """Sygnał +1 / 0 / ‑1 na bazie przecięcia EMA(fast) i EMA(slow)."""
    fast_ema = ema(df["close"], fast)
    slow_ema = ema(df["close"], slow)

    signal = np.where(
        fast_ema > slow_ema,
        1,
        np.where(fast_ema < slow_ema, -1, 0),
    )

    return pd.Series(signal, index=df.index, name="signal", dtype=int)

