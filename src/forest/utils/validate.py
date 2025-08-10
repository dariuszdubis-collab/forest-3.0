from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class FrameCheckResult:
    ok: bool
    msg: str | None = None


REQUIRED_OHLC: tuple[str, ...] = ("open", "high", "low", "close")


def ensure_backtest_ready(
    df: pd.DataFrame,
    *,
    required: Sequence[str] = REQUIRED_OHLC,
) -> pd.DataFrame:
    """
    Upewnia się, że ramka do backtestu:
    - ma kolumny OHLC,
    - ma DatetimeIndex,
    - indeks jest posortowany rosnąco i bez duplikatów.

    Funkcja działa defensywnie: nie zmienia wartości danych,
    jedynie porządkuje indeks/czyszczy duplikaty i zwraca kopię.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("ensure_backtest_ready: expected pandas.DataFrame")

    if any(col not in df.columns for col in required):
        missing = [c for c in required if c not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()

    # DatetimeIndex (jeżeli nie ma, spróbuj sparsować)
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index)
        except Exception as exc:  # pragma: no cover
            raise ValueError("Cannot convert index to DatetimeIndex") from exc

    # sortuj rosnąco
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()

    # usuń duplikaty (zostaw ostatnią świecę dla powtarzającego się znacznika czasu)
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]

    return out

