from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

from forest.strategy.base import Strategy, Signal
from forest.core.indicators import ema


class EMACrossStrategy(Strategy):
    """
    Sygnał BUY gdy EMA_fast > EMA_slow, SELL gdy EMA_fast < EMA_slow, w przeciwnym razie HOLD.
    Parametry: fast:int, slow:int  (slow > fast zalecane)
    """
    def __init__(self, fast: int = 12, slow: int = 26, column: str = "close") -> None:
        super().__init__(fast=fast, slow=slow, column=column)
        self.fast = int(fast)
        self.slow = int(slow)
        self.column = column
        self._pos_open: bool = False  # stan pozycji (long/flat)

    def init(self, df: pd.DataFrame) -> pd.DataFrame:
        series = df[self.column].astype(float).to_numpy()
        df["ema_fast"] = ema(series, length=self.fast)
        df["ema_slow"] = ema(series, length=self.slow)
        # sygnał wektorowy (-1/0/1) pomocniczo — może się przydać UI
        cross = np.where(
            (df["ema_fast"] > df["ema_slow"]), 1,
            np.where((df["ema_fast"] < df["ema_slow"]), -1, 0)
        )
        df["signal_vec"] = cross
        return df

    def on_bar(self, row: pd.Series) -> Signal:
        # pomijamy wczesne NaN okresów EMA
        if pd.isna(row.get("ema_fast")) or pd.isna(row.get("ema_slow")):
            return Signal("HOLD", "warmup")
        if row["ema_fast"] > row["ema_slow"] and not self._pos_open:
            self._pos_open = True
            return Signal("BUY", "ema_fast>ema_slow")
        if row["ema_fast"] < row["ema_slow"] and self._pos_open:
            self._pos_open = False
            return Signal("SELL", "ema_fast<ema_slow")
        return Signal("HOLD")

