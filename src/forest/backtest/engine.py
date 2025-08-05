from __future__ import annotations

import numpy as np
import pandas as pd

from forest.backtest.risk import RiskManager
from forest.backtest.trace import DecisionTrace
from forest.backtest.tradebook import Trade, TradeBook
from forest.core.indicators import atr, ema
from forest.utils.log import setup_logger


def test_trace_dict():
    tr = DecisionTrace("2025-01-01", "EURUSD", {"atr": True}, "BUY")
    assert tr.final == "BUY"
    assert tr.filters["atr"] is True

def test_setup_logger():
    setup_logger("DEBUG")  # nie rzuca wyjątków


def ema_cross_strategy(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.Series:
    fast_ma = ema(df["close"].to_numpy(), fast)
    slow_ma = ema(df["close"].to_numpy(), slow)
    signal = np.where(fast_ma > slow_ma, 1, -1)   # 1 LONG, −1 SHORT
    signal[: slow] = 0                             # brak sygnału przed pełną historią
    return pd.Series(signal, index=df.index, name="signal")


def run_backtest(df: pd.DataFrame, risk: RiskManager) -> pd.DataFrame:
    out = df.copy()

    # 1. sygnał
    out["signal"] = ema_cross_strategy(df)

    # 2. ATR do sizingu
    out["atr"] = atr(df["high"], df["low"], df["close"], period=14)

    tb = TradeBook()

    for idx, row in out.iterrows():
        if row.signal == 0:
            continue

        qty = risk.position_size(row.atr)
        if qty == 0:
            continue

        side = "LONG" if row.signal == 1 else "SHORT"
        tb.add(Trade(time=idx, price=row.close, qty=qty, side=side))

        # zapisz PnL
        sign = 1 if side == "LONG" else -1
        risk.record_trade(sign * qty * row.close)

        # Stop trading, jeśli limit DD przekroczony
        if risk.exceeded_max_dd():
            break

    out["equity"] = tb.equity_curve().reindex(out.index).ffill()
    return out
