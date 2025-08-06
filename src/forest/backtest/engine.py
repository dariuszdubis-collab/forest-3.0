from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import structlog

from forest.backtest.risk import RiskManager
from forest.backtest.trace import DecisionTrace
from forest.backtest.tradebook import Trade, TradeBook
from forest.core.indicators import atr, ema

log = structlog.get_logger()


# ---------------------------------------------------------------------------#
#  Strategie                                                                 #
# ---------------------------------------------------------------------------#
def ema_cross_strategy(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.Series:
    """Sygnał: 1 = LONG, -1 = SHORT, 0 = brak sygnału (przed wypełnieniem historii)."""
    fast_ma = ema(df["close"].to_numpy(), fast)
    slow_ma = ema(df["close"].to_numpy(), slow)
    signal = np.where(fast_ma > slow_ma, 1, -1)
    signal[: slow] = 0
    return pd.Series(signal, index=df.index, name="signal")


# ---------------------------------------------------------------------------#
#  Back‑tester                                                               #
# ---------------------------------------------------------------------------#
def run_backtest(df: pd.DataFrame, risk: RiskManager) -> pd.DataFrame:
    """Uruchamia wektorowy back‑test na DataFrame świec."""
    out = df.copy()

    # 1. sygnał strategii
    out["signal"] = ema_cross_strategy(df)

    # 2. ATR do position sizingu
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

        # zaksięguj PnL
        sign = 1 if side == "LONG" else -1
        risk.record_trade(sign * qty * row.close)

        # DecisionTrace + log (JSON)
        trace = DecisionTrace(
            time=str(idx),
            symbol="SYN",
            filters={"atr_ok": qty > 0},
            final="BUY" if side == "LONG" else "SELL",
        )
        log.info("decision", **asdict(trace))

        # globalny stop, jeśli max DD przekroczone
        if risk.exceeded_max_dd():
            log.warning("max_dd_reached", equity=risk.equity)
            break

    # 3. equity curve uzupełniona do pełnej długości df
    out["equity"] = tb.equity_curve().reindex(out.index).ffill()
    return out

