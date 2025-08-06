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
    sig = np.where(ema(df["close"].to_numpy(), fast) > ema(df["close"].to_numpy(), slow), 1, -1)
    sig[: slow] = 0
    return pd.Series(sig, index=df.index, name="signal")


# ---------------------------------------------------------------------------#
#  Back‑tester                                                               #
# ---------------------------------------------------------------------------#
def run_backtest(df: pd.DataFrame, risk: RiskManager) -> pd.DataFrame:
    out = df.copy()
    out["signal"] = ema_cross_strategy(df)
    out["atr"] = atr(df["high"], df["low"], df["close"], period=14)

    tb = TradeBook()

    position: int | None = None        # 1 LONG, -1 SHORT, None = flat
    entry_price: float | None = None
    entry_qty: float | None = None

    for idx, row in out.iterrows():
        sig = int(row.signal)

        # ---------- trailing‑SL aktualizacja i ewentualne zamknięcie ----------
        if position is not None:
            risk.update_trailing_sl(row.close, row.atr)
            if risk.hit_trailing_sl(row.close):
                pnl = (row.close - entry_price) * position * entry_qty
                cost = risk.position_cost(entry_qty, row.close)
                risk.record_trade(pnl - cost)
                tb.add(Trade(idx, row.close, entry_qty, "LONG" if position == 1 else "SHORT"))
                log.warning("trailing_sl_hit", time=str(idx), price=row.close)
                position = entry_price = entry_qty = None
                break                                            # kończymy back‑test

        # ---------- zmiana sygnału ⇒ zamknięcie starej + otwarcie nowej ----------
        if sig != 0 and sig != position:
            # zamknij starą pozycję (jeśli była)
            if position is not None:
                pnl = (row.close - entry_price) * position * entry_qty
                cost = risk.position_cost(entry_qty, row.close)
                risk.record_trade(pnl - cost)
                tb.add(Trade(idx, row.close, entry_qty, "LONG" if position == 1 else "SHORT"))

            # otwórz nową pozycję
            qty = risk.position_size(row.atr)
            if qty == 0:
                continue

            cost = risk.position_cost(qty, row.close)
            risk.record_trade(-cost)              # pobierz koszty otwarcia
            tb.add(Trade(idx, row.close, qty, "LONG" if sig == 1 else "SHORT"))

            position, entry_price, entry_qty = sig, row.close, qty

            trace = DecisionTrace(
                time=str(idx),
                symbol="SYN",
                filters={"atr_ok": qty > 0},
                final="BUY" if sig == 1 else "SELL",
            )
            log.info("decision", **asdict(trace))

            if risk.exceeded_max_dd():
                log.error("max_dd_reached", equity=risk.equity)
                break

    # ---------- zamknięcie pozycji na końcu danych ----------
    if position is not None:
        last_idx = out.index[-1]
        last_price = out["close"].iloc[-1]
        pnl = (last_price - entry_price) * position * entry_qty
        cost = risk.position_cost(entry_qty, last_price)
        risk.record_trade(pnl - cost)
        tb.add(Trade(last_idx, last_price, entry_qty, "LONG" if position == 1 else "SHORT"))

    out["equity"] = tb.equity_curve().reindex(out.index).ffill()
    return out

