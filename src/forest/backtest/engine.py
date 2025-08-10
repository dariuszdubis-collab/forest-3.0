# src/forest/backtest/engine.py
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from forest.backtest.risk import RiskManager
from forest.backtest.trace import DecisionTrace
from forest.backtest.tradebook import Trade, TradeBook
from forest.core.indicators import atr, ema
from forest.utils.log import log


def ema_cross_strategy(
    df: pd.DataFrame, fast: int = 12, slow: int = 26
) -> pd.Series:
    """
    Prosta strategia: sygnał z przecięcia EMA(fast) i EMA(slow).
    Zwraca serię {-1, 0, 1}.
    """
    f = ema(df["close"].values, fast)
    s = ema(df["close"].values, slow)
    sig = np.sign(f - s).astype(np.int32)
    # Na początkowych NaN z EMA zwracamy 0
    sig = np.where(np.isnan(f) | np.isnan(s), 0, sig)
    return pd.Series(sig, index=df.index, name="signal")


def _trace_to_payload(trace: DecisionTrace) -> Dict[str, Any]:
    """Zamień DecisionTrace na dict niezależnie od implementacji."""
    if hasattr(trace, "model_dump"):
        try:
            return trace.model_dump()  # Pydantic v2
        except Exception:
            pass
    if hasattr(trace, "dict"):
        try:
            return trace.dict()  # Pydantic v1
        except Exception:
            pass
    if is_dataclass(trace):
        try:
            return asdict(trace)
        except Exception:
            pass
    # Fallback
    return getattr(trace, "__dict__", {"time": None, "symbol": None, "filters": {}, "final": None})


def run_backtest(
    df: pd.DataFrame,
    risk: RiskManager,
    fast: int = 12,
    slow: int = 26,
) -> pd.DataFrame:
    """
    Uruchamia wektorowy back‑test na DF świec.

    Zwraca kopię wejściowego DF z kolumnami:
    - signal: -1/0/1 z ema_cross_strategy
    - atr: ATR(14)
    - equity: kapitał konta (mark‑to‑market, po domknięciu pozycji na końcu)
    """
    out = df.copy()

    # 1) sygnał strategii
    out["signal"] = ema_cross_strategy(out, fast=fast, slow=slow)

    # 2) ATR do position sizingu
    out["atr"] = atr(out["high"].values, out["low"].values, out["close"].values, period=14)

    tb = TradeBook()

    position: int | None = None  # 1 LONG, -1 SHORT, None = flat
    entry_price: float | None = None
    entry_qty: float | None = None

    for idx, row in out.iterrows():
        sig = int(row.signal)

        # ---------- trailing‑SL aktualizacja i ewentualne zamknięcie ----------
        if position is not None:
            risk.update_trailing_sl(float(row.close), float(row.atr))
            if risk.hit_trailing_sl(float(row.close)):
                pnl = (float(row.close) - float(entry_price)) * position * float(entry_qty)
                cost = risk.position_cost(float(entry_qty), float(row.close))
                risk.record_trade(pnl - cost)
                tb.add(Trade(idx, float(row.close), float(entry_qty), "LONG" if position == 1 else "SHORT"))
                log.warning("trailing_sl_hit", time=str(idx), price=float(row.close))
                position = entry_price = entry_qty = None
                # nie przerywamy — pozwalamy strategii dalej działać

        # ---------- zmiana sygnału ⇒ zamknięcie starej + otwarcie nowej ----------
        if sig != 0 and sig != position:
            # zamknij starą pozycję (jeśli była)
            if position is not None:
                pnl = (float(row.close) - float(entry_price)) * position * float(entry_qty)
                cost = risk.position_cost(float(entry_qty), float(row.close))
                risk.record_trade(pnl - cost)
                tb.add(Trade(idx, float(row.close), float(entry_qty), "LONG" if position == 1 else "SHORT"))

            # otwórz nową pozycję
            qty = risk.position_size(float(row.atr))
            if qty == 0:
                continue

            position = 1 if sig > 0 else -1
            entry_price = float(row.close)
            entry_qty = float(qty)

            trace = DecisionTrace(
                time=str(idx),
                symbol="SYN",
                filters={"atr_ok": bool(qty > 0), "trailing_hit": False},
                final="BUY" if position == 1 else "SELL",
            )
            log.info("decision", **_trace_to_payload(trace))

    # ---------- domknij ewentualnie otwartą pozycję na końcu ----------
    if position is not None:
        last_idx = out.index[-1]
        last_close = float(out["close"].iloc[-1])
        pnl = (last_close - float(entry_price)) * position * float(entry_qty)
        cost = risk.position_cost(float(entry_qty), last_close)
        risk.record_trade(pnl - cost)
        tb.add(Trade(last_idx, last_close, float(entry_qty), "LONG" if position == 1 else "SHORT"))
        position = entry_price = entry_qty = None

    # ---------- zbuduj equity: dopasuj PnL z TradeBook do absolutnego equity z RiskManager ----------
    final_equity = float(risk._equity_curve[-1]) if getattr(risk, "_equity_curve", None) else float(risk.capital)

    eq_pnl = tb.equity_curve()  # zazwyczaj seria PnL (cumulative), indeks po momentach transakcji
    if eq_pnl is not None and len(eq_pnl) > 0:
        eq_pnl = eq_pnl.astype(float)
        # Usuń ewentualne duplikaty indeksu, zostaw ostatnią wartość
        if eq_pnl.index.has_duplicates:
            eq_pnl = eq_pnl[~eq_pnl.index.duplicated(keep="last")]

        # Skoryguj stałą tak, aby ostatnia wartość serii == final_equity
        shift = final_equity - float(eq_pnl.iloc[-1])
        eq_abs = (eq_pnl + shift).reindex(out.index).ffill()
        out["equity"] = eq_abs.astype(float)
    else:
        # Brak transakcji — wpisz płaską linię kapitału
        out["equity"] = pd.Series(final_equity, index=out.index, dtype=float)

    return out

