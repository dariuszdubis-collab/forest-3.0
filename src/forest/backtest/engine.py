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


def _trace_to_dict(trace: DecisionTrace) -> Dict[str, Any]:
    """Zwraca słownik dla loggera niezależnie czy DecisionTrace to dataclass/Pydantic."""
    if is_dataclass(trace):
        return asdict(trace)
    if hasattr(trace, "model_dump") and callable(getattr(trace, "model_dump")):
        return trace.model_dump()  # pydantic v2
    if hasattr(trace, "dict") and callable(getattr(trace, "dict")):
        return trace.dict()  # pydantic v1
    # awaryjnie
    return {k: getattr(trace, k) for k in ("time", "symbol", "filters", "final") if hasattr(trace, k)}


def ema_cross_strategy(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
) -> pd.Series:
    """Prosty sygnał: znak różnicy EMA(fast) i EMA(slow)."""
    close = df["close"].astype(float).to_numpy()
    f = ema(close, period=fast)
    s = ema(close, period=slow)

    diff = f - s
    sig = np.zeros_like(diff, dtype=np.int8)
    mask = ~np.isnan(diff)
    sig[mask] = np.sign(diff[mask]).astype(np.int8)

    return pd.Series(sig, index=df.index, name="signal")


def run_backtest(df: pd.DataFrame, risk: RiskManager) -> pd.DataFrame:
    """Uruchamia wektorowy back‑test na syntetycznym DF świec.

    Zwraca kopię wejściowego DF z kolumnami:
      - signal: -1/0/1 (EMA cross)
      - atr: ATR(14)
      - equity: kapitał konta (mark‑to‑market, po zamknięciach), z ffill po indeksie
    """
    out = df.copy()

    # 1) sygnał strategii
    out["signal"] = ema_cross_strategy(out)

    # 2) ATR do position sizingu
    out["atr"] = atr(out["high"].values, out["low"].values, out["close"].values, period=14)

    tb = TradeBook()

    position: int | None = None        # 1 LONG, -1 SHORT, None = flat
    entry_price: float | None = None
    entry_qty: float | None = None

    # tu będziemy odkładać „znaczniki” equity w chwilach księgowania PnL
    eq_marks: dict[pd.Timestamp, float] = {}

    for idx, row in out.iterrows():
        price = float(row.close)
        vol_atr = float(row.atr) if not pd.isna(row.atr) else np.nan
        sig = int(row.signal)

        # ---------- trailing‑SL aktualizacja i ewentualne zamknięcie ----------
        if position is not None:
            risk.update_trailing_sl(price, vol_atr)
            if risk.hit_trailing_sl(price):
                # zamknięcie pozycji na trailing SL
                pnl = (price - float(entry_price)) * position * float(entry_qty)
                cost = risk.position_cost(float(entry_qty), price)
                risk.record_trade(pnl - cost)
                tb.add(Trade(idx, price, float(entry_qty), "LONG" if position == 1 else "SHORT"))
                log.warning("trailing_sl_hit", time=str(idx), price=price)

                # znacznik equity w tej chwili
                eq_marks[idx] = float(risk._equity_curve[-1])  # wykorzystujemy aktualny equity z RiskManager
                position = entry_price = entry_qty = None
                # kontynuujemy pętlę (nie przerywamy testu), żeby equity było wypełnione do końca
                continue

        # ---------- zmiana sygnału ⇒ zamknięcie starej + otwarcie nowej ----------
        if sig != 0 and sig != position:
            # zamknij starą pozycję (jeśli była)
            if position is not None:
                pnl = (price - float(entry_price)) * position * float(entry_qty)
                cost = risk.position_cost(float(entry_qty), price)
                risk.record_trade(pnl - cost)
                tb.add(Trade(idx, price, float(entry_qty), "LONG" if position == 1 else "SHORT"))
                # znacznik equity
                eq_marks[idx] = float(risk._equity_curve[-1])

            # otwórz nową pozycję
            qty = risk.position_size(vol_atr)
            if qty == 0:
                continue

            position = 1 if sig > 0 else -1
            entry_price = price
            entry_qty = float(qty)

            trace = DecisionTrace(
                time=str(idx),
                symbol="SYN",
                filters={"atr_ok": bool(qty > 0), "trailing_hit": False},
                final="BUY" if position == 1 else "SELL",
            )
            log.info("decision", **_trace_to_dict(trace))

    # ---------- domknięcie otwartej pozycji na końcu (mark‑to‑market) ----------
    if position is not None:
        last_idx = out.index[-1]
        last_price = float(out["close"].iloc[-1])
        pnl = (last_price - float(entry_price)) * position * float(entry_qty)
        cost = risk.position_cost(float(entry_qty), last_price)
        risk.record_trade(pnl - cost)
        tb.add(Trade(last_idx, last_price, float(entry_qty), "LONG" if position == 1 else "SHORT"))
        eq_marks[last_idx] = float(risk._equity_curve[-1])

    # ---------- budowa kolumny equity ----------
    if eq_marks:
        # to są rzeczywiste poziomy kapitału w chwilach księgowania PnL
        eq_series = pd.Series(eq_marks).sort_index()
    else:
        # brak transakcji – płaska linia z kapitałem początkowym
        eq_series = pd.Series([float(risk.capital)], index=[out.index[0]])

    # reindeks i wypełnienie do całego DF
    out["equity"] = eq_series.reindex(out.index).ffill()

    # upewnij się, że pierwszy punkt to kapitał początkowy
    if pd.isna(out["equity"].iloc[0]):
        out.at[out.index[0], "equity"] = float(risk.capital)
    else:
        # jeśli seria pochodziła z PnL (gdyby TradeBook zwracał PnL),
        # to i tak mamy już equity z RiskManager – nic nie dodajemy
        pass

    return out

