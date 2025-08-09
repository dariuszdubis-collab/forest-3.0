# src/forest/backtest/engine.py
from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import pandas as pd

from forest.backtest.risk import RiskManager
from forest.backtest.trace import DecisionTrace
from forest.backtest.tradebook import Trade, TradeBook
from forest.core.indicators import atr as _atr
from forest.core.indicators import ema as _ema

log = logging.getLogger("forest")


def ema_cross_strategy(
    df: pd.DataFrame, *, fast: int = 12, slow: int = 26
) -> pd.Series:
    """Prosty sygnał: znak różnicy EMA(fast) i EMA(slow)."""
    close = df["close"].to_numpy(dtype=float, copy=False)
    f = _ema(close, period=fast)
    s = _ema(close, period=slow)
    # gdzie są NaN-y (początek) → 0
    raw = np.sign(f - s)
    raw = np.where(np.isfinite(raw), raw, 0.0).astype(np.int32)
    return pd.Series(raw, index=df.index, name="signal")


def _trace_to_dict(obj: Any) -> dict[str, Any]:
    """Bezpieczna serializacja DecisionTrace do dict dla loggera."""
    if hasattr(obj, "model_dump"):  # pydantic v2
        return obj.model_dump()  # type: ignore[attr-defined]
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": repr(obj)}


def run_backtest(
    df: pd.DataFrame,
    risk: RiskManager,
    *strategy_args: Any,
    **strategy_kwargs: Any,
) -> pd.DataFrame:
    """Uruchamia wektorowy back‑test.

    Parametry strategii można podać pozycyjnie (fast, slow) albo nazwami
    (fast=…, slow=…). Dodatkowe klucze są ignorowane.
    """
    # Parametry strategii (obsługa zarówno pozycyjnych, jak i nazwanych)
    fast: int | None = None
    slow: int | None = None
    if len(strategy_args) >= 2:
        fast = int(strategy_args[0])
        slow = int(strategy_args[1])
    fast = int(strategy_kwargs.get("fast", fast if fast is not None else 12))
    slow = int(strategy_kwargs.get("slow", slow if slow is not None else 26))

    out = df.copy()

    # 1) sygnał strategii i ATR
    out["signal"] = ema_cross_strategy(out, fast=fast, slow=slow)
    out["atr"] = _atr(
        out["high"].to_numpy(dtype=float, copy=False),
        out["low"].to_numpy(dtype=float, copy=False),
        out["close"].to_numpy(dtype=float, copy=False),
        period=14,
    )

    tb = TradeBook()

    position: int | None = None  # 1 LONG, -1 SHORT, None = flat
    entry_price: float | None = None
    entry_qty: float | None = None

    account_equity = float(risk.capital)
    equity_curve: list[float] = []

    for idx, row in out.iterrows():
        price = float(row.close)
        sig = int(row.signal)
        atr_val = float(row.atr) if np.isfinite(row.atr) else 0.0

        # --- trailing‑SL (jeśli mamy otwartą pozycję) ---
        if position is not None:
            risk.update_trailing_sl(price, atr_val)
            if risk.hit_trailing_sl(price):
                # zamknięcie po SL
                pnl = (price - float(entry_price)) * position * float(entry_qty)
                cost = risk.position_cost(float(entry_qty), price)
                account_equity += pnl - cost
                tb.add(Trade(idx, price, float(entry_qty), "LONG" if position == 1 else "SHORT"))
                log.warning("trailing_sl_hit", extra={"time": str(idx), "price": price})
                position = entry_price = entry_qty = None  # zamknięto

        # --- zmiana sygnału ⇒ zamknięcie starej + otwarcie nowej ---
        if sig != 0 and sig != position:
            # zamknij starą
            if position is not None:
                pnl = (price - float(entry_price)) * position * float(entry_qty)
                cost = risk.position_cost(float(entry_qty), price)
                account_equity += pnl - cost
                tb.add(Trade(idx, price, float(entry_qty), "LONG" if position == 1 else "SHORT"))

            # otwórz nową (jeśli sizing > 0)
            qty = risk.position_size(atr_val)
            if qty > 0:
                # koszt wejścia
                open_cost = risk.position_cost(float(qty), price)
                account_equity -= open_cost

                position = 1 if sig > 0 else -1
                entry_price = price
                entry_qty = float(qty)

                trace = DecisionTrace(
                    time=str(idx),
                    symbol="SYN",
                    filters={"atr_ok": bool(qty > 0), "trailing_hit": False},
                    final="BUY" if position == 1 else "SELL",
                )
                log.info("decision", extra=_trace_to_dict(trace))

        # mark‑to‑market na koniec świecy
        if position is None:
            equity_curve.append(account_equity)
        else:
            mtm = account_equity + (price - float(entry_price)) * position * float(entry_qty)
            equity_curve.append(mtm)

    # jeżeli po pętli pozycja otwarta → zamknij po ostatniej cenie
    if position is not None:
        last_price = float(out["close"].iloc[-1])
        pnl = (last_price - float(entry_price)) * position * float(entry_qty)
        cost = risk.position_cost(float(entry_qty), last_price)
        account_equity += pnl - cost
        tb.add(
            Trade(out.index[-1], last_price, float(entry_qty), "LONG" if position == 1 else "SHORT")
        )
        position = entry_price = entry_qty = None
        equity_curve[-1] = account_equity  # ostatni punkt jako zrealizowany

    # Zapis krzywej kapitału
    out["equity"] = pd.Series(equity_curve, index=out.index, dtype=float)

    return out

