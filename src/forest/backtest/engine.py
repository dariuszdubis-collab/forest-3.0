from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from forest.results import BacktestResult
from forest.strategy.base import Strategy
from forest.utils.validate import ensure_backtest_ready
from forest.utils.log import log
from forest.backtest.tradebook import TradeBook, Trade
from forest.backtest.risk import RiskManager
from forest.backtest.trace import DecisionTrace
from forest.core.indicators import atr


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: Optional[RiskManager] = None,
    symbol: str = "SYMBOL",
    price_col: str = "close",
    atr_period: int = 14,
    atr_multiple: float = 2.0,
) -> BacktestResult:
    """
    Główny backtest: data -> (strategy -> sygnały) -> egzekucja -> wynik.
    """
    # 1) Sanity & przygotowanie danych
    df = ensure_backtest_ready(df, price_col=price_col).copy()
    # 2) Inicjalizacja strategii (obliczenia wektorowe)
    df = strategy.init(df)

    # ATR dla kontroli ryzyka / SL (opcjonalnie używany przez RiskManager)
    atr_vals = atr(df["high"].to_numpy(), df["low"].to_numpy(), df[price_col].to_numpy(), length=atr_period)
    df["_atr"] = atr_vals

    # 3) Stan portfela / księga transakcji
    tb = TradeBook()
    risk = risk or RiskManager()
    equity_curve: list[float] = []
    pos_qty = 0.0
    entry_price = None  # dla śledzenia SL (opcjonalnie)

    # 4) Pętla po świecach
    for ts, row in df.iterrows():
        price = float(row[price_col])
        signal = strategy.on_bar(row)

        # trailing stop (opcjonalnie) – przykładowe użycie ATR
        if pos_qty > 0 and np.isfinite(row["_atr"]):
            risk.update_trailing_sl(entry_price=entry_price or price, atr=row["_atr"], atr_multiple=atr_multiple)

        action = signal.action
        trace = DecisionTrace(
            time=ts,
            symbol=symbol,
            filters={"atr_warmup": not np.isfinite(row["_atr"])},
            final=action if action in ("BUY", "SELL") else "WAIT",
        )
        log.info("decision", **{
            "time": str(ts), "symbol": symbol, "action": action, "reason": signal.reason or "", "price": price
        })

        # SL trafiony?
        if pos_qty > 0 and risk.hit_trailing_sl(price):
            # zamknięcie pozycji
            tb.add(Trade(time=ts, price=price, qty=pos_qty, side="SELL"))
            risk.record_trade(entry=entry_price or price, exit=price, qty=pos_qty)
            pos_qty = 0.0
            entry_price = None

        # sygnał zmiany
        if action == "BUY" and pos_qty == 0:
            qty = risk.position_size(price=price, atr=row["_atr"])
            if qty > 0:
                tb.add(Trade(time=ts, price=price, qty=qty, side="BUY"))
                pos_qty = qty
                entry_price = price
                risk.reset_trailing_sl(entry_price=entry_price, atr=row["_atr"], atr_multiple=atr_multiple)

        elif action == "SELL" and pos_qty > 0:
            tb.add(Trade(time=ts, price=price, qty=pos_qty, side="SELL"))
            risk.record_trade(entry=entry_price or price, exit=price, qty=pos_qty)
            pos_qty = 0.0
            entry_price = None

        equity_curve.append(risk.equity)

        if risk.exceeded_max_dd():
            log.warning("max_dd_exceeded", time=str(ts))
            break

    # 5) Zamknij pozycję na końcu (jeśli otwarta)
    if pos_qty > 0:
        last_ts = df.index[-1]
        last_price = float(df.iloc[-1][price_col])
        tb.add(Trade(time=last_ts, price=last_price, qty=pos_qty, side="SELL"))
        risk.record_trade(entry=entry_price or last_price, exit=last_price, qty=pos_qty)
        pos_qty = 0.0
        entry_price = None
        equity_curve.append(risk.equity)

    # 6) Wyniki
    equity = pd.Series(equity_curve, index=df.index[:len(equity_curve)], name="equity")
    trades_df = tb.to_frame()
    return BacktestResult.from_equity_and_trades(equity=equity, trades=trades_df)

