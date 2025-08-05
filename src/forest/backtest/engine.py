from __future__ import annotations

import numpy as np
import pandas as pd

from forest.backtest.risk import RiskManager
from forest.backtest.tradebook import Trade, TradeBook
from forest.core.indicators import atr, ema


# ----------------------------------------------------------------------
# 1. Strategia testowa – przecięcie dwóch EMA
# ----------------------------------------------------------------------
def ema_cross_strategy(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.Series:
    """
    Jeżeli EMA(fast) > EMA(slow)   → sygnał +1 (LONG)
    Jeżeli EMA(fast) < EMA(slow)   → sygnał -1 (SHORT)
    0 przed wypełnieniem historii.
    """
    fast_ma = ema(df["close"].to_numpy(), fast)
    slow_ma = ema(df["close"].to_numpy(), slow)

    signal = np.where(fast_ma > slow_ma, 1, -1)
    signal[: slow] = 0          # brak sygnału, zanim slow‑EMA się wypełni

    return pd.Series(signal, index=df.index, name="signal")


# ----------------------------------------------------------------------
# 2. Silnik back‑testu (v‑0, pętla po świecach)
# ----------------------------------------------------------------------
def run_backtest(df: pd.DataFrame, risk: RiskManager) -> pd.DataFrame:
    """
    Uruchom back‑test na DataFrame świec z kolumnami
    open, high, low, close (w dowolnym interwale).

    Zwraca DataFrame z dodatkowymi kolumnami:
        signal  – 1 / -1 / 0
        atr     – ATR użyty do sizingu
        equity  – krzywa kapitału (narastająco, ffill)
    """
    out = df.copy()

    # --------- 2.1 sygnał z EMA‑cross ---------------------------------
    out["signal"] = ema_cross_strategy(out)

    # --------- 2.2 ATR do position sizingu ----------------------------
    out["atr"] = atr(out["high"], out["low"], out["close"], period=14)

    tb = TradeBook()

    # --------- 2.3 główna pętla po świecach ----------------------------
    for idx, row in out.iterrows():
        if row.signal == 0:
            continue

        qty = risk.position_size(row.atr)          # kalkulacja lotów
        if qty == 0:
            continue

        side = "LONG" if row.signal == 1 else "SHORT"
        tb.add(Trade(time=idx, price=row.close, qty=qty, side=side))

        # Aktualizacja equity w RiskManager
        pnl = (+1 if side == "LONG" else -1) * qty * row.close
        risk.record_trade(pnl)

        # Stop trading przy przekroczeniu max DD
        if risk.exceeded_max_dd():
            break

    # --------- 2.4 krzywa kapitału do DataFrame ------------------------
    out["equity"] = tb.equity_curve().reindex(out.index).ffill()

    return out

