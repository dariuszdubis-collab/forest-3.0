from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

import pandas as pd

Side = Literal["LONG", "SHORT"]


@dataclass(slots=True, frozen=True)
class Trade:
    """Jedna wykonana transakcja."""

    time: pd.Timestamp
    price: float
    qty: float
    side: Side  # "LONG" → kupno,  "SHORT" → sprzedaż


@dataclass
class TradeBook:
    """Prosty rejestr transakcji + metryki equity."""

    _trades: List[Trade] = field(default_factory=list)

    # ---------- API użytkowe ----------
    def add(self, trade: Trade) -> None:
        self._trades.append(trade)

    # ---------- Analizy ----------
    def equity_curve(self) -> pd.Series:
        """Kapitał narastająco po każdej transakcji (brutto, bez kosztów)."""
        pnl = []
        for t in self._trades:
            sign = 1 if t.side == "LONG" else -1
            pnl.append(sign * t.qty * t.price)
        return pd.Series(pnl, index=[t.time for t in self._trades]).cumsum()

    def max_drawdown(self) -> float:
        """Maksymalne obsunięcie kapitału (wartość dodatnia)."""
        eq = self.equity_curve()
        peak = eq.cummax()
        drawdown = peak - eq
        return drawdown.max() if not drawdown.empty else 0.0

