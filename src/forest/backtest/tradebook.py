from __future__ import annotations
from dataclasses import dataclass
from typing import List, Iterator
import pandas as pd


@dataclass
class Trade:
    time: pd.Timestamp
    price: float
    qty: float
    side: str  # "BUY" lub "SELL"


class TradeBook:
    """Prosta księga transakcji do rejestrowania wykonanych zleceń."""
    def __init__(self) -> None:
        self._trades: List[Trade] = []

    def add(self, trade: Trade) -> None:
        self._trades.append(trade)

    def __len__(self) -> int:
        return len(self._trades)

    def __iter__(self) -> Iterator[Trade]:
        return iter(self._trades)

    def to_frame(self) -> pd.DataFrame:
        """Zwraca transakcje jako DataFrame: time, price, qty, side."""
        if not self._trades:
            return pd.DataFrame(columns=["time", "price", "qty", "side"])
        rows = [
            {"time": t.time, "price": float(t.price), "qty": float(t.qty), "side": t.side}
            for t in self._trades
        ]
        return pd.DataFrame(rows)

    # poniższe metody nie są wymagane przez engine 4.0, ale zostawiamy dla kompatybilności:
    def equity_curve(self, initial: float = 0.0) -> pd.Series:
        """
        Prosty equity oparty o ZREALIZOWANY PnL (aktualizacja przy SELL).
        Służy jako pomocnicza metryka – engine 4.0 śledzi equity w RiskManager.
        """
        eq_vals: List[float] = [initial]
        idx: List[pd.Timestamp] = []

        pos_qty = 0.0
        pos_price = None

        for t in self._trades:
            if t.side.upper() == "BUY" and pos_qty == 0.0:
                pos_qty = float(t.qty)
                pos_price = float(t.price)
            elif t.side.upper() == "SELL" and pos_qty > 0.0:
                pnl = (float(t.price) - float(pos_price or t.price)) * float(pos_qty)
                eq_vals.append(eq_vals[-1] + pnl)
                idx.append(pd.to_datetime(t.time))
                pos_qty = 0.0
                pos_price = None

        if not idx:
            # brak zamkniętych pozycji – zwróć jednopunktową serię
            return pd.Series([initial], index=[pd.NaT], name="equity_realized")

        eq = pd.Series(eq_vals[1:], index=idx, name="equity_realized")
        return eq

    def max_drawdown(self) -> float:
        eq = self.equity_curve(initial=0.0)
        if eq.empty:
            return 0.0
        peak = eq.cummax()
        dd = (eq / peak) - 1.0
        return float(-dd.min()) if not dd.empty else 0.0

