from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class RiskManager:
    capital: float               # początkowy equity
    risk_per_trade: float = 0.01 # 1 % konta
    max_drawdown: float = 0.20   # 20 % absolutne obsunięcie

    _equity_curve: list[float] = None

    # ---------- sizing ----------
    def position_size(self, atr: float, atr_multiple: float = 2.0) -> float:
        """
        Lot size (qty) = (capital * risk%) / (ATR * atr_multiple)
        """
        dollar_risk = self.capital * self.risk_per_trade
        if atr == 0:
            return 0.0
        return dollar_risk / (atr * atr_multiple)

    # ---------- equity tracking ----------
    def record_trade(self, pnl: float) -> None:
        """
        Aktualizuj equity o zrealizowany PnL (plusowy lub minusowy).
        """
        if self._equity_curve is None:
            self._equity_curve = [self.capital + pnl]
        else:
            self._equity_curve.append(self._equity_curve[-1] + pnl)

    @property
    def equity(self) -> float:
        if self._equity_curve:
            return self._equity_curve[-1]
        return self.capital

    def exceeded_max_dd(self) -> bool:
        if not self._equity_curve:
            return False
        series = pd.Series(self._equity_curve)
        drawdown = series.cummax() - series
        return bool((drawdown / series.cummax()).max() >= self.max_drawdown)
