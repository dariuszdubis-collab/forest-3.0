"""RiskManager – zarządzanie wielkością pozycji i ryzykiem.

v‑2:
* ATR‑position sizing  (jak w v‑1)
* globalny max DD guard
* trailing SL (Chandelier)
* symulacja kosztów transakcyjnych: spread + commission + slippage
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class RiskManager:
    capital: float
    risk_per_trade: float = 0.01         # 1 % equity
    max_drawdown: float = 0.20           # 20 % DD absolutny

    _equity_curve: list[float] | None = None
    _trail: float | None = None          # trailing SL

    # ------------------------------------------------------------------ #
    #  Position sizing                                                   #
    # ------------------------------------------------------------------ #
    def position_size(self, atr: float, atr_multiple: float = 2.0) -> float:
        """Lot size = (equity * risk%) / (ATR * atr_multiple)."""
        if atr <= 0:
            return 0.0
        dollar_risk = self.equity * self.risk_per_trade
        return dollar_risk / (atr * atr_multiple)

    # ------------------------------------------------------------------ #
    #  Trailing Stop (Chandelier)                                        #
    # ------------------------------------------------------------------ #
    def update_trailing_sl(self, price: float, atr: float, k: float = 3.0) -> None:
        """Podciąga trailing SL tylko w kierunku zysku."""
        new_trail = price - k * atr
        if self._trail is None or new_trail > self._trail:
            self._trail = new_trail

    def hit_trailing_sl(self, price: float) -> bool:
        """Czy cena dotknęła trailing SL?"""
        return self._trail is not None and price < self._trail

    # ------------------------------------------------------------------ #
    #  Transaction costs                                                 #
    # ------------------------------------------------------------------ #
    def position_cost(
        self,
        qty: float,
        price: float,
        spread: float = 0.0002,          # 2 pipette przy cenie 1.0000
        commission: float = 0.0005,      # 5 pipette round‑turn
        slippage: float = 0.0001,        # dodatkowy poślizg
    ) -> float:
        pct = spread + commission + slippage
        return qty * price * pct

    # ------------------------------------------------------------------ #
    #  Equity tracking                                                   #
    # ------------------------------------------------------------------ #
    def record_trade(self, pnl: float) -> None:
        if self._equity_curve is None:
            self._equity_curve = [self.capital + pnl]
        else:
            self._equity_curve.append(self._equity_curve[-1] + pnl)

    @property
    def equity(self) -> float:
        return self._equity_curve[-1] if self._equity_curve else self.capital

    def exceeded_max_dd(self) -> bool:
        if not self._equity_curve:
            return False
        series = pd.Series(self._equity_curve, dtype="float64")
        drawdown = series.cummax() - series
        return bool((drawdown / series.cummax()).max() >= self.max_drawdown)

