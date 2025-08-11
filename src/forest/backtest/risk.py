from __future__ import annotations
from typing import Optional


class RiskManager:
    """
    Zarządzanie ryzykiem i krzywą kapitału.
    - position_size: wielkość pozycji wg: (risk% * equity) / (ATR * atr_multiple)
    - trailing SL: reset/update/hit
    - record_trade: aktualizacja equity po zamknięciu pozycji, z kosztami transakcyjnymi
    """
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        risk_per_trade: float = 0.01,
        max_drawdown: float = 0.25,
        fee_perc: float = 0.0005,        # prowizja 5 bps
        spread_perc: float = 0.0,
        slippage_perc: float = 0.0,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.risk_per_trade = float(risk_per_trade)
        self.max_drawdown = float(max_drawdown)
        self.fee_perc = float(fee_perc)
        self.spread_perc = float(spread_perc)
        self.slippage_perc = float(slippage_perc)

        self._equity_curve = [self.initial_capital]
        self._trail: Optional[float] = None  # poziom trailing stop dla long

    # --- Equity / DD ---

    @property
    def equity(self) -> float:
        return float(self._equity_curve[-1])

    def exceeded_max_dd(self) -> bool:
        peak = max(self._equity_curve) if self._equity_curve else self.initial_capital
        if peak <= 0:
            return False
        dd = (peak - self.equity) / peak
        return dd >= (self.max_drawdown - 1e-12)

    # --- Koszty / pozycja ---

    def position_cost(self, notional: float) -> float:
        """Łączny koszt transakcyjny jako % wartości zlecenia (fee+spread+slippage)."""
        rate = self.fee_perc + self.spread_perc + self.slippage_perc
        return float(notional) * float(rate)

    def position_size(self, price: float, atr: Optional[float], atr_multiple: float = 2.0) -> float:
        """
        Wielkość pozycji long: (risk% * equity) / (ATR * atr_multiple)
        Zwraca 0.0, jeśli ATR niegotowy lub parametry niepoprawne.
        """
        if atr is None or not (atr > 0) or price <= 0:
            return 0.0
        risk_dollars = self.risk_per_trade * self.equity
        risk_per_unit = float(atr_multiple) * float(atr)
        if risk_per_unit <= 0:
            return 0.0
        qty = risk_dollars / risk_per_unit
        return float(qty) if qty > 0 else 0.0

    # --- Trailing stop (long) ---

    def reset_trailing_sl(self, current_price: float, atr: Optional[float], atr_multiple: float = 2.0) -> None:
        """Ustaw początkowy trailing SL w oparciu o cenę i ATR."""
        if atr is None or not (atr > 0):
            self._trail = None
            return
        self._trail = float(current_price) - float(atr_multiple) * float(atr)

    def update_trailing_sl(self, current_price: float, atr: Optional[float], atr_multiple: float = 2.0) -> None:
        """Podnoś trailing SL tylko w górę, gdy cena idzie na naszą korzyść (long)."""
        if atr is None or not (atr > 0):
            return
        candidate = float(current_price) - float(atr_multiple) * float(atr)
        if self._trail is None:
            self._trail = candidate
        else:
            self._trail = max(self._trail, candidate)

    def hit_trailing_sl(self, price: float) -> bool:
        """Czy aktualna cena przebiła trailing SL (dla pozycji long)?"""
        return self._trail is not None and float(price) <= float(self._trail)

    # --- Rejestracja transakcji ---

    def record_trade(self, entry: float, exit: float, qty: float) -> None:
        """
        Zapisz realizację transakcji (zamknięcie pozycji) i zaktualizuj equity (z kosztami).
        Koszty naliczane po obu stronach: entry i exit.
        """
        gross = (float(exit) - float(entry)) * float(qty)
        cost = self.position_cost(float(entry) * float(qty)) + self.position_cost(float(exit) * float(qty))
        pnl = gross - cost
        self._equity_curve.append(self.equity + pnl)
        # po zamknięciu pozycji resetujemy trailing
        self._trail = None

