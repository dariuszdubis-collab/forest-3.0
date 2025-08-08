from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Protocol

Side = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class Order:
    symbol: str
    side: Side
    qty: float
    # Dla PaperBroker dopuszczamy podanie ceny fill (np. z ostatniego ticka).
    price: Optional[float] = None
    # Na razie tylko rynek, bez limit/stop.
    type: Literal["market"] = "market"


@dataclass(frozen=True)
class OrderResult:
    id: str
    status: Literal["filled", "rejected"]
    filled_qty: float
    avg_price: float
    error: Optional[str] = None


class OrderRouter(Protocol):
    """Minimalny interfejs routingu zleceń dla trybu 'live'."""

    def connect(self) -> None: ...
    def close(self) -> None: ...

    def market_order(self, order: Order) -> OrderResult: ...
    def position_qty(self, symbol: str) -> float: ...
    def set_price(self, symbol: str, price: float) -> None: ...
    def equity(self) -> float: ...


class PaperBroker:
    """
    Prosty broker papierowy:
    - prowadzi 'cash' oraz słownik pozycji,
    - fill na cenie podanej w Order.price lub ostatniej znanej (set_price),
    - prowizja jako procent od wartości transakcji,
    - tylko pozycje long (sprzedaż możliwa do wielkości pozycji).
    """

    _id_seq = itertools.count(1)

    def __init__(self, initial_cash: float = 0.0, fee_perc: float = 0.0) -> None:
        self._cash: float = float(initial_cash)
        self._fee: float = float(fee_perc)
        self._positions: Dict[str, float] = {}
        self._last_price: Dict[str, float] = {}
        self._connected: bool = False

    # --- interfejs ---

    def connect(self) -> None:
        self._connected = True

    def close(self) -> None:
        self._connected = False

    def set_price(self, symbol: str, price: float) -> None:
        self._last_price[symbol] = float(price)

    def position_qty(self, symbol: str) -> float:
        return float(self._positions.get(symbol, 0.0))

    def equity(self) -> float:
        mtm = sum(self.position_qty(sym) * self._last_price.get(sym, 0.0)
                  for sym in self._positions)
        return self._cash + mtm

    def market_order(self, order: Order) -> OrderResult:
        if not self._connected:
            return OrderResult(
                id="paper-0",
                status="rejected",
                filled_qty=0.0,
                avg_price=0.0,
                error="not_connected",
            )

        price = order.price if order.price is not None else self._last_price.get(order.symbol)
        if price is None:
            return OrderResult(
                id="paper-0",
                status="rejected",
                filled_qty=0.0,
                avg_price=0.0,
                error="no_price",
            )

        qty = float(order.qty)
        if qty <= 0:
            return OrderResult(
                id="paper-0",
                status="rejected",
                filled_qty=0.0,
                avg_price=0.0,
                error="invalid_qty",
            )

        cost = qty * price
        fee = cost * self._fee

        if order.side == "BUY":
            # kupno zmniejsza cash, zwiększa pozycję
            self._cash -= (cost + fee)
            self._positions[order.symbol] = self.position_qty(order.symbol) + qty
        else:  # SELL
            current = self.position_qty(order.symbol)
            if qty > current + 1e-12:
                return OrderResult(
                    id="paper-0",
                    status="rejected",
                    filled_qty=0.0,
                    avg_price=0.0,
                    error="insufficient_position",
                )
            self._cash += (cost - fee)
            self._positions[order.symbol] = current - qty
            if abs(self._positions[order.symbol]) < 1e-12:
                # czyścimy 'zerową' pozycję
                self._positions.pop(order.symbol, None)

        oid = f"paper-{next(self._id_seq)}"
        return OrderResult(
            id=oid,
            status="filled",
            filled_qty=qty,
            avg_price=price,
        )

