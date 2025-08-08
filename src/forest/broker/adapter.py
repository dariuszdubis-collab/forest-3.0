from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Protocol


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class Position:
    symbol: str
    side: Side
    qty: float
    entry: float  # average entry price


@dataclass(frozen=True)
class TradeResult:
    symbol: str
    qty: float
    side: Side
    price: float
    realized_pnl: float


class BrokerAdapter(Protocol):
    """Minimalny interfejs brokera, pod który podepniemy MT4/MT5."""

    def price(self, symbol: str) -> float: ...
    def market_order(
        self,
        symbol: str,
        side: Side,
        qty: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> TradeResult: ...
    def close_position(self, symbol: str) -> TradeResult: ...
    def positions(self) -> Dict[str, Position]: ...
    def balance(self) -> float: ...
    def equity(self) -> float: ...


class PaperBroker(BrokerAdapter):
    """Prosta implementacja papierowa (in-memory).

    Założenia:
    - jedna pozycja na symbol; zlecenia rynkowe wypełniają się po bieżącej cenie,
    - jeśli nowy sygnał jest przeciwny do istniejącej pozycji, najpierw realizujemy PnL,
      potem otwieramy nową pozycję,
    - średnia cena wejścia uaktualnia się przy dokładaniu tej samej strony.
    - brak prowizji/slippage (można dodać później).
    """

    def __init__(self, initial_balance: float = 0.0) -> None:
        self._balance: float = float(initial_balance)
        self._prices: Dict[str, float] = {}
        self._pos: Dict[str, Position] = {}

    # ---------- price feed (papierowy) ----------
    def update_price(self, symbol: str, price: float) -> None:
        self._prices[symbol] = float(price)

    def price(self, symbol: str) -> float:
        if symbol not in self._prices:
            raise ValueError(f"No price for symbol: {symbol}")
        return self._prices[symbol]

    # ---------- PnL helpers ----------
    @staticmethod
    def _pnl(pos: Position, exit_price: float) -> float:
        sign = 1.0 if pos.side == Side.BUY else -1.0
        return (exit_price - pos.entry) * sign * pos.qty

    # ---------- mandatory API ----------
    def market_order(
        self,
        symbol: str,
        side: Side,
        qty: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> TradeResult:
        del sl, tp  # placeholders; mini-adapter na razie ich nie używa

        px = self.price(symbol)
        cur = self._pos.get(symbol)

        if cur is None:
            self._pos[symbol] = Position(symbol=symbol, side=side, qty=float(qty), entry=px)
            return TradeResult(symbol, float(qty), side, px, realized_pnl=0.0)

        # ten sam kierunek => uśredniamy wejście
        if cur.side == side:
            new_qty = cur.qty + float(qty)
            new_entry = (cur.entry * cur.qty + px * float(qty)) / new_qty
            self._pos[symbol] = Position(symbol=symbol, side=side, qty=new_qty, entry=new_entry)
            return TradeResult(symbol, float(qty), side, px, realized_pnl=0.0)

        # przeciwny kierunek => zamknij starą, otwórz nową
        realized = self._pnl(cur, px)
        self._balance += realized
        self._pos[symbol] = Position(symbol=symbol, side=side, qty=float(qty), entry=px)
        return TradeResult(symbol, float(qty), side, px, realized_pnl=realized)

    def close_position(self, symbol: str) -> TradeResult:
        cur = self._pos.get(symbol)
        if cur is None:
            # nic do zamknięcia
            return TradeResult(symbol, 0.0, Side.BUY, self.price(symbol), realized_pnl=0.0)

        px = self.price(symbol)
        realized = self._pnl(cur, px)
        self._balance += realized
        del self._pos[symbol]
        # side w wyniku przyjmijmy kierunek zamykanej pozycji
        return TradeResult(symbol, cur.qty, cur.side, px, realized)

    def positions(self) -> Dict[str, Position]:
        return dict(self._pos)

    def balance(self) -> float:
        return self._balance

    def equity(self) -> float:
        unrealized = 0.0
        for sym, pos in self._pos.items():
            unrealized += self._pnl(pos, self.price(sym))
        return self._balance + unrealized


