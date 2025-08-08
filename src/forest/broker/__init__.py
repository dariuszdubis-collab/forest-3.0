"""Broker mini-adapter.

Ujednolicony interfejs brokera oraz prosta implementacja papierowa
do lokalnych testów i symulacji zamówień bez realnego API.
"""

from .adapter import BrokerAdapter, PaperBroker, Position, Side, TradeResult

__all__ = [
    "Side",
    "Position",
    "TradeResult",
    "BrokerAdapter",
    "PaperBroker",
]

