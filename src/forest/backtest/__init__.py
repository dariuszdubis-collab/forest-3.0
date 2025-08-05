"""
Sub‑pakiet `forest.backtest`
===========================

Udostępnia najważniejsze klasy i funkcje back‑testu, tak aby można
było je importować jednym krótkim łańcuchem:

    from forest.backtest import Trade, TradeBook, RiskManager, run_backtest
"""

from .engine import run_backtest
from .risk import RiskManager
from .tradebook import Trade, TradeBook

__all__: list[str] = [
    "Trade",
    "TradeBook",
    "RiskManager",
    "run_backtest",
]

