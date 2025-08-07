# src/forest/backtest/trace.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(slots=True, frozen=True)
class DecisionTrace:
    """
    Minimalistyczny „ślad decyzyjny” zapisywany w logach.

    * `time` – znacznik czasu świecy / ticka,
    * `symbol` – instrument,
    * `filters` – słownik flag pomocniczych (ATR_ok, trailing_hit, itp.),
    * `final` – końcowa decyzja (BUY/SELL/WAIT).
    """

    time: str
    symbol: str
    filters: Dict[str, Any]
    final: str

    # --------------------------------------------------------------------- helpers
    def as_dict(self) -> Dict[str, Any]:
        """Alias wstecznej kompatybilności – zwraca siebie jako `dict`."""
        return asdict(self)

    # Pydantic‑like API – używane w `engine.py`
    def model_dump(self) -> Dict[str, Any]:  # noqa: D401 – nazwa jak w Pydantic
        """Dump model to plain ``dict``."""
        return asdict(self)

    # Czytelny wpis w logach / debug print
    def __str__(self) -> str:  # noqa: D401 – krótka reprezentacja
        return (
            f"DecisionTrace(time={self.time}, symbol={self.symbol}, "
            f"final={self.final}, filters={self.filters})"
        )

