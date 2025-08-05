from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

Decision = Literal["BUY", "SELL", "WAIT"]

@dataclass(slots=True)
class DecisionTrace:
    time: str
    symbol: str
    filters: Dict[str, bool]
    final: Decision
