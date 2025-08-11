from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional
import pandas as pd


@dataclass
class Signal:
    """Prosty sygnał strategii."""
    action: str  # "BUY", "SELL" lub "HOLD"
    reason: Optional[str] = None


class Strategy(ABC):
    """Abstrakcyjna baza strategii."""
    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def init(self, df: pd.DataFrame) -> pd.DataFrame:
        """Przygotuj kolumny/cechy w df (np. wskaźniki). Zwraca df z dodatkami."""
        raise NotImplementedError

    @abstractmethod
    def on_bar(self, row: pd.Series) -> Signal:
        """Zwróć sygnał dla pojedynczej świecy."""
        raise NotImplementedError

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "Strategy":
        return cls(**cfg)

