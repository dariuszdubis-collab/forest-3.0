"""
Typowana konfiguracja FOREST 3.0 (Pydantic v2).

Umożliwia:
- ładowanie z YAML/JSON/DICT,
- walidację pól,
- normalizację interwału czasowego (timeframe),
- wygodny dostęp do ustawień ryzyka i parametrów strategii.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class RiskSettings(BaseModel):
    capital: float = Field(10_000, ge=0, description="Kapitał początkowy")
    risk_per_trade: float = Field(0.01, ge=0, le=1, description="Ułamek kapitału na trade")
    max_drawdown: float = Field(0.2, ge=0, le=1, description="Maksymalny DD (np. 0.2 = 20%)")
    fee_perc: float = Field(0.0002, ge=0, description="Prowizja % od notional (w jedną stronę)")
    slippage: float = Field(0.0, ge=0, description="Dodatkowy poślizg ceny na wejście/wyjście")
    atr_multiple: float = Field(1.5, ge=0, description="Mnożnik ATR dla wyznaczenia wielkości pozycji")


class StrategySettings(BaseModel):
    mode: Literal["classic", "ml"] = "classic"
    fast: int = Field(12, ge=1)
    slow: int = Field(26, ge=1)
    atr_period: int = Field(14, ge=1)


class BacktestSettings(BaseModel):
    symbol: str = "SYN"
    timeframe: str = "1h"  # normalizujemy do postaci: 1m, 5m, 15m, 1h, 4h, 1d
    seed: int = 42

    risk: RiskSettings = RiskSettings()
    strategy: StrategySettings = StrategySettings()

    @field_validator("timeframe")
    @classmethod
    def _normalize_timeframe(cls, v: str) -> str:
        # Lazy import aby uniknąć cyklicznych importów
        from forest.utils.timeframes import normalize_timeframe

        return normalize_timeframe(v)

    # ---------- Ładowanie zapisanej konfiguracji ----------
    @classmethod
    def from_file(cls, path: str | Path) -> "BacktestSettings":
        """Wczytaj konfigurację z pliku YAML/JSON."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)

        if p.suffix.lower() in {".yml", ".yaml"}:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        elif p.suffix.lower() == ".json":
            import json

            data = json.loads(p.read_text(encoding="utf-8"))
        else:
            raise ValueError(f"Nieznane rozszerzenie pliku konfig: {p.suffix}")

        if not isinstance(data, dict):
            raise ValueError("Konfiguracja musi być słownikiem (mapą klucz→wartość).")

        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

