from __future__ import annotations
from typing import Any, Dict, Optional, Mapping
from pydantic import BaseModel, Field, field_validator
import json, yaml

from forest.utils.timeframes import normalize_timeframe
from forest.strategy.ema_cross import EMACrossStrategy
from forest.strategy.base import Strategy


class RiskSettings(BaseModel):
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.01
    max_drawdown: float = 0.25
    fee_perc: float = 0.0005  # 5 bps

class StrategySettings(BaseModel):
    name: str = "ema_cross"
    params: Dict[str, Any] = Field(default_factory=lambda: {"fast": 12, "slow": 26})
    price_col: str = "close"

class BacktestSettings(BaseModel):
    symbol: str = "SYMBOL"
    timeframe: str = "1h"
    strategy: StrategySettings = StrategySettings()
    risk: RiskSettings = RiskSettings()
    atr_period: int = 14
    atr_multiple: float = 2.0

    @field_validator("timeframe")
    @classmethod
    def _normalize_tf(cls, v: str) -> str:
        return normalize_timeframe(v)

    @classmethod
    def from_file(cls, path: str) -> "BacktestSettings":
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".json"):
                data = json.load(f)
            else:
                data = yaml.safe_load(f)
        return cls(**data)

    def build_strategy(self) -> Strategy:
        if self.strategy.name.lower() in ("ema", "ema_cross", "ema-cross"):
            return EMACrossStrategy(**self.strategy.params)
        raise ValueError(f"Unknown strategy: {self.strategy.name}")

