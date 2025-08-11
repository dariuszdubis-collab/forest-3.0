from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def _max_drawdown(equity: pd.Series) -> float:
    cummax = equity.cummax()
    dd = (equity / cummax) - 1.0
    return float(dd.min()) * -1.0  # relatywny max DD (0..1)

def _daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()

def _cagr(equity: pd.Series, dates: pd.DatetimeIndex) -> float:
    if len(equity) < 2:
        return 0.0
    years = (dates[-1] - dates[0]).days / 365.25
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)

def _sharpe(equity: pd.Series, risk_free: float = 0.0) -> float:
    rets = _daily_returns(equity)
    if rets.empty:
        return 0.0
    excess = rets - risk_free / 252.0
    if excess.std(ddof=1) == 0:
        return 0.0
    # uproszczony Sharpe z annualizacją
    return float((excess.mean() / excess.std(ddof=1)) * np.sqrt(252.0))


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame  # kolumny: time, price, qty, side
    metrics: Dict[str, float]
    meta: Optional[Dict] = None

    @classmethod
    def from_equity_and_trades(cls, equity: pd.Series, trades: pd.DataFrame) -> "BacktestResult":
        dd = _max_drawdown(equity)
        cagr = _cagr(equity, equity.index)
        shp = _sharpe(equity)
        metrics = {
            "equity_end": float(equity.iloc[-1]) if len(equity) else 0.0,
            "max_dd": dd,     # 0..1
            "cagr": cagr,     # rocznie
            "sharpe": shp,
            "rar": (cagr / dd) if dd > 0 else 0.0,  # risk-adjusted return
        }
        return cls(equity=equity, trades=trades, metrics=metrics)

    def to_dict(self) -> Dict:
        d = {
            "equity": self.equity.reset_index().to_dict(orient="list"),
            "trades": self.trades.to_dict(orient="records"),
            "metrics": self.metrics,
            "meta": self.meta or {},
        }
        return d

