from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import itertools
import pandas as pd

from forest.backtest.engine import run_backtest
from forest.results import BacktestResult
from forest.config import BacktestSettings


@dataclass
class GridResult:
    params: Dict
    equity_end: float
    max_dd: float
    cagr: float
    rar: float
    sharpe: float


def param_grid(ranges: Dict[str, Iterable]) -> List[Dict]:
    keys = list(ranges.keys())
    combos = list(itertools.product(*[ranges[k] for k in keys]))
    return [dict(zip(keys, vals)) for vals in combos]


def _single_run(data: pd.DataFrame, base_cfg: BacktestSettings, strategy_params: Dict) -> GridResult:
    cfg = base_cfg.model_copy(deep=True)
    cfg.strategy.params.update(strategy_params)
    strat = cfg.build_strategy()
    res: BacktestResult = run_backtest(
        df=data,
        strategy=strat,
        symbol=cfg.symbol,
        price_col=cfg.strategy.price_col,
        atr_period=cfg.atr_period,
        atr_multiple=cfg.atr_multiple,
    )
    m = res.metrics
    return GridResult(
        params=strategy_params,
        equity_end=m["equity_end"],
        max_dd=m["max_dd"],
        cagr=m["cagr"],
        rar=m["rar"],
        sharpe=m["sharpe"],
    )


def run_grid(
    data: pd.DataFrame,
    base_cfg: BacktestSettings,
    ranges: Dict[str, Iterable],
) -> pd.DataFrame:
    combos = param_grid(ranges)
    results = [_single_run(data, base_cfg, p) for p in combos]
    df = pd.DataFrame([{
        **r.params,
        "equity_end": r.equity_end,
        "max_dd": r.max_dd,
        "cagr": r.cagr,
        "rar": r.rar,
        "sharpe": r.sharpe,
    } for r in results])
    return df

