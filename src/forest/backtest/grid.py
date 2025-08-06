from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable

import pandas as pd
from tqdm.auto import tqdm

from forest.backtest.engine import run_backtest
from forest.backtest.risk import RiskManager


@dataclass(slots=True)
class GridResult:
    params: Dict[str, Any]
    equity_end: float
    max_dd: float


def param_grid(**param_ranges) -> Iterable[dict]:
    """Generator wszystkich kombinacji parametrów (`**param_ranges`)."""
    keys, values = zip(*param_ranges.items())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def run_grid(
    df: pd.DataFrame,
    grid: Iterable[dict],
    make_risk: Callable[[], RiskManager] | None = None,
) -> pd.DataFrame:
    """Uruchamia back‑test dla każdej kombinacji; zwraca DataFrame wyników."""
    records: list[GridResult] = []
    make_risk = make_risk or (lambda: RiskManager(capital=10_000))

    for params in tqdm(list(grid), desc="Param grid"):
        # w tej wersji zmieniamy tylko okresy EMA
        fast, slow = params["fast"], params["slow"]
        res = run_backtest(df, make_risk(), fast, slow)

        equity_end = res["equity"].iloc[-1]
        dd = (res["equity"].cummax() - res["equity"]) / res["equity"].cummax()
        records.append(GridResult(params, equity_end, dd.max()))

    return pd.DataFrame([asdict(r) for r in records])
