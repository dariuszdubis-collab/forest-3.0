from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List

import pandas as pd
from joblib import Parallel, delayed
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


# ---------------------------------------------------------------------------#
#  Wersja wektorowa + Parallel joblib                                        #
# ---------------------------------------------------------------------------#
def _single_run(df: pd.DataFrame, params: dict, make_risk: Callable[[], RiskManager]) -> GridResult:
    fast, slow = params["fast"], params["slow"]
    res = run_backtest(df, make_risk(), fast, slow)

    equity_end = res["equity"].iloc[-1]
    dd = (res["equity"].cummax() - res["equity"]) / res["equity"].cummax()
    return GridResult(params, float(equity_end), float(dd.max()))


def run_grid(
    df: pd.DataFrame,
    grid: Iterable[dict],
    make_risk: Callable[[], RiskManager] | None = None,
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Uruchamia back‑test w równoległych procesach.

    Parameters
    ----------
    df : DataFrame
        Dane OHLC.
    grid : iterable of dict
        Kombinacje parametrów.
    make_risk : callable
        Funkcja generująca świeży obiekt RiskManager dla każdego przebiegu.
    n_jobs : int
        Liczba procesów (‑1 = wszystkie CPU, 1 = bez multiprocessing).

    Returns
    -------
    DataFrame
        Kolumny: params, equity_end, max_dd
    """
    make_risk = make_risk or (lambda: RiskManager(capital=10_000))
    grid_list: List[dict] = list(grid)

    if n_jobs == 1:
        results = [_single_run(df, p, make_risk) for p in tqdm(grid_list, desc="Param grid")]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_single_run)(df, params, make_risk) for params in tqdm(grid_list, desc="Param grid")
        )

    return pd.DataFrame([asdict(r) for r in results])
